"""Trimmed bert_score utils, patched for transformers v5 compatibility."""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from math import log
from multiprocessing import Pool

import torch
from packaging import version
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import __version__ as trans_version


def _is_roberta_or_gpt2(tokenizer) -> bool:
    """Detect RoBERTa/GPT-2 family tokenizers in a way that works on v5.

    Upstream bert-score used `isinstance(tokenizer, (GPT2Tokenizer, RobertaTokenizer))`,
    but those classes are not importable under transformers v5. Duck-typing by
    class name covers both the slow (v4 PythonBackend) and fast
    (TokenizersBackend) variants across versions.
    """
    name = type(tokenizer).__name__
    return (
        "GPT2" in name
        or "Roberta" in name
        or "RoBERTa" in name
        or "BART" in name
        or "Bart" in name
    )


def sent_encode(tokenizer, sent: str):
    """Encode a sentence, preserving upstream's RoBERTa-family special casing."""
    sent = sent.strip()
    if sent == "":
        # Upstream called tokenizer.build_inputs_with_special_tokens([]) here.
        # That method was removed from TokenizersBackend in transformers v5, so
        # we use tokenizer.encode("") instead — which goes through the fast
        # Rust backend and returns exactly the special-token-only sequence.
        return tokenizer.encode("", add_special_tokens=True)

    max_length = tokenizer.model_max_length
    # NOTE: RoBERTa / GPT-2 / BART tokenizers already have add_prefix_space=True
    # baked in at from_pretrained time (see `get_tokenizer` above). Passing
    # it here again would be a no-op on transformers v5 fast tokenizers, so
    # we keep a single encode() path for every model family.
    return tokenizer.encode(
        sent,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )


def get_model(model_type: str, num_layers: int, all_layers: bool = False):
    if "t5" in model_type:
        from transformers import T5EncoderModel
        model = T5EncoderModel.from_pretrained(model_type)
    else:
        model = AutoModel.from_pretrained(model_type)
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    if not all_layers:
        if hasattr(model, "n_layers"):  # xlm
            assert 0 <= num_layers <= model.n_layers
            model.n_layers = num_layers
        elif hasattr(model, "layer"):  # xlnet
            assert 0 <= num_layers <= len(model.layer)
            model.layer = torch.nn.ModuleList(list(model.layer[:num_layers]))
        elif hasattr(model, "encoder"):  # albert / bert / roberta / modernbert
            if hasattr(model.encoder, "albert_layer_groups"):
                assert 0 <= num_layers <= model.encoder.config.num_hidden_layers
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):  # t5
                assert 0 <= num_layers <= len(model.encoder.block)
                model.encoder.block = torch.nn.ModuleList(
                    list(model.encoder.block[:num_layers])
                )
            else:  # bert / roberta / modernbert
                assert 0 <= num_layers <= len(model.encoder.layer)
                model.encoder.layer = torch.nn.ModuleList(
                    list(model.encoder.layer[:num_layers])
                )
        elif hasattr(model, "transformer"):  # distilbert
            assert 0 <= num_layers <= len(model.transformer.layer)
            model.transformer.layer = torch.nn.ModuleList(
                list(model.transformer.layer[:num_layers])
            )
        elif hasattr(model, "layers"):  # bart
            assert 0 <= num_layers <= len(model.layers)
            model.layers = torch.nn.ModuleList(list(model.layers[:num_layers]))
        else:
            raise ValueError(
                f"Cannot truncate layers on model of type {type(model).__name__}"
            )
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True

    return model


def _needs_prefix_space(model_type: str) -> bool:
    """Return True for RoBERTa / GPT-2 / BART family tokenizers.

    Upstream bert-score (0.3.13) passed `add_prefix_space=True` at each
    `encode(...)` call site for these families (see `sent_encode` below).
    On transformers >= 4.44 that kwarg path was closed at the fast-tokenizer
    `encode()` layer; passing `add_prefix_space` at `from_pretrained(...)` is
    the supported replacement. Doing so preserves bit-exact id output
    against the legacy slow RoBERTa tokenizer, which matters for
    reference-anchored metrics like RadCliQ-v1. See issue notes in
    RadEval/metrics/bertscore/_vendor/__init__.py.
    """
    lowered = (model_type or "").lower()
    return any(
        marker in lowered
        for marker in ("roberta", "gpt2", "gpt-2", "bart")
    )


def get_tokenizer(model_type: str, use_fast: bool = True):
    # In transformers v5 slow tokenizers are gone, so use_fast is effectively
    # always True. We keep the argument for API compatibility.
    kwargs = {}
    if _needs_prefix_space(model_type):
        # Loading RoBERTa/GPT-2/BART with add_prefix_space=True at
        # from_pretrained time reproduces the legacy slow-tokenizer output
        # (the "Ġ"-prefixed first token) that upstream bert-score relied on
        # via its `encode(..., add_prefix_space=True)` calls. Required for
        # numerical parity with the transformers 4.x reference.
        kwargs["add_prefix_space"] = True
    return AutoTokenizer.from_pretrained(model_type, use_fast=use_fast, **kwargs)


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask


def bert_encode(model, x, attention_mask, all_layers=False):
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask, output_hidden_states=all_layers)
    if all_layers:
        emb = torch.stack(out[-1], dim=2)
    else:
        emb = out[0]
    return emb


def _process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads: int = 4):
    """Inverse document frequency over word-piece ids."""
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(_process, tokenizer=tokenizer)

    if nthreads > 0:
        with Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))
    else:
        idf_count.update(chain.from_iterable(map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / 1))
    idf_dict.update(
        {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )
    return idf_dict


def collate_idf(arr, tokenizer, idf_dict, device="cuda:0"):
    arr = [sent_encode(tokenizer, a) for a in arr]
    idf_weights = [[idf_dict[i] for i in a] for a in arr]
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = 0
    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)
    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask


def get_bert_embedding(
    all_sens,
    model,
    tokenizer,
    idf_dict,
    batch_size: int = -1,
    device: str = "cuda:0",
    all_layers: bool = False,
):
    padded_sens, padded_idf, lens, mask = collate_idf(
        all_sens, tokenizer, idf_dict, device=device
    )
    if batch_size == -1:
        batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(
                model,
                padded_sens[i : i + batch_size],
                attention_mask=mask[i : i + batch_size],
                all_layers=all_layers,
            )
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)
    return total_embedding, mask, padded_idf


def greedy_cos_idf(
    ref_embedding,
    ref_masks,
    ref_idf,
    hyp_embedding,
    hyp_masks,
    hyp_idf,
    all_layers: bool = False,
):
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    if all_layers:
        B, _, L, D = hyp_embedding.size()
        hyp_embedding = (
            hyp_embedding.transpose(1, 2)
            .transpose(0, 1)
            .contiguous()
            .view(L * B, hyp_embedding.size(1), D)
        )
        ref_embedding = (
            ref_embedding.transpose(1, 2)
            .transpose(0, 1)
            .contiguous()
            .view(L * B, ref_embedding.size(1), D)
        )
    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    if all_layers:
        masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(sim)
    else:
        masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)
    if all_layers:
        precision_scale = (
            precision_scale.unsqueeze(0)
            .expand(L, B, -1)
            .contiguous()
            .view_as(word_precision)
        )
        recall_scale = (
            recall_scale.unsqueeze(0)
            .expand(L, B, -1)
            .contiguous()
            .view_as(word_recall)
        )
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
    ref_zero_mask = ref_masks.sum(dim=1).eq(2)

    if all_layers:
        P = P.view(L, B)
        R = R.view(L, B)
        F = F.view(L, B)

    if torch.any(hyp_zero_mask):
        print(
            "Warning: Empty candidate sentence detected; setting raw BERTscores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(hyp_zero_mask, 0.0)
        R = R.masked_fill(hyp_zero_mask, 0.0)
    if torch.any(ref_zero_mask):
        print(
            "Warning: Empty reference sentence detected; setting raw BERTScores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(ref_zero_mask, 0.0)
        R = R.masked_fill(ref_zero_mask, 0.0)

    F = F.masked_fill(torch.isnan(F), 0.0)
    return P, R, F


def bert_cos_score_idf(
    model,
    refs,
    hyps,
    tokenizer,
    idf_dict,
    verbose: bool = False,
    batch_size: int = 64,
    device: str = "cuda:0",
    all_layers: bool = False,
):
    preds = []

    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)

    sentences = dedup_and_sort(refs + hyps)
    iter_range = range(0, len(sentences), batch_size)
    if verbose:
        print("computing bert embedding.")
        iter_range = tqdm(iter_range)
    stats_dict: dict = {}
    for batch_start in iter_range:
        sen_batch = sentences[batch_start : batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            sen_batch, model, tokenizer, idf_dict, device=device, all_layers=all_layers
        )
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()
        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            stats_dict[sen] = (emb, idf)

    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(device) for e in emb]
        idf = [i.to(device) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad

    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)
    if verbose:
        print("computing greedy matching.")
        iter_range = tqdm(iter_range)

    with torch.no_grad():
        for batch_start in iter_range:
            batch_refs = refs[batch_start : batch_start + batch_size]
            batch_hyps = hyps[batch_start : batch_start + batch_size]
            ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
            hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)

            P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())
    preds = torch.cat(preds, dim=1 if all_layers else 0)
    return preds


VERSION = "0.3.13-radeval"
