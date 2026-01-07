import os
import re
import time
import warnings
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

import multiprocessing as mp
import sys

# Set start method safely (no-op if already set by pytest/another module)
try:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
except RuntimeError:
    # start method already set in this interpreter; ignore
    pass

from RadEval.factual.green_score.utils import (
    make_prompt,
    clean_responses,
)

# Suppress benign warnings from transformers
logging.get_logger("transformers").setLevel(logging.ERROR)


def _worker_generate(
    worker_id: int,
    device: str,
    model_name: str,
    prompts: List[str],
    batch_size: int,
    max_length: int,
    padding_side: str = "left",
    progress_queue=None,
) -> List[str]:
    """
    Runs entirely on one GPU/CPU: loads the model+tokenizer on `device`,
    generates completions for `prompts` in batches, returns cleaned strings.
    """
    torch.set_grad_enabled(False)

    # Load model/tokenizer on the specific device
    # Use float16 when on CUDA; fallback to float32 on CPU
    use_cuda = device.startswith("cuda")
    torch_dtype = torch.float16 if use_cuda else torch.float32

    # Some models don't support trust_remote_code=False; mirror your logic
    trust_remote_code = False if "Phi" in model_name else True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        device_map={"": device} if use_cuda else {"": "cpu"},
        torch_dtype=torch_dtype,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        use_fast=True,
        trust_remote_code=True,
        padding_side=padding_side,
    )
    # Chat template (same as yours)
    chat_template = (
        "{% for message in messages %}\n"
        "{% if message['from'] == 'human' %}\n"
        "{{ '<|user|>\\n' + message['value'] + eos_token }}\n"
        "{% elif message['from'] == 'system' %}\n"
        "{{ '<|system|>\\n' + message['value'] + eos_token }}\n"
        "{% elif message['from'] == 'gpt' %}\n"
        "{{ '<|assistant|>\\n'  + message['value'] + eos_token }}\n"
        "{% endif %}\n"
        "{% if loop.last and add_generation_prompt %}\n"
        "{{ '<|assistant|>' }}\n"
        "{% endif %}\n"
        "{% endfor %}"
    )
    tokenizer.chat_template = chat_template
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.clean_up_tokenization_spaces = True
    assert tokenizer.padding_side == "left"

    completions = []
    # Local loop with a quiet tqdm to keep logs tidy across workers
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        chats = [
            [{"from": "human", "value": p}, {"from": "gpt", "value": ""}]
            for p in batch_prompts
        ]
        texts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=True)
            for c in chats
        ]

        toks = tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if use_cuda:
            toks = {k: v.to(device) for k, v in toks.items()}

        outputs = model.generate(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_length=max_length,
            do_sample=False,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if isinstance(decoded, list):
            completions.extend([clean_responses(r) for r in decoded])
        else:
            completions.append(clean_responses(decoded))
        if progress_queue is not None:
            progress_queue.put(len(batch_prompts))
    return completions


class GREEN:
    def __init__(
        self,
        model_name=None,
        output_dir=".",
        cpu=False,
        num_gpus=None,  # NEW: set None to auto-detect (capped at 8)
    ):
        super().__init__()
        warnings.filterwarnings(
            "ignore", message="A decoder-only architecture is being used*"
        )

        self.cpu = cpu
        self.output_dir = output_dir
        self.batch_size = 8
        self.max_length = 2048
        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]

        # Device logic
        if torch.cuda.is_available() and not self.cpu:
            avail = torch.cuda.device_count()
            # Cap at 8 as requested
            self.num_gpus = min(num_gpus if num_gpus is not None else avail, 8)
            self.num_gpus = max(self.num_gpus, 1)
            # If only 1 GPU, we keep old single-device behavior
            if self.num_gpus == 1:
                torch.cuda.set_device(0)
            self._use_multi_gpu = (self.num_gpus > 1)
        else:
            # CPU mode
            self.cpu = True
            self.num_gpus = 0
            self._use_multi_gpu = False

        # Defer model/tokenizer init to:
        # - single-GPU/CPU: initialize now (like your original)
        # - multi-GPU: initialize inside each worker to avoid duplicate weights in parent
        self.model = None
        self.tokenizer = None
        self.model_name = None

        if model_name:
            self.model_id = model_name
            self.model_name = model_name.split("/")[-1]

            if not self._use_multi_gpu:
                # Single device path: load once here
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=False if "Phi" in self.model_id else True,
                    device_map={
                        "": ("cuda:0" if (torch.cuda.is_available() and not self.cpu) else "cpu")},
                    torch_dtype=torch.float16 if not self.cpu else torch.float32,
                )
                self.model.eval()

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    add_eos_token=True,
                    use_fast=True,
                    trust_remote_code=True,
                    padding_side="left",
                )
                # Set up chat template for chat-style prompts
                chat_template = (
                    "{% for message in messages %}\n"
                    "{% if message['from'] == 'human' %}\n"
                    "{{ '<|user|>\\n' + message['value'] + eos_token }}\n"
                    "{% elif message['from'] == 'system' %}\n"
                    "{{ '<|system|>\\n' + message['value'] + eos_token }}\n"
                    "{% elif message['from'] == 'gpt' %}\n"
                    "{{ '<|assistant|>\\n'  + message['value'] + eos_token }}\n"
                    "{% endif %}\n"
                    "{% if loop.last and add_generation_prompt %}\n"
                    "{{ '<|assistant|>' }}\n"
                    "{% endif %}\n"
                    "{% endfor %}"
                )
                self.tokenizer.chat_template = chat_template
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.clean_up_tokenization_spaces = True
                assert self.tokenizer.padding_side == "left"

    def __call__(self, refs, hyps):
        dataset = Dataset.from_dict({"reference": refs, "prediction": hyps})
        dataset = self.process_data(dataset)

        self.dataset = dataset
        start = time.time()
        mean, std, green_scores, results_df = self.infer()
        elapsed = time.time() - start

        return mean, std, green_scores, results_df

    def process_data(self, dataset):
        def prompting(examples):
            return {
                "prompt": [
                    make_prompt(r, p)
                    for r, p in zip(examples["reference"], examples["prediction"])
                ]
            }

        return dataset.map(prompting, batched=True)

    @torch.inference_mode()
    def infer(self):
        assert self.model_id is not None, "You must pass a model_name to GREEN(...)"

        prompts = list(self.dataset["prompt"])
        n = len(prompts)

        if self._use_multi_gpu:
            print(f"(GREEN) Multi-GPU inference across {self.num_gpus} GPUs")
            # Shard prompts evenly across workers
            shards = np.array_split(np.arange(n), self.num_gpus)
            shard_prompts = [[prompts[i] for i in idxs] for idxs in shards]

            # Build args for each worker
            devices = [f"cuda:{i}" for i in range(self.num_gpus)]

            # --- progress wiring (spawn-safe, manager-backed queue) ---
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            mgr = ctx.Manager()          # manager so the queue is picklable for Pool workers
            q = mgr.Queue()

            # start the monitor thread
            import sys
            import threading
            from tqdm import tqdm

            def _progress_monitor(q_, total_):
                disable_bar = not sys.stdout.isatty()
                with tqdm(total=total_, desc="Generating", unit="ex", smoothing=0.1, disable=disable_bar) as pbar:
                    done = 0
                    while True:
                        msg = q_.get()
                        if msg is None:  # sentinel
                            break
                        done += int(msg)
                        pbar.n = done
                        pbar.refresh()

            monitor = threading.Thread(
                target=_progress_monitor, args=(q, n), daemon=True)
            monitor.start()
            # ----------------------------------------------------------

            args = [
                (
                    i,                 # worker_id
                    devices[i],        # device
                    self.model_id,     # model identifier
                    shard_prompts[i],  # prompts for this shard
                    self.batch_size,
                    self.max_length,
                    "left",
                    q,                 # progress_queue (Manager-backed)
                )
                for i in range(self.num_gpus)
            ]

            with ctx.Pool(processes=self.num_gpus) as pool:
                worker_outputs = pool.starmap(_worker_generate, args)

            # stop monitor and manager
            q.put(None)
            monitor.join()
            mgr.shutdown()

            # Reassemble outputs in original order
            completions = [None] * n
            for idxs, outs in zip(shards, worker_outputs):
                for k, global_i in enumerate(idxs):
                    completions[int(global_i)] = outs[k]

            self.prompts = prompts
            self.completions = completions

        else:
            # Single-device path (GPU:0 or CPU)
            local_completions = []
            local_references = []

            # Optional single-device progress bar
            import sys
            from tqdm import tqdm
            disable_bar = not sys.stdout.isatty()
            with tqdm(total=n, desc="Generating", unit="ex", disable=disable_bar) as pbar:
                for i in range(0, n, self.batch_size):
                    batch_prompts = prompts[i: i + self.batch_size]
                    batch = {"prompt": batch_prompts}
                    local_references.extend(batch["prompt"])
                    local_completions.extend(self.get_response(batch))
                    pbar.update(len(batch_prompts))

            self.prompts = local_references
            self.completions = local_completions

        return self.process_results()

    def get_response(self, batch):
        # Build chat-formatted inputs
        chats = [
            [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
            for prompt in batch["prompt"]
        ]
        texts = [
            self.tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        toks = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to("cuda:0" if not self.cpu else "cpu")

        outputs = self.model.generate(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.max_length,
            do_sample=False,
        )
        responses = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        # Clean and return as list
        if isinstance(responses, list):
            return [clean_responses(r) for r in responses]
        else:
            return [clean_responses(responses)]

    def process_results(self):
        # Compute green scores and error counts
        self.green_scores = [self.compute_green(r) for r in self.completions]
        self.error_counts = pd.DataFrame(
            [self.compute_error_count(r) for r in self.completions],
            columns=self.sub_categories + ["Matched Findings"],
        )

        results_df = pd.DataFrame(
            {
                "reference": self.dataset["reference"],
                "predictions": self.dataset["prediction"],
                "green_analysis": self.completions,
                "green_score": self.green_scores,
                **self.error_counts,
            }
        )

        mean = float(np.mean(self.green_scores))
        std = float(np.std(self.green_scores))

        return mean, std, self.green_scores, results_df

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(
            response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):
        sig_present, sig_errors = self.parse_error_counts(
            response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(
            response, self.categories[2])

        if matched_findings == 0:
            return 0
        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for _ in range(len(self.sub_categories))]

        if not category_text:
            if for_reward:
                return None, None
            return sum_counts, sub_counts

        body = category_text.group(1)
        if body.startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", body, re.MULTILINE)
            if counts:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts

        # Clinically Significant / Insignificant
        sub_cats = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
        matches = sorted(re.findall(r"\([a-f]\) .*", body))
        if not matches:
            # fallback to numeric markers
            matches = sorted(re.findall(r"\([1-6]\) .*", body))
            sub_cats = [f"({i}) " for i in range(
                1, len(self.sub_categories) + 1)]

        for idx, sub_cat in enumerate(sub_cats):
            for m in matches:
                if m.startswith(sub_cat):
                    num = re.search(r"(?<=: )\b\d+\b(?=\.)", m)
                    if num:
                        sub_counts[idx] = int(num.group(0))
        return sum(sub_counts), sub_counts

    def compute_accuracy(self, responses):
        counts = [self.parse_error_counts(r, self.categories[0])[
            1] for r in responses]
        arr = np.array(counts)
        return {
            sub: float((arr[:, idx] == 0).mean())
            for idx, sub in enumerate(self.sub_categories)
        }


if __name__ == "__main__":
    # If you plan to invoke GREEN here and you're on macOS/Windows,
    # make sure the calls that trigger multi-GPU are under this guard.
    pass
