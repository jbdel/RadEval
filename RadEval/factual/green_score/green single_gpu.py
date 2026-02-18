import re
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

# Import necessary functions (ensure these are available in your environment)
from green_score.utils import (
    make_prompt,
    clean_responses,
    compute_largest_cluster,
    flatten_values_lists_of_list_dicts_to_dict,
)

# Set the logging level for the transformers library to ERROR to suppress benign warnings
logging.get_logger("transformers").setLevel(logging.ERROR)


class GREEN:
    def __init__(
        self,
        model_name=None,
        output_dir=".",
        cpu=False,
        compute_summary_stats=False,
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
        self.compute_summary_stats = compute_summary_stats

        # Force single‐GPU (cuda:0) or CPU
        if torch.cuda.is_available() and not self.cpu:
            torch.cuda.set_device(0)
        else:
            self.cpu = True

        # Model + tokenizer will be set up only if model_name is provided
        self.model = None
        self.tokenizer = None
        if model_name:
            self.model_name = model_name.split("/")[-1]
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=False if "Phi" in model_name else True,
                device_map={"": "cuda:0"} if (torch.cuda.is_available() and not self.cpu) else {"": "cpu"},
                torch_dtype=torch.float16,
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                add_eos_token=True,
                use_fast=True,
                trust_remote_code=True,
                padding_side="left",
            )
            # Set up chat template for chat-style prompts
            chat_template = (
                "{% for message in messages %}\n"
                "{% if message['from'] == 'human' %}\n"
                "{{ '<|user|>\n' + message['value'] + eos_token }}\n"
                "{% elif message['from'] == 'system' %}\n"
                "{{ '<|system|>\n' + message['value'] + eos_token }}\n"
                "{% elif message['from'] == 'gpt' %}\n"
                "{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n"
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
        print("Processing data...making prompts")
        dataset = Dataset.from_dict({"reference": refs, "prediction": hyps})
        dataset = self.process_data(dataset)
        print("Done.")

        self.dataset = dataset
        start = time.time()
        mean, std, green_scores, summary, results_df = self.infer()
        elapsed = time.time() - start
        print(f"Seconds per example: {elapsed / len(refs):.3f}")

        return mean, std, green_scores, summary, results_df

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
        assert self.model is not None and self.tokenizer is not None
        local_completions = []
        local_references = []

        for batch in tqdm(self.dataset.iter(batch_size=self.batch_size),
                          total=len(self.dataset) // self.batch_size):
            local_references.extend(batch["prompt"])
            local_completions.extend(self.get_response(batch))

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
            self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        toks = self.tokenizer(
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
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

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

        if self.compute_summary_stats:
            summary = self.compute_summary(mean, std)
        else:
            summary = None

        return mean, std, self.green_scores, summary, results_df

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

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
            sub_cats = [f"({i}) " for i in range(1, len(self.sub_categories) + 1)]

        for idx, sub_cat in enumerate(sub_cats):
            for m in matches:
                if m.startswith(sub_cat):
                    num = re.search(r"(?<=: )\b\d+\b(?=\.)", m)
                    if num:
                        sub_counts[idx] = int(num.group(0))
        return sum(sub_counts), sub_counts

    def parse_error_sentences(self, response, category):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, response, re.DOTALL)

        sentences = {sub: [] for sub in self.sub_categories}
        if not category_text:
            return sentences

        body = category_text.group(1)
        if body.startswith("No"):
            return sentences

        if category == "Matched Findings":
            # last clause after colon, split on semicolon
            return body.rsplit(":", 1)[-1].rsplit(".", 1)[-1].split(";")

        matches = sorted(re.findall(r"\([a-f]\) .*", body))
        if not matches:
            matches = sorted(re.findall(r"\([1-6]\) .*", body))
            self.sub_categories = [f"({i}) " for i in range(1, len(self.sub_categories) + 1)]

        for idx, sub_cat in enumerate(self.sub_categories):
            for m in matches:
                if m.startswith(sub_cat):
                    part = m.rsplit(":", 1)[-1]
                    sentences[sub_cat] = part.split(";")
        return sentences

    def compute_sentences(self, response):
        return self.parse_error_sentences(response, self.categories[0])

    def get_representative_sentences(self, responses):
        list_sentences = [self.compute_sentences(r) for r in responses]
        flat = flatten_values_lists_of_list_dicts_to_dict(list_sentences)

        result = {}
        for sub in self.sub_categories:
            items = [s for s in flat.get(sub, []) if s and s.strip()]
            # if fewer than 2 distinct items, just return them
            if len(set(items)) < 2:
                result[sub] = items
            else:
                try:
                    _, cluster = compute_largest_cluster(items)
                except ValueError:
                    cluster = items
                result[sub] = cluster
        return result

    def compute_accuracy(self, responses):
        counts = [self.parse_error_counts(r, self.categories[0])[1] for r in responses]
        arr = np.array(counts)
        return {
            sub: float((arr[:, idx] == 0).mean())
            for idx, sub in enumerate(self.sub_categories)
        }

    def compute_summary(self, mean, std):
        print("Computing summary ...")
        reps = self.get_representative_sentences(self.completions)
        accs = self.compute_accuracy(self.completions)
        summary = [f"-------------{self.model_name}----------------",
                   f"[Summary]: Green average {mean} and standard deviation {std}",
                   "[Clinically Significant Errors Analyses]: <accuracy>. <representative error>"]
        for sub in self.sub_categories:
            summary.append(f"{sub}: {accs[sub]}. {reps.get(sub, [])}")
        summary.append("----------------------------------")
        return "\n".join(summary)


if __name__ == "__main__":
    pass
