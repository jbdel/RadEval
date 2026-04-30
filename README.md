# RadEval

<!--- BADGES: START --->
[![PyPI](https://img.shields.io/badge/RadEval-v2.2.0-00B7EB?logo=python&logoColor=00B7EB)](https://pypi.org/project/RadEval/)
[![Python version](https://img.shields.io/badge/python-3.11+-important?logo=python&logoColor=important)]()
[![Expert Dataset](https://img.shields.io/badge/Expert-%20Dataset-4CAF50?logo=googlecloudstorage&logoColor=9BF0E1)](https://huggingface.co/datasets/IAMJB/RadEvalExpertDataset)
[![Model](https://img.shields.io/badge/Model-RadEvalModernBERT-0066CC?logo=huggingface&labelColor=grey)](https://huggingface.co/IAMJB/RadEvalModernBERT)
[![Video](https://img.shields.io/badge/Talk-Video-9C27B0?logo=youtubeshorts&labelColor=grey)](https://justin13601.github.io/files/radeval.mp4)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-FFD21E.svg?logo=gradio&logoColor=gold)](https://huggingface.co/spaces/X-iZhang/RadEval)
[![EMNLP](https://img.shields.io/badge/paper-EMNLP-red)](https://aclanthology.org/2025.emnlp-demos.40/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?)](https://github.com/jbdel/RadEval/main/LICENSE)
<!--- BADGES: END --->

RadEval (EMNLP, 2025) is a Python framework for evaluating AI-generated radiology reports. It serves two use cases:

1. **Evaluation:** 16 metrics spanning lexical, semantic, clinical, and LLM-based evaluation, all behind a single interface with lazy loading and config-file support.
2. **Reinforcement-learning (RL) rewards:** every RL-eligible metric exposed as a drop-in HuggingFace TRL reward function for GRPO (and other trainers that accept a reward callable).

## Table of Contents

- [Installation](#installation)
- [Usage: Evaluation](#usage-evaluation)
  - [Basic](#basic)
  - [Config file](#config-file)
  - [Output modes](#output-modes)
  - [Comparing systems](#comparing-systems)
- [Usage: RL rewards](#usage-rl-rewards)
  - [Quickstart](#rl-quickstart)
  - [Benchmarks: cost & divergence](#rl-benchmarks-cost--divergence)
  - [Reward API & docs](#rl-reward-api--docs)
- [Supported Metrics](#supported-metrics)
- [API Keys for LLM Metrics](#api-keys-for-llm-metrics)
- [Documentation](#documentation)
- [Expert Dataset](#radeval-expert-dataset)
- [Contributing](#contributing)
- [Citation](#citation)

## Installation

```bash
pip install radeval              # from PyPI
pip install radeval[api]         # include OpenAI/Gemini for LLM-based metrics
```

Or install from source:
```bash
git clone https://github.com/jbdel/RadEval.git && cd RadEval
conda create -n radeval python=3.11 -y && conda activate radeval
pip install -e '.[api]'
# Torch wheels are CUDA-version specific. If the default wheel from PyPI does
# not match your local NVIDIA driver, install a matching build first, e.g.:
# pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1
```
> **Known-good stack (for RadEval 2.1+):** Python 3.11, `torch==2.9.1+cu128`,
> `transformers==5.6.2`, `tokenizers==0.22.2`, `huggingface_hub>=1.0`,
> `accelerate>=1.1`, `numpy<3`. Full test suite passes on this configuration.
> For the `[rl]` extras, add `trl>=1.3.0,<2`.

## Usage: Evaluation

### Basic

Pass a list of metric names. Each metric is loaded lazily; only the ones you enable import their dependencies.

```python
from radeval import RadEval
import json

refs = [
    "Mild cardiomegaly with small bilateral pleural effusions and basilar atelectasis.",
    "No pleural effusions or pneumothoraces.",
]
hyps = [
    "Mildly enlarged cardiac silhouette with small pleural effusions and dependent bibasilar atelectasis.",
    "No pleural effusions or pneumothoraces.",
]

evaluator = RadEval(metrics=["radgraph", "bleu"])
results = evaluator(refs=refs, hyps=hyps)
print(json.dumps(results, indent=4))
```
```json
{
    "radgraph_simple": 0.72,
    "radgraph_partial": 0.61,
    "radgraph_complete": 0.61,
    "bleu": 0.36
}
```

### Config file

For per-metric settings (model, provider, concurrency) or reproducible evaluation configs, use a YAML file:

```yaml
# config.yaml
metrics:
  - bleu
  - rouge
  - crimson:
      provider: openai
      model_name: gpt-4o-mini
  - radfact_ct:
      filter_negatives: true

output:
  mode: per_sample    # or "default" or "detailed"
```

```python
evaluator = RadEval.from_config("config.yaml")
results = evaluator(refs=refs, hyps=hyps)
```

See [`examples/config.yaml`](examples/config.yaml) for a complete example.

### Output modes

| Mode | Flag | Values |
|------|------|--------|
| Default | — | `float` per metric |
| Per-sample | `per_sample=True` | `list[float]` per metric (one per report) |
| Detailed | `detailed=True` | Extra keys: label breakdowns, BLEU-1/2/3, std |

```python
# Per-sample scores
evaluator = RadEval(metrics=["bleu", "bertscore"], per_sample=True)
results = evaluator(refs=refs, hyps=hyps)
# results["bleu"]      → [0.85, 0.40, ...]   (one per sample)
# results["bertscore"] → [0.95, 0.89, ...]

# Detailed output (label F1s, sub-scores, std)
evaluator = RadEval(metrics=["bleu", "f1chexbert"], detailed=True)
results = evaluator(refs=refs, hyps=hyps)
# results["bleu_1"]    → 0.55   (extra: BLEU-1)
# results["bleu_2"]    → 0.42   (extra: BLEU-2)
```

See [docs/metrics.md](docs/metrics.md) for the full output schema of each metric.

### Comparing systems

Use `compare_systems` to run paired approximate randomization tests between any number of systems:

```python
from radeval import RadEval, compare_systems

evaluator = RadEval(metrics=["bleu"])
signatures, scores = compare_systems(
    systems={'baseline': baseline_reports, 'improved': improved_reports},
    metrics={'bleu': lambda hyps, refs: evaluator(refs, hyps)['bleu']},
    references=reference_reports,
    n_samples=10000,
)
```

See [docs/hypothesis_testing.md](docs/hypothesis_testing.md) for a full walkthrough and interpretation guide.

## Usage: RL rewards

RadEval metrics aren't just for offline evaluation — every RL-eligible metric is a drop-in [HuggingFace TRL](https://github.com/huggingface/trl) reward function. GRPO is the flagship, tested path; RLOO and other TRL trainers that consume a reward-function callable use the same interface.

Three things to look at, in increasing depth:

### RL quickstart

```bash
pip install radeval[rl]    # adds trl>=1.3.0,<2
```

```python
from radeval.rewards import make_reward_fn
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[make_reward_fn("bleu")],   # or bertscore, radgraph (key=...), radcliq, ...
    train_dataset=dataset,                   # must have a "ground_truth" column
)
trainer.train()
```

Runnable end-to-end: `python examples/trl_grpo_quickstart.py`.

### RL benchmarks: cost & divergence

How expensive is each metric when used as a per-step reward, how does reward choice change what the model learns? See **[docs/trl_rewards_benchmarks.md](docs/trl_rewards_benchmarks.md)** for:

- A **speed table** covering all 16 public metrics, from **0.09 ms/sample** (BLEU, CPU) to **~2,200 ms/sample** (GREEN, 7B local LLM). RadCliQ, a metric with strong correlation to radiologist preferences, comes in at **~161 ms/sample**.
- A **reward-divergence gallery**: same rollouts, scored by several metrics side-by-side. **Headline finding**: on a negation flip ("No pleural effusion." → "Pleural effusion."), BERTScore rewards the clinically-wrong rollout at **0.893**, nearly its 1.0 ceiling; a GRPO policy trained against BERTScore would be pushed *toward* this rollout. Clinical metrics penalize the flip, but by widely varying magnitudes: RadGraph drops from 1.0 to 0.50, RadCliQ rises by ~1.7 distance units, and CRIMSON (LLM judge, signed range (−1, 1]) scores **−0.333**: a single hallucinated abnormal finding against a normal reference. The benchmarks page lays out the full per-metric reaction across several other rollout types.

### RL reward API & docs

- **[docs/trl_rewards.md](docs/trl_rewards.md):** `make_reward_fn` contract, required `key=` for multi-key metrics, conversational-completion handling, multi-metric composition, VLM pointer, known limitations.
- Note: For *distance* metrics (lower = better) such as RadCliQ, use the safe inversion `make_reward_fn("radcliq", score_transform=lambda x: -x)`.

## Supported Metrics

| Category | Metric | Key | Modality | Provider | Best For | Usage |
|----------|--------|-----|----------|----------|----------|-------|
| **Lexical** | [BLEU](https://aclanthology.org/P02-1040.pdf) | `"bleu"` | -- | -- | Surface-level n-gram overlap | [docs](docs/metrics.md#bleu-bleu) |
| | [ROUGE](https://aclanthology.org/W04-1013.pdf) | `"rouge"` | -- | -- | Content coverage | [docs](docs/metrics.md#rouge-rouge) |
| **Semantic** | [BERTScore](https://openreview.net/forum?id=SkeHuCVFDr) | `"bertscore"` | -- | -- | Semantic similarity | [docs](docs/metrics.md#bertscore-bertscore) |
| | [RadEval BERTScore](https://aclanthology.org/2025.emnlp-demos.40.pdf) | `"radeval_bertscore"` | -- | -- | Domain-adapted radiology semantics | [docs](docs/metrics.md#radeval-bertscore-radeval_bertscore) |
| **Clinical** | [F1CheXbert](https://aclanthology.org/2020.emnlp-main.117.pdf) | `"f1chexbert"` | CXR | -- | CheXpert finding classification | [docs](docs/metrics.md#f1chexbert-f1chexbert) |
| | [F1RadBERT-CT](https://www.nature.com/articles/s41551-025-01599-y) | `"f1radbert_ct"` | CT | -- | CT finding classification | [docs](docs/metrics.md#f1radbert-ct-f1radbert_ct) |
| | [F1RadGraph](https://aclanthology.org/2022.findings-emnlp.319.pdf) | `"radgraph"` | CXR | -- | Clinical entity/relation accuracy | [docs](docs/metrics.md#f1radgraph-radgraph) |
| | [RaTEScore](https://aclanthology.org/2024.emnlp-main.836.pdf) | `"ratescore"` | CXR | -- | Entity-level synonym-aware scoring | [docs](docs/metrics.md#ratescore-ratescore) |
| **Specialized** | [RadGraph-RadCliQ](https://www.cell.com/patterns/pdfExtended/S2666-3899(23)00157-5) | `"radgraph_radcliq"` | CXR | -- | Per-pair entity+relation F1 (RadCliQ variant) | [docs](docs/metrics.md#radgraph-radcliq-radgraph_radcliq) |
| | [RadCliQ-v1](https://www.cell.com/patterns/pdfExtended/S2666-3899(23)00157-5) | `"radcliq"` | CXR | -- | Composite clinical relevance | [docs](docs/metrics.md#radcliq-v1-radcliq) |
| | [SRRBert](https://aclanthology.org/2025.acl-long.1301.pdf) | `"srrbert"` | CXR | -- | Structured report evaluation | [docs](docs/metrics.md#srrbert-srrbert) |
| | [Temporal F1](https://aclanthology.org/2025.findings-acl.888.pdf) | `"temporal"` | CXR | -- | Temporal consistency | [docs](docs/metrics.md#temporal-f1-temporal) |
| | [GREEN](https://aclanthology.org/2024.findings-emnlp.21.pdf) | `"green"` | CXR | Local HF | LLM-based overall quality (7B model) | [docs](docs/metrics.md#green-green) |
| | MammoGREEN | `"mammo_green"` | Mammo | OpenAI / Gemini | Mammography-specific LLM scoring | [docs](docs/metrics.md#mammogreen-mammo_green) |
| | [CRIMSON](https://arxiv.org/pdf/2603.06183) | `"crimson"` | CXR | OpenAI / HF | LLM-based clinical significance scoring | [docs](docs/metrics.md#crimson-crimson) |
| | [RadFact-CT](https://arxiv.org/pdf/2510.15042) | `"radfact_ct"` | CT | OpenAI | LLM-based factual precision/recall | [docs](docs/metrics.md#radfact-ct-radfact_ct) |

> **Modality:** CXR = Chest X-Ray, CT = Computed Tomography, Mammo = Mammography, -- = modality-agnostic.

Enable only the metrics you need; each one is loaded lazily.

## API Keys for LLM Metrics

LLM-based metrics (CRIMSON, MammoGREEN, RadFact-CT) share two **global** API key arguments:

```python
evaluator = RadEval(
    metrics=["crimson", "mammo_green", "radfact_ct"],
    openai_api_key="sk-...",   # used by CRIMSON (openai), MammoGREEN (openai), RadFact-CT
    gemini_api_key="AIza...",  # used by MammoGREEN (gemini)
)
```

If not passed explicitly, keys fall back to the environment variables `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY`. An error is raised if the chosen provider requires a key that is neither passed nor in the environment.

## Documentation

| Page | Contents |
|------|----------|
| [docs/metrics.md](docs/metrics.md) | What each metric measures, `per_sample` / `detailed` output schemas |
| [docs/hypothesis_testing.md](docs/hypothesis_testing.md) | Statistical background, full example, performance notes |
| [docs/file_formats.md](docs/file_formats.md) | Loading data from .tok, .json, and Python lists |
| [docs/trl_rewards.md](docs/trl_rewards.md) | Using RadEval metrics as RL reward functions with HuggingFace TRL |
| [docs/trl_rewards_benchmarks.md](docs/trl_rewards_benchmarks.md) | Speed table + reward-divergence gallery for picking an RL reward metric |

## RadEval Expert Dataset

A curated evaluation set annotated by board-certified radiologists for validating automatic metrics. Available on [HuggingFace](https://huggingface.co/datasets/IAMJB/RadEvalExpertDataset).

## Citation

```BibTeX
@inproceedings{xu-etal-2025-radeval,
    title = "{R}ad{E}val: A framework for radiology text evaluation",
    author = "Xu, Justin  and
      Zhang, Xi  and
      Abderezaei, Javid  and
      Bauml, Julie  and
      Boodoo, Roger  and
      Haghighi, Fatemeh  and
      Ganjizadeh, Ali  and
      Brattain, Eric  and
      Van Veen, Dave  and
      Meng, Zaiqiao  and
      Eyre, David W  and
      Delbrouck, Jean-Benoit",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-demos.40/",
    doi = "10.18653/v1/2025.emnlp-demos.40",
    pages = "546--557",
}
```

### Contributors
<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://jbdel.github.io/">
          <img src="https://aimi.stanford.edu/sites/g/files/sbiybj20451/files/styles/medium_square/public/media/image/image5_0.png?h=f4e62a0a&itok=euaj9VoF"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Jean-Benoit Delbrouck"/>
          <br />
          <sub><b>Jean-Benoit Delbrouck</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://justin13601.github.io/">
          <img src="https://justin13601.github.io/images/pfp2.JPG"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Justin Xu"/>
          <br />
          <sub><b>Justin Xu</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://x-izhang.github.io/">
          <img src="https://x-izhang.github.io/author/xi-zhang/avatar_hu13660783057866068725.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Xi Zhang"/>
          <br />
          <sub><b>Xi Zhang</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://davevanveen.com">
          <img src="https://davevanveen.com/assets/img/prof_pic.jpg"
               width="100" height="100"
               style="object-fit: cover; border-radius: 20%;" alt="Dave Van Veen"/>
          <br />
          <sub><b>Dave Van Veen</b></sub>
        </a>
      </td>
    </tr>
  </tbody>
</table>

## Contributing

RadEval is open source and we welcome contributions from the community. Whether it's a new metric, a bug fix, or improved documentation; feel free to open an issue or submit a pull request on [GitHub](https://github.com/jbdel/RadEval).

## Acknowledgments

Built on the work of the radiology AI community: [CheXbert](https://github.com/stanfordmlgroup/CheXbert), [RadGraph](https://github.com/jbdel/RadGraph), [BERTScore](https://github.com/Tiiiger/bert_score), [RaTEScore](https://github.com/MAGIC-AI4Med/RaTEScore), [SRR-BERT](https://github.com/StanfordAIMI/SRR-BERT), [GREEN](https://github.com/Stanford-AIMI/GREEN), [CRIMSON](https://github.com/rajpurkarlab/CRIMSON), and datasets like [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/).

---
<div align="center">
  <p>Please give us a star if you find RadEval useful! ⭐</p>
</div>
