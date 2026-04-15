# RadEval

<div align="center">

**All-in-one metrics for evaluating AI-generated radiology text**

</div>

<!--- BADGES: START --->
[![PyPI](https://img.shields.io/badge/RadEval-v1.0.0-00B7EB?logo=python&logoColor=00B7EB)](https://pypi.org/project/RadEval/)
[![Python version](https://img.shields.io/badge/python-3.11+-important?logo=python&logoColor=important)]()
[![Expert Dataset](https://img.shields.io/badge/Expert-%20Dataset-4CAF50?logo=googlecloudstorage&logoColor=9BF0E1)](https://huggingface.co/datasets/IAMJB/RadEvalExpertDataset)
[![Model](https://img.shields.io/badge/Model-RadEvalModernBERT-0066CC?logo=huggingface&labelColor=grey)](https://huggingface.co/IAMJB/RadEvalModernBERT)
[![Video](https://img.shields.io/badge/Talk-Video-9C27B0?logo=youtubeshorts&labelColor=grey)](https://justin13601.github.io/files/radeval.mp4)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-FFD21E.svg?logo=gradio&logoColor=gold)](https://huggingface.co/spaces/X-iZhang/RadEval)
[![EMNLP](https://img.shields.io/badge/paper-EMNLP-red)](https://aclanthology.org/2025.emnlp-demos.40/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?)](https://github.com/jbdel/RadEval/main/LICENSE)
<!--- BADGES: END --->


### TL;DR
```
pip install -e .
```
```python
from RadEval import RadEval
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
print(json.dumps(results, indent=2))
```
```json
{
  "radgraph_simple": 0.72,
  "radgraph_partial": 0.61,
  "radgraph_complete": 0.61,
  "bleu": 0.36
}
```

## Installation

```bash
pip install RadEval              # from PyPI
pip install RadEval[api]         # include OpenAI/Gemini for LLM-based metrics
```

Or install from source:
```bash
git clone https://github.com/jbdel/RadEval.git && cd RadEval
conda create -n radeval python=3.11 -y && conda activate radeval
pip install -e '.[api]'
```

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

Enable only the metrics you need -- each one is loaded lazily.

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

## Per-Sample Output

Pass `per_sample=True` to get per-sample scores for every enabled metric. The output uses the **same flat keys** as the default mode, but each value is a `list[float]` of length `n_samples` instead of a single aggregate.

```python
evaluator = RadEval(metrics=["bleu", "bertscore"], per_sample=True)
results = evaluator(refs=refs, hyps=hyps)
# results["bleu"]      → [0.85, 0.40, ...]   (one per sample)
# results["bertscore"] → [0.95, 0.89, ...]
```

See [docs/metrics.md](docs/metrics.md) for the full list of per-sample output keys for each metric.

## Detailed Output

Pass `detailed=True` to get additional aggregate scores beyond the defaults: per-label F1 breakdowns for classifiers, BLEU-1/2/3, standard deviations for LLM-based metrics. Same flat keys as default, no nesting.

```python
evaluator = RadEval(metrics=["bleu", "f1chexbert", "crimson"], detailed=True)
results = evaluator(refs=refs, hyps=hyps)
# results["bleu"]       → 0.36     (same as default)
# results["bleu_1"]     → 0.55     (extra: BLEU-1)
# results["bleu_2"]     → 0.42     (extra: BLEU-2)
# results["crimson_std"] → 0.15    (extra: std)
# results["f1chexbert_label_scores_f1"] → {"f1chexbert_5": {"Cardiomegaly": 0.59, ...}, ...}
```

See [docs/metrics.md](docs/metrics.md) for the full output schema of each metric.

## Comparing Systems

Use `compare_systems` to run paired approximate randomization tests between any number of systems:

```python
from RadEval import RadEval, compare_systems

evaluator = RadEval(metrics=["bleu"])
signatures, scores = compare_systems(
    systems={
        'baseline': baseline_reports,
        'improved': improved_reports,
    },
    metrics={'bleu': lambda hyps, refs: evaluator(refs, hyps)['bleu']},
    references=reference_reports,
    n_samples=10000,
)
```

See [docs/hypothesis_testing.md](docs/hypothesis_testing.md) for a full walkthrough and interpretation guide.

## Documentation

| Page | Contents |
|------|----------|
| [docs/metrics.md](docs/metrics.md) | What each metric measures, `per_sample` / `detailed` output schemas |
| [docs/hypothesis_testing.md](docs/hypothesis_testing.md) | Statistical background, full example, performance notes |
| [docs/file_formats.md](docs/file_formats.md) | Loading data from .tok, .json, and Python lists |
| [docs/trl_rewards.md](docs/trl_rewards.md) | Using RadEval metrics as RL reward functions with HuggingFace TRL |

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

## Acknowledgments

Built on the work of the radiology AI community: [CheXbert](https://github.com/stanfordmlgroup/CheXbert), [RadGraph](https://github.com/jbdel/RadGraph), [BERTScore](https://github.com/Tiiiger/bert_score), [RaTEScore](https://github.com/MAGIC-AI4Med/RaTEScore), [SRR-BERT](https://github.com/StanfordAIMI/SRR-BERT), [GREEN](https://github.com/Stanford-AIMI/GREEN), [CRIMSON](https://github.com/rajpurkarlab/CRIMSON), and datasets like [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/).

---
<div align="center">
  <p>If you find RadEval useful, please give us a star!</p>
</div>
