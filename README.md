# RadEval

<div align="center">

**All-in-one metrics for evaluating AI-generated radiology text**

</div>

<!--- BADGES: START --->
[![PyPI](https://img.shields.io/badge/RadEval-v0.0.6-00B7EB?logo=python&logoColor=00B7EB)](https://pypi.org/project/RadEval/)
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

evaluator = RadEval(
    do_radgraph=True,
    do_bleu=True
)

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
pip install RadEval[api]         # include OpenAI/Gemini for MammoGREEN
```

Or install from source:
```bash
git clone https://github.com/jbdel/RadEval.git && cd RadEval
conda create -n radeval python=3.11 -y && conda activate radeval
pip install -e '.[api]'
```

## Supported Metrics

| Category | Metric | Flag | Best For |
|----------|--------|------|----------|
| **Lexical** | BLEU | `do_bleu` | Surface-level n-gram overlap |
| | ROUGE | `do_rouge` | Content coverage |
| **Semantic** | BERTScore | `do_bertscore` | Semantic similarity |
| | RadEval BERTScore | `do_radeval_bertscore` | Domain-adapted radiology semantics |
| **Clinical** | F1CheXbert | `do_chexbert` | CheXpert finding classification |
| | F1RadBERT-CT | `do_f1radbert_ct` | CT finding classification |
| | F1RadGraph | `do_radgraph` | Clinical entity/relation accuracy |
| | RaTEScore | `do_ratescore` | Entity-level synonym-aware scoring |
| **Specialized** | RadGraph-RadCliQ | `do_radgraph_radcliq` | Per-pair entity+relation F1 (RadCliQ variant) |
| | RadCliQ-v1 | `do_radcliq` | Composite clinical relevance |
| | SRR-BERT | `do_srr_bert` | Structured report evaluation |
| | Temporal F1 | `do_temporal` | Temporal consistency |
| | GREEN | `do_green` | LLM-based overall quality (7B model) |
| | MammoGREEN | `do_mammo_green` | Mammography-specific LLM scoring |
| | RadFact-CT | `do_radfact_ct` | LLM-based factual precision/recall for CT |

Enable only the metrics you need -- each one is loaded lazily.

## Detailed Output

Pass `do_details=True` to get per-sample scores, label breakdowns, and entity annotations for every enabled metric. See [docs/metrics.md](docs/metrics.md) for the full output schema of each metric.

## Comparing Systems

Use `compare_systems` to run paired approximate randomization tests between any number of systems:

```python
from RadEval import RadEval, compare_systems

evaluator = RadEval(do_bleu=True)
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
| [docs/metrics.md](docs/metrics.md) | What each metric measures, `do_details` output schemas |
| [docs/configuration.md](docs/configuration.md) | Full parameter reference, example presets |
| [docs/hypothesis_testing.md](docs/hypothesis_testing.md) | Statistical background, full example, performance notes |
| [docs/file_formats.md](docs/file_formats.md) | Loading data from .tok, .json, and Python lists |

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
    </tr>
  </tbody>
</table>

## Acknowledgments

Built on the work of the radiology AI community: [CheXbert](https://github.com/stanfordmlgroup/CheXbert), [RadGraph](https://github.com/jbdel/RadGraph), [BERTScore](https://github.com/Tiiiger/bert_score), [RaTEScore](https://github.com/MAGIC-AI4Med/RaTEScore), [SRR-BERT](https://github.com/StanfordAIMI/SRR-BERT), [GREEN](https://github.com/Stanford-AIMI/GREEN), and datasets like [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/).

---
<div align="center">
  <p>If you find RadEval useful, please give us a star!</p>
</div>
