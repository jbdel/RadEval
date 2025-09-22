<div align="center">
  <a href="https://github.com/jbdel/RadEval">
    <img src="https://github.com/jbdel/RadEval/raw/libra_run/RadEval_banner.png" alt="RadEval" width="100%" style="border-radius: 16px;">
  </a>
</div>

<div align="center">

**All-in-one metrics for evaluating AI-generated radiology text**

</div>

<!--- BADGES: START --->
[![PyPI](https://img.shields.io/badge/RadEval-v0.0.1-00B7EB?logo=python&logoColor=00B7EB)](https://pypi.org/project/RadEval/)
[![Python version](https://img.shields.io/badge/python-3.10+-important?logo=python&logoColor=important)]()
[![Expert Dataset](https://img.shields.io/badge/Expert-%20Dataset-4CAF50?logo=googlecloudstorage&logoColor=9BF0E1)]()
[![Model](https://img.shields.io/badge/Model-RadEvalModernBERT-0066CC?logo=huggingface&labelColor=grey)](https://huggingface.co/IAMJB/RadEvalModernBERT)
[![Video](https://img.shields.io/badge/Talk-Video-9C27B0?logo=youtubeshorts&labelColor=grey)](https://justin13601.github.io/files/radeval.mp4)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-FFD21E.svg?logo=gradio&logoColor=gold)](https://huggingface.co/spaces/X-iZhang/RadEval)
[![Arxiv](https://img.shields.io/badge/arXiv-coming_soon-B31B1B.svg?logo=arxiv&logoColor=B31B1B)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg?)](https://github.com/jbdel/RadEval/main/LICENSE)
<!--- BADGES: END --->

## 📖 Table of Contents

- [🌟 Overview](#-overview)
  - [❓ Why RadEval](#-why-radeval)
  - [✨ Key Features](#-key-features)
- [⚙️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [🔧 Configuration Options](#-configuration-options)
- [📁 File Format Suggestion](#-file-format-suggestion)
- [🧪 Hypothesis Testing (Significance Evaluation)](#-hypothesis-testing-significance-evaluation)
- [🧠 RadEval Expert Dataset](#-radeval-expert-dataset)
- [🚦 Performance Tips](#-performance-tips)
- [📚 Citation](#-citation)

## 🌟 Overview

**RadEval** is a comprehensive evaluation framework specifically designed for assessing the quality of AI-generated radiology text. It provides a unified interface to multiple state-of-the-art evaluation metrics, enabling researchers and practitioners to thoroughly evaluate their radiology text generation models.

### ❓ Why RadEval
> [!TIP]
> - **Domain-Specific**: Tailored for radiology text evaluation with medical knowledge integration
> - **Multi-Metric**: Supports 11+ different evaluation metrics in one framework
> - **Easy to Use**: Simple API with flexible configuration options
> - **Comprehensive**: From traditional n-gram metrics to advanced LLM-based evaluations
> - **Research-Ready**: Built for reproducible evaluation in radiology AI research

### ✨ Key Features
> [!NOTE]
> - **Multiple Evaluation Perspectives**: Lexical, semantic, clinical, and temporal evaluations
> - **Statistical Testing**: Built-in hypothesis testing for system comparison
> - **Batch Processing**: Efficient evaluation of large datasets
> - **Flexible Configuration**: Enable/disable specific metrics based on your needs
> - **Detailed Results**: Comprehensive output with metric explanations
> - **File Format Support**: Direct evaluation from common file formats (.tok, .txt, .json)

## ⚙️ Installation
RadEval supports Python **3.10+** and can be installed via PyPI or from source.

### Option 1: Install via PyPI (Recommended)

```bash
pip install RadEval
```
> [!TIP]
> We recommend using a virtual environment to avoid dependency conflicts, especially since some metrics require loading large inference models.

### Option 2: Install from GitHub (Latest Development Version)
Install the most up-to-date version directly from GitHub:
```bash
pip install git+https://github.com/jbdel/RadEval.git
```
> This is useful if you want the latest features or bug fixes before the next PyPI release.

### Option 3: Install in Development Mode (Recommended for Contributors)
```bash
# Clone the repository
git clone https://github.com/jbdel/RadEval.git
cd RadEval

# Create and activate a conda environment
conda create -n RadEval python=3.10 -y
conda activate RadEval

# Install in development (editable) mode
pip install -e .
```
> This setup allows you to modify the source code and reflect changes immediately without reinstallation.

## 🚀 Quick Start

### Example 1: Basic Evaluation
Evaluate a few reports using selected metrics:
```python
from RadEval import RadEval
import json

refs = [
    "No definite acute cardiopulmonary process.Enlarged cardiac silhouette could be accentuated by patient's positioning.",
    "Increased mild pulmonary edema and left basal atelectasis.",
]
hyps = [
    "Relatively lower lung volumes with no focal airspace consolidation appreciated.",
    "No pleural effusions or pneumothoraces.",
]

evaluator = RadEval(
    do_radgraph=True,
    do_bleu=True
)

results = evaluator(refs=refs, hyps=hyps)
print(json.dumps(results, indent=2))
```
<details>
<summary> Output </summary>

```json
{
  "radgraph_simple": 0.5,
  "radgraph_partial": 0.5,
  "radgraph_complete": 0.5,
  "bleu": 0.5852363407461811
}
```

</details>

### Example 2: Comprehensive Evaluation
Set `do_details=True` to enable per-metric detailed outputs, including entity-level comparisons and score-specific breakdowns when supported.

```python
from RadEval import RadEval
import json

evaluator = RadEval(
    do_srr_bert=True,
    do_rouge=True,
    do_details=True
)

refs = [
    "No definite acute cardiopulmonary process.Enlarged cardiac silhouette could be accentuated by patient's positioning.",
    "Increased mild pulmonary edema and left basal atelectasis.",
]
hyps = [
    "Relatively lower lung volumes with no focal airspace consolidation appreciated.",
    "No pleural effusions or pneumothoraces.",
]

results = evaluator(refs=refs, hyps=hyps)
print(json.dumps(results, indent=2))
```

<details>
<summary> Output </summary>

```json
{
  "rouge": {
    "rouge1": {
      "mean_score": 0.04,
      "sample_scores": [
        0.08,
        0.0
      ]
    },
    "rouge2": {
      "mean_score": 0.0,
      "sample_scores": [
        0.0,
        0.0
      ]
    },
    "rougeL": {
      "mean_score": 0.04,
      "sample_scores": [
        0.08,
        0.0
      ]
    }
  },
  "srr_bert": {
    "srr_bert_weighted_f1": 0.16666666666666666,
    "srr_bert_weighted_precision": 0.125,
    "srr_bert_weighted_recall": 0.25,
    "label_scores": {
      "Edema (Present)": {
        "f1-score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "support": 1.0
      },
      "Atelectasis (Present)": {
        "f1-score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "support": 1.0
      },
      "Cardiomegaly (Uncertain)": {
        "f1-score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "support": 1.0
      },
      "No Finding": {
        "f1-score": 0.6666666666666666,
        "precision": 0.5,
        "recall": 1.0,
        "support": 1.0
      }
    }
  }
}
```

</details>

### Example 3: Quick Hypothesis Testing
Compare two systems statistically to validate improvements:

```python
from RadEval import RadEval, compare_systems

# Define systems to compare
systems = {
    'baseline': [
        "No acute findings.",
        "Mild heart enlargement."
    ],
    'improved': [
        "No acute cardiopulmonary process.",
        "Mild cardiomegaly with clear lung fields."
    ]
}

# Reference ground truth
references = [
    "No acute cardiopulmonary process.",
    "Mild cardiomegaly with clear lung fields."
]

# Initialise evaluators only for selected metrics
bleu_evaluator = RadEval(do_bleu=True)
rouge_evaluator = RadEval(do_rouge=True)

# Wrap metrics into callable functions
metrics = {
    'bleu': lambda hyps, refs: bleu_evaluator(refs, hyps)['bleu'],
    'rouge1': lambda hyps, refs: rouge_evaluator(refs, hyps)['rouge1'],
}

# Run statistical test
signatures, scores = compare_systems(
    systems=systems,
    metrics=metrics, 
    references=references,
    n_samples=50,           # Number of bootstrap samples
    print_results=True      # Print significance table
)
```

<details>
<summary> Output </summary>

<pre lang="md">
================================================================================
PAIRED SIGNIFICANCE TEST RESULTS
================================================================================
System                                             bleu         rouge1
----------------------------------------------------------------------
Baseline: baseline                              0.0000         0.3968   
----------------------------------------------------------------------
improved                                      1.0000         1.0000   
                                           (p=0.4800)     (p=0.4600)  
----------------------------------------------------------------------
- Significance level: 0.05
- '*' indicates significant difference (p < significance level)
- Null hypothesis: systems are essentially the same
- Significant results suggest systems are meaningfully different

METRIC SIGNATURES:
- bleu: bleu|ar:50|seed:12345
- rouge1: rouge1|ar:50|seed:12345
</pre>

</details>

### Example 4: File-based Evaluation
Recommended for batch evaluation of large sets of generated reports.
```python
import json
from RadEval import RadEval

def evaluate_from_files():
    def read_reports(filepath):
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    refs = read_reports('ground_truth.tok')
    hyps = read_reports('model_predictions.tok')
    
    evaluator = RadEval(
        do_radgraph=True,
        do_bleu=True,
        do_bertscore=True,
        do_chexbert=True
    )
    
    results = evaluator(refs=refs, hyps=hyps)
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results
```

## 📊 Evaluation Metrics

RadEval currently supports the following evaluation metrics:

| Category | Metric | Description | Best For |
|----------|--------|-------------|----------|
| **Lexical** | BLEU | N-gram overlap measurement | Surface-level similarity |
| | ROUGE | Recall-oriented evaluation | Content coverage |
| **Semantic** | BERTScore | BERT-based semantic similarity | Semantic meaning preservation |
| | RadEval BERTScore | Domain-adapted ModernBertModel evaluation | Medical text semantics |
| **Clinical** | CheXbert | Clinical finding classification | Medical accuracy |
| | RadGraph | Knowledge graph-based evaluation | Clinical relationship accuracy |
| | RaTEScore |  Entity-level assessments | Medical synonyms |
| **Specialized** | RadCLIQ | Composite multiple metrics | Clinical relevance |
| | SRR-BERT | Structured report evaluation | Report structure quality |
| | Temporal F1  | Time-sensitive evaluation | Temporal consistency |
| | GREEN | LLM-based metric | Overall radiology report quality |

## 🔧 Configuration Options

### RadEval Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `do_radgraph` | bool | False | Enable RadGraph evaluation |
| `do_green` | bool | False | Enable GREEN metric |
| `do_bleu` | bool | False | Enable BLEU evaluation |
| `do_rouge` | bool | False | Enable ROUGE metrics |
| `do_bertscore` | bool | False | Enable BERTScore |
| `do_srr_bert` | bool | False | Enable SRR-BERT |
| `do_chexbert` | bool | False | Enable CheXbert classification |
| `do_temporal` | bool | False | Enable temporal evaluation |
| `do_ratescore` | bool | False | Enable RateScore |
| `do_radcliq` | bool | False | Enable RadCLIQ |
| `do_radeval_bertsore` | bool | False | Enable RadEval BERTScore |
| `do_details` | bool | False | Include detailed metrics |

### Example Configurations

```python
# Lightweight evaluation (fast)
light_evaluator = RadEval(
    do_bleu=True,
    do_rouge=True
)

# Medical focus (clinical accuracy)
medical_evaluator = RadEval(
    do_radgraph=True,
    do_chexbert=True,
    do_green=True
)

# Comprehensive evaluation (all metrics)
full_evaluator = RadEval(
    do_radgraph=True,
    do_green=True,
    do_bleu=True,
    do_rouge=True,
    do_bertscore=True,
    do_srr_bert=True,
    do_chexbert=True,
    do_temporal=True,
    do_ratescore=True,
    do_radcliq=True,
    do_radeval_bertsore=True,
    do_details=False           # Optional: return detailed metric breakdowns
)
```

## 📁 File Format Suggestion

To ensure efficient evaluation, we recommend formatting your data in one of the following ways:

### 📄 Text Files (.tok, .txt)
Each line contains one report
```
No acute cardiopulmonary process.
Mild cardiomegaly noted.
Normal chest radiograph.
```
Use two separate files:
> - ground_truth.tok — reference reports
> - model_predictions.tok — generated reports

### 🧾 JSON Files
```json
{
  "references": [
    "No acute cardiopulmonary process.",
    "Mild cardiomegaly noted."
  ],
  "hypotheses": [
    "Normal chest X-ray.",
    "Enlarged heart observed."
  ]
}
```

### 🐍 Python Lists
```python
refs = ["Report 1", "Report 2"]
hyps = ["Generated 1", "Generated 2"]
```
> [!TIP]
> File-based input is recommended for batch evaluation and reproducibility in research workflows.


## 🧪 Hypothesis Testing (Significance Evaluation)
RadEval supports **paired significance testing** to statistically compare different radiology report generation systems using **Approximate Randomization (AR)**.

This allows you to determine whether an observed improvement in metric scores is **statistically significant**, rather than due to chance.

### 📌 Key Features

- **Paired comparison** of any number of systems against a baseline
- **Statistical rigor** using Approximate Randomization (AR) testing
- **All built-in metrics** supported (BLEU, ROUGE, BERTScore, RadGraph, CheXbert, etc.)  
- **Custom metrics** integration for domain-specific evaluation
- **P-values** and significance markers (`*`) for easy interpretation

### 🧮 Statistical Background

The hypothesis testing uses **Approximate Randomization** to determine if observed metric differences are statistically significant:

1. **Null Hypothesis (H₀)**: The two systems perform equally well
2. **Test Statistic**: Difference in metric scores between systems
3. **Randomization**: Shuffle system assignments and recalculate differences
4. **P-value**: Proportion of random shuffles with differences ≥ observed
5. **Significance**: If p < 0.05, reject H₀ (systems are significantly different)

> [!NOTE]
> **Why AR testing?** 
> Unlike parametric tests, AR makes no assumptions about score distributions, making it ideal for evaluation metrics that may not follow normal distributions.

### 👀 Understanding the Results

**Interpreting P-values:**
- **p < 0.05**: Statistically significant difference (marked with `*`)
- **p ≥ 0.05**: No significant evidence of difference
- **Lower p-values**: Stronger evidence of real differences

**Practical Significance:**
- Look for consistent improvements across multiple metrics
- Consider domain relevance (e.g., RadGraph for clinical accuracy)  
- Balance statistical and clinical significance

### 🖇️ Example: Compare RadEval Default Metrics and a Custom Metric

#### Step 1: Initialize packages and dataset
```python
from RadEval import RadEval, compare_systems

# Reference ground truth reports
references = [
    "No acute cardiopulmonary process.",
    "No radiographic findings to suggest pneumonia.",
    "Mild cardiomegaly with clear lung fields.",
    "Small pleural effusion on the right side.",
    "Status post cardiac surgery with stable appearance.",
]
# Three systems: baseline, improved, and poor
systems = {
    'baseline': [
        "No acute findings.",
        "No pneumonia.",
        "Mild cardiomegaly, clear lungs.",
        "Small right pleural effusion.",
        "Post-cardiac surgery, stable."
    ],
    'improved': [
        "No acute cardiopulmonary process.",
        "No radiographic findings suggesting pneumonia.",
        "Mild cardiomegaly with clear lung fields bilaterally.",
        "Small pleural effusion present on the right side.",
        "Status post cardiac surgery with stable appearance."
    ],
    'poor': [
        "Normal.",
        "OK.",
        "Heart big.",
        "Some fluid.",
        "Surgery done."
    ]
}
```

#### Step 2: Define Evaluation Metrics and Parameters
We define each evaluation metric using a dedicated RadEval instance (configured to compute one specific score), and also include a simple custom metric — average word count. All metrics are wrapped into a unified metrics dictionary for flexible evaluation and comparison.

```python
# Initialise each evaluator with the corresponding metric
bleu_evaluator = RadEval(do_bleu=True)
rouge_evaluator = RadEval(do_rouge=True)
bertscore_evaluator = RadEval(do_bertscore=True)
radgraph_evaluator = RadEval(do_radgraph=True)
chexbert_evaluator = RadEval(do_chexbert=True)

# Define a custom metric: average word count of generated reports
def word_count_metric(hyps, refs):
    return sum(len(report.split()) for report in hyps) / len(hyps)

# Wrap metrics into a unified dictionary of callables
metrics = {
    'bleu': lambda hyps, refs: bleu_evaluator(refs, hyps)['bleu'],
    'rouge1': lambda hyps, refs: rouge_evaluator(refs, hyps)['rouge1'],
    'rouge2': lambda hyps, refs: rouge_evaluator(refs, hyps)['rouge2'],
    'rougeL': lambda hyps, refs: rouge_evaluator(refs, hyps)['rougeL'],
    'bertscore': lambda hyps, refs: bertscore_evaluator(refs, hyps)['bertscore'],
    'radgraph': lambda hyps, refs: radgraph_evaluator(refs, hyps)['radgraph_partial'],
    'chexbert': lambda hyps, refs: chexbert_evaluator(refs, hyps)['chexbert-5_macro avg_f1-score'],
    'word_count': word_count_metric  # ← example of a simple custom-defined metric
}
```

> [!TIP] 
> - Each metric function takes (hyps, refs) as input and returns a single float score.
> - This modular design allows you to flexibly plug in or remove metrics without changing the core logic of RadEval or compare_systems.
> - For advanced, you may define your own `RadEval(do_xxx=True)` variant or custom metrics and include them seamlessly here.

#### Step 3 Run significance testing

Use `compare_systems` to evaluate all defined systems against the reference reports using the metrics specified above. This step performs randomization-based significance testing to assess whether differences between systems are statistically meaningful.

```python
print("Running significance tests...")

signatures, scores = compare_systems(
    systems=systems,
    metrics=metrics,
    references=references,
    n_samples=50,                    # Number of randomization samples
    significance_level=0.05,         # Alpha level for significance testing
    print_results=True              # Print formatted results table
)
```

<details>
<summary> Output </summary>

<pre lang="md">
Running tests...
================================================================================
PAIRED SIGNIFICANCE TEST RESULTS
================================================================================
System                                             bleu         rouge1         rouge2         rougeL      bertscore       radgraph       chexbert     word_count
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Baseline: baseline                              0.0000         0.6652         0.3133         0.6288         0.6881         0.5538         1.0000         3.2000   
----------------------------------------------------------------------------------------------------------------------------------------------------------------
improved                                      0.6874         0.9531         0.8690         0.9531         0.9642         0.9818         1.0000         6.2000   
                                           (p=0.0000)*    (p=0.0800)     (p=0.1200)     (p=0.0600)     (p=0.0400)*    (p=0.1200)     (p=1.0000)     (p=0.0600)  
----------------------------------------------------------------------------------------------------------------------------------------------------------------
poor                                          0.0000         0.0444         0.0000         0.0444         0.1276         0.0000         0.8000         1.6000   
                                           (p=0.4000)     (p=0.0400)*    (p=0.0600)     (p=0.1200)     (p=0.0400)*    (p=0.0200)*    (p=1.0000)     (p=0.0400)* 
----------------------------------------------------------------------------------------------------------------------------------------------------------------
- Significance level: 0.05
- '*' indicates significant difference (p < significance level)
- Null hypothesis: systems are essentially the same
- Significant results suggest systems are meaningfully different

METRIC SIGNATURES:
- bleu: bleu|ar:50|seed:12345
- rouge1: rouge1|ar:50|seed:12345
- rouge2: rouge2|ar:50|seed:12345
- rougeL: rougeL|ar:50|seed:12345
- bertscore: bertscore|ar:50|seed:12345
- radgraph: radgraph|ar:50|seed:12345
- chexbert: chexbert|ar:50|seed:12345
- word_count: word_count|ar:50|seed:12345
</pre>

</details>

> [!TIP]
> - The output includes mean scores for each metric and system, along with p-values comparing each system to the baseline.
> - Statistically significant improvements (or declines) are marked with an asterisk `*` if p < 0.05.
> - `signatures` stores each metric configuration (e.g. random seed, sample size), and `scores` contains raw score values per system for further analysis or plotting.

#### Step 4: Summarise Significant Findings

```python
# Significance testing
print("\nSignificant differences (p < 0.05):")
baseline_name = list(systems.keys())[0] # Assume first one is the baseline

for system_name in systems.keys():
    if system_name == baseline_name:
        continue
        
    significant_metrics = []
    for metric_name in metrics.keys():
        pvalue_key = f"{metric_name}_pvalue"
        if pvalue_key in scores[system_name]:
            p_val = scores[system_name][pvalue_key]
            if p_val < 0.05:
                significant_metrics.append(metric_name)
    
    if significant_metrics:
        print(f"  {system_name} vs {baseline_name}: {', '.join(significant_metrics)}")
    else:
        print(f"  {system_name} vs {baseline_name}: No significant differences")
```

<details>
<summary> Output </summary>

<pre lang="md">
Significant differences (p < 0.05):
  improved vs baseline: bleu, bertscore
  poor vs baseline: rouge1, bertscore, radgraph, word_count
</pre>

</details>

> [!TIP]
> This makes it easy to:
> - Verify whether model improvements are meaningful
> - Test new metrics or design your own
> - Report statistically sound results in your paper

## 🧠 RadEval Expert Dataset
To support reliable benchmarking, we introduce the **RadEval Expert Dataset**, a carefully curated evaluation set annotated by board-certified radiologists. This dataset consists of realistic radiology reports and challenging model generations, enabling nuanced evaluation across clinical accuracy, temporal consistency, and language quality. It serves as a gold standard to validate automatic metrics and model performance under expert review.

## 🚦 Performance Tips

1. **Start Small**: Test with a few examples before full evaluation
2. **Select Metrics**: Only enable metrics you actually need
3. **Batch Processing**: Process large datasets in smaller chunks
4. **GPU Usage**: Ensure CUDA is available for faster computation


## 📚 Citation

If you use RadEval in your research, please cite:

```BibTeX
@software{radeval2025,
  author = {Jean-Benoit Delbrouck, Justin Xu, Xi Zhang},
  title = {RadEval: A framework for radiology text evaluation},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/jbdel/RadEval}},
}
```

### 📦 Codebase Contributors
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

## 🙏 Acknowledgments

This project would not be possible without the foundational work of the radiology AI community.  
We extend our gratitude to the authors and maintainers of the following open-source projects and metrics:

- 🧠 **CheXbert**, **RadGraph**, and **CheXpert** from Stanford AIMI for their powerful labelers and benchmarks.
- 📐 **BERTScore** and **BLEU/ROUGE** for general-purpose NLP evaluation.
- 🏥 **RadCliQ** and **RaTE Score** for clinically grounded evaluation of radiology reports.
- 🧪 **SRR-BERT** for structured report understanding in radiology.
- 🔍 Researchers contributing to temporal and factual consistency metrics in medical imaging.

Special thanks to:
- All contributors to open datasets such as **MIMIC-CXR**, which make reproducible research possible.
- Our collaborators for their support and inspiration throughout development.

We aim to build on these contributions and promote accessible, fair, and robust evaluation of AI-generated radiology text.


---

<div align="center">
  <p>⭐ If you find RadEval useful, please give us a star! ⭐</p>
  <p>Made with ❤️ for the radiology AI research community</p>
</div>