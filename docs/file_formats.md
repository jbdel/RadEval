[Back to README](../README.md)

# File Formats

RadEval takes Python lists of strings. This page shows how to load data from common file formats.

## Text files (.tok, .txt)

One report per line:

```
No acute cardiopulmonary process.
Mild cardiomegaly noted.
Normal chest radiograph.
```

```python
def read_reports(filepath):
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]

refs = read_reports("ground_truth.tok")
hyps = read_reports("model_predictions.tok")
```

## JSON

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

```python
import json

with open("data.json") as f:
    data = json.load(f)

refs = data["references"]
hyps = data["hypotheses"]
```

## JSONL (one object per line)

```jsonl
{"ref": "No acute findings.", "hyp": "Clear lungs."}
{"ref": "Mild cardiomegaly.", "hyp": "Heart is enlarged."}
```

```python
import json

with open("data.jsonl") as f:
    rows = [json.loads(line) for line in f]

refs = [r["ref"] for r in rows]
hyps = [r["hyp"] for r in rows]
```

## CSV / Pandas

```python
import pandas as pd

df = pd.read_csv("reports.csv")
refs = df["reference"].tolist()
hyps = df["hypothesis"].tolist()
```

## Evaluate

Once loaded, pass the lists to RadEval:

```python
from RadEval import RadEval

evaluator = RadEval(metrics={"bleu": {}, "rouge": {}})
results = evaluator(refs=refs, hyps=hyps)
```
