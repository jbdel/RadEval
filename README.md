## Installation
> Python 3.10 (recommend)
1. Clone this repository and navigate to RadEval folder
```bash
git clone https://github.com/jbdel/RadEval
cd RadEval
```

2. Install Package
```Shell
conda create -n RadEval python=3.10 -y
conda activate RadEval
pip install -e .
```

## Usage
```python
from RadEval import RadEval
import json

def main():
    refs = [
        "No acute cardiopulmonary process.",
        "No radiographic findings to suggest pneumonia.",
        "1.Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.Relatively lower lung volumes with no focal airspace consolidation appreciated.Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.No pleural effusions or pneumothoraces.",
        "1. Left PICC tip appears to terminate in the distal left brachiocephalic vein.2. Mild pulmonary vascular congestion.3. Interval improvement in aeration of the lung bases with residual streaky opacity likely reflective of atelectasis.Interval resolution of the left pleural effusion.",
        "No definite acute cardiopulmonary process.Enlarged cardiac silhouette could be accentuated by patient's positioning.",
        "Increased mild pulmonary edema and left basal atelectasis.",
    ]

    hyps = [
        "No acute cardiopulmonary process.",
        "No radiographic findings to suggest pneumonia.",
        "Status post median sternotomy for CABG with stable cardiac enlargement and calcification of the aorta consistent with atherosclerosis.",
        "Relatively lower lung volumes with no focal airspace consolidation appreciated.",
        "Crowding of the pulmonary vasculature with possible minimal perihilar edema, but no overt pulmonary edema.",
        "No pleural effusions or pneumothoraces.",
    ]

    evaluator = RadEval(do_radgraph=True,
                        do_green=True,
                        do_bleu=True,
                        do_rouge=True,
                        do_bertscore=True,
                        do_srr_bert=True,
                        do_chexbert=True,
                        do_temporal=True,
                        do_ratescore=True,
                        do_radcliq=True,
                        do_radeval_bertsore=True)

    results = evaluator(refs=refs, hyps=hyps)
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()
```
Output
```
{
    "radgraph_simple": 0.41111111111111115,
    "radgraph_partial": 0.41111111111111115,
    "radgraph_complete": 0.41414141414141414,
    "bleu": 0.16681006823938177,
    "bertscore": 0.63274747133255,
    "green": 0.39999999999999997,
    "rouge1": 0.44681719607092746,
    "rouge2": 0.4205128205128205,
    "rougeL": 0.44681719607092746,
    "srr_bert_weighted_f1": 0.2857142857142857,
    "srr_bert_weighted_precision": 0.2857142857142857,
    "srr_bert_weighted_recall": 0.2857142857142857,
    "chexbert-5_micro avg_f1-score": 0.2857142857142857,
    "chexbert-all_micro avg_f1-score": 0.3333333333333333,
    "chexbert-5_macro avg_f1-score": 0.13333333333333333,
    "chexbert-all_macro avg_f1-score": 0.08333333333333333,
    "chexbert-5_weighted_f1": 0.2222222222222222,
    "chexbert-all_weighted_f1": 0.22916666666666666,
    "ratescore": 0.5877872315410949,
    "radcliq-v1": 1.6447780902700346,
    "temporal_f1": 0.500000000075,
    "radeval_bertsore": 0.4910106658935547
}
```