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
    evaluator = RadEval(do_radgraph=False,
                        do_green=False,
                        do_bleu=True,
                        do_rouge=False,
                        do_bertscore=False,
                        do_diseases=False,
                        do_chexbert=False,
                        do_ratescore=True,
                        do_radcliq=True,
                        do_temporal=True)
    results = evaluator(refs=refs, hyps=hyps)
    print(json.dumps(results, indent=4))
main()
```