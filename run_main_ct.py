# srun --time=24:00:00 --gpus=1 --cpus-per-task=64 --mem=128G --pty bash -l
# conda activate radeval
from dotenv import load_dotenv
load_dotenv()

from RadEval import RadEval
import json


def main():

    path = "/fss/jb/vlm/ckpt/ctc_rrg_ctc_internal_v1_22_qformer_tuned_overfit_guard_true_290/predictions/pred_ref_epoch46_seed518596_val.jsonl"
    prediction_filename = path.split("/")[-1].split(".")[0] + ".json"
    print(prediction_filename)

    preds = []
    gts = []
    for line in open(path, "r"):
        data = json.loads(line)
        if data["pred"] is None or data["gt"] is None or data["pred"] == "" or data["gt"] == "":
            continue
        preds.append(data["pred"].strip())
        gts.append(data["gt"].strip())

    print(len(preds), len(gts))

    evaluator = RadEval(
        # do_rouge=True,
        # do_bleu=True,
        # do_bertscore=True,
        # do_f1radbert_ct=True,
        # do_f1hopprchexbert_ct=True,
        # do_radeval_bertscore=True,
        do_radfact_ct=True,
        do_details=True,
    )

    results = evaluator(refs=gts, hyps=preds)
    print(json.dumps(results, indent=4))

    with open(prediction_filename, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
