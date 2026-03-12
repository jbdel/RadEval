# srun --time=24:00:00 --gpus=1 --cpus-per-task=64 --mem=128G --pty bash -l
# conda activate radeval
from RadEval import RadEval
import json


def main():

    # path = "/fss/jb/vlm/ckpt/cxr_rrg_internal_temporal_1M_v001_v2_true_fixedhyperparams/predictions/pred_ref_epoch40_seed876032_val.jsonl"
    # path = "/fss/jb/vlm/ckpt/cxr_rrg_internal_non_temporal_FULL_v002_22_qformer_true/predictions/pred_ref_epoch14_seed998798_val.jsonl"
    # path = "/fss/jb/vlm/ckpt/cxr_rrg_internal_temporal_1M_v001_v2_true_fixedhyperparams/predictions/pred_ref_epoch40_seed876032_val.jsonl"
    path = "/fss/jb/vlm/ckpt/cxr_rrg_internal_non_temporal_FULL_v002_22_qformer_true/predictions/pred_ref_epoch26_seed998798_test_inference_eager.jsonl"
    path = "/fss/jb/RadEval/pred_ref_epoch37_seed476104_val.jsonl"
    path = "/fss/jb/vlm/ckpt/cxr_rrg_internal_non_temporal_FULL_v002_qwen3_vl_true/predictions/pred_ref_epoch37_seed476104_test_inference.jsonl"
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
        do_radgraph=True,
        do_rouge=True,
        do_bleu=True,
        do_bertscore=True,
        do_f1hopprchexbert=True,
        do_f1chexbert=True,
        do_radeval_bertscore=True,
        do_details=True,
    )

    results = evaluator(refs=gts, hyps=preds)
    print(json.dumps(results, indent=4))

    with open(prediction_filename, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
