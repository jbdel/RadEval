"""HopprF1CheXbert: ModernBERT-based radiology report evaluator (Hoppr labels)."""
from __future__ import annotations

import os
import warnings
from typing import Union

import torch
from transformers import ModernBertModel, AutoTokenizer

from RadEval.metrics._chexbert_base import BaseBertLabeler, BaseCheXbertEvaluator

_DEFAULT_HOPPR_CKPT = "/fss/pranta_das/CheXbert/expermints_folder/25507/checkpoint_2.pth"


class _HopprLabeler(BaseBertLabeler):
    """Hoppr CheXbert backbone (ModernBERT-base, 26+1 heads)."""

    def __init__(self, *, device, checkpoint_path: str = _DEFAULT_HOPPR_CKPT):
        backbone = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"HopprCheXbert checkpoint not found: {checkpoint_path}. "
                "Pass checkpoint_path= to HopprF1CheXbert() or set the file path."
            )
        state = torch.load(checkpoint_path, map_location="cpu")["model"]

        super().__init__(
            device=device,
            backbone=backbone,
            num_4way_heads=26,
            state_dict=state,
        )


class HopprF1CheXbert(BaseCheXbertEvaluator):
    """Hoppr CheXbert label extraction + F1 evaluation (27 conditions)."""

    CONDITION_NAMES = [
        'acute_rib_fracture', 'air_space_opacity', 'cardiomegaly',
        'lung_nodule_or_mass', 'non_acute_rib_fracture', 'pleural_fluid',
        'pneumothorax', 'pulmonary_artery_enlargement', 'atelectasis',
        'bronchial_wall_thickening', 'bullous_disease',
        'hilar_lymphadenopathy', 'hiatus_hernia', 'hyperinflation',
        'implantable_electronic_device', 'intercostal_drain',
        'interstitial_thickening', 'lobar_segmental_collapse',
        'nonsurgical_internal_foreign_body',
        'pacemaker_electronic_cardiac_device_or_wires', 'peribronchial_cuffing',
        'pulmonary_congestion_pulmonary_venous_congestion',
        'shoulder_dislocation', 'subcutaneous_emphysema', 'tracheal_deviation',
        'whole_lung_or_majority_collapse',
    ]
    NO_FINDING = "no_finding"
    TOP5 = ["cardiomegaly", "air_space_opacity", "atelectasis", "pleural_fluid"]

    def __init__(
        self,
        *,
        refs_filename: str | None = None,
        hyps_filename: str | None = None,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 64,
        checkpoint_path: str = _DEFAULT_HOPPR_CKPT,
    ):
        super().__init__()

        if isinstance(device, str):
            device_obj = torch.device(device)
        else:
            device_obj = device
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            device_obj = torch.device("cpu")

        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = _HopprLabeler(
            device=device_obj, checkpoint_path=checkpoint_path).eval()
        self._init_evaluator(
            model=model, tokenizer=tokenizer, device=device_obj,
            batch_size=batch_size, refs_filename=refs_filename,
            hyps_filename=hyps_filename,
        )
