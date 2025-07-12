import pytest
import json
import re
import nltk
from RadEval.factual.SRRBert.srr_bert import srr_bert_parse_sentences

nltk.download("punkt", quiet=True)


def test_radeval():
    # Sample references and hypotheses
    refs = [
        "1. No acute cardiopulmonary process 2. sentence",
        "1.Status post median sternotomy for CABG with stable.No pleural effusions or pneumothoraces.",
        "1. Left PICC tip appears.2.    Mild pulmonary vascular congestion.3.Interval improve.",
        "2.    Crowding edema 1.hello",
        "2.    Crowding edema 1. hello",
        "2.    Crowding edema 1. Hello",
        "2.Crowding edema 1.Hello",
        "2.Crowding edema1.Hello",
        "No pleural effusions or pneumothoraces.sentence2. mass is 3.5 cm.3. sentence",
        "Tip of endotracheal tube terminates 6.2 cm above the carina, and a nasogastric tube courses below the diaphragm, but tip is not visualized on this study, located outside the field of view. Cardiac silhouette is enlarged, and accompanied by mild pulmonary vascular congestion and minimal interstitial edema. Bibasilar areas of atelectasis are also demonstrated."
    ]

    expected_outputs = [
        ['No acute cardiopulmonary process.', 'sentence.'],
        ['Status post median sternotomy for CABG with stable.', 'No pleural effusions or pneumothoraces.'],
        ['Left PICC tip appears.', 'Mild pulmonary vascular congestion.', 'Interval improve.'],
        ['Crowding edema.', 'hello.'],
        ['Crowding edema.', 'hello.'],
        ['Crowding edema.', 'Hello.'],
        ['Crowding edema.', 'Hello.'],
        ['Crowding edema.', 'Hello.'],
        ['No pleural effusions or pneumothoraces.sentence.', 'mass is 3.5 cm.', 'sentence.'],
        ["Tip of endotracheal tube terminates 6.2 cm above the carina, and a nasogastric tube courses below the diaphragm, but tip is not visualized on this study, located outside the field of view.","Cardiac silhouette is enlarged, and accompanied by mild pulmonary vascular congestion and minimal interstitial edema.","Bibasilar areas of atelectasis are also demonstrated."]
    ]

    for i, ref in enumerate(refs):
        parsed = srr_bert_parse_sentences(ref)
        # print(parsed)
        assert parsed == expected_outputs[i], f"Failed on example {i}. Got {parsed}, expected {expected_outputs[i]}"

if __name__ == "__main__":
    test_radeval()
