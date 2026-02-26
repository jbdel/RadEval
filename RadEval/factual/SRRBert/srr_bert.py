import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
import re
import nltk


def srr_bert_parse_sentences(text):
    # Handle numbers followed by a dot, not followed by a digit (to avoid decimals like 3.5)
    
    # Case 1: Number at beginning of text
    text = re.sub(r'^\s*\d+\.(?!\d)\s*', '', text)
    
    # Case 2: Number after a period, like "word.2."
    text = re.sub(r'(\w)\.(\d+)\.(?!\d)\s*', r'\1. ', text)
    
    # Case 3: Number attached to a word, like "word2."
    text = re.sub(r'(\w)(\d+)\.(?!\d)\s*', r'\1. ', text)
    
    # Case 4: Number after space following a word, like "word 2."
    text = re.sub(r'(\w)\s+\d+\.(?!\d)\s*', r'\1. ', text)
    
    # Case 5: Standalone number in the middle, like ". 2. word"
    text = re.sub(r'([.!?])\s*\d+\.(?!\d)\s*', r'\1 ', text)
    
    # Add space after periods followed immediately by uppercase letter (new sentence without space)
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    # Make sure the text ends with a period
    if not text.strip().endswith(('.', '!', '?')):
        text = text.strip() + '.'
    
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)
    
    return sentences
    

class SRRBert(nn.Module):
    # Supported model types and their configs
    MODEL_CONFIGS = {
        "leaves": {
            "model_path": "StanfordAIMI/SRR-BERT-Leaves",
            "mapping_file": "leaves_mapping.json"
        },
        "upper": {
            "model_path": "StanfordAIMI/SRR-BERT-Upper",
            "mapping_file": "upper_mapping.json"
        },
        "leaves_with_statuses": {
            "model_path": "StanfordAIMI/SRR-BERT-Leaves-with-Statuses",
            "mapping_file": "leaves_with_statuses_mapping.json"
        },
        "upper_with_statuses": {
            "model_path": "StanfordAIMI/SRRG-BERT-Upper-with-Statuses",
            "mapping_file": "upper_with_statuses_mapping.json"
        },
    }

    def __init__(
        self,
        model_type: str = "leaves",
        batch_size: int = 4,
        tqdm_enable: bool = False
    ):
        super().__init__()
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(
                f"model_type must be one of {list(self.MODEL_CONFIGS.keys())}"
            )
        config = self.MODEL_CONFIGS[model_type]

        # Load mapping
        mapping_path = os.path.join(
            os.path.dirname(__file__),
            config["mapping_file"]
        )
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)

        # Device setup
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Load model
        self.model = BertForSequenceClassification.from_pretrained(
            config["model_path"],
            num_labels=len(self.mapping)
        )
        self.model.to(self.device)
        self.model.eval()

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-general"
        )

        # Settings
        self.batch_size = batch_size
        self.tqdm_enable = tqdm_enable

    def map_predictions_to_labels(self, outputs):
        inverted_mapping = {v: k for k, v in self.mapping.items()}
        all_labels = []
        for output in outputs:
            labels = [inverted_mapping[i] for i, flag in enumerate(output) if flag == 1]
            all_labels.append(labels)
        return all_labels

    def forward(self, sentences):
        # Batch sentences
        batches = [
            sentences[i:i + self.batch_size]
            for i in range(0, len(sentences), self.batch_size)
        ]
        outputs = []
        with torch.no_grad():
            for batch in tqdm(
                batches, desc="Predicting", disable=not self.tqdm_enable
            ):
                inputs = self.tokenizer(
                    batch,
                    add_special_tokens=True,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
                outputs.append(preds)

        outputs = np.concatenate(outputs, axis=0)
        return outputs, self.map_predictions_to_labels(outputs)

    
if __name__ == "__main__":
    example_sentences = [
        "Layering pleural effusions",
        "Moderate pulmonary edema.",
        "Chronic fracture and dislocation involving the left humeral surgical neck and glenoid.",
        "Stable cardiomegaly.",
    ]

    # Initialize model (choose one of: leaves, upper, leaves_with_statuses, upper_with_statuses)
    model = SRRBert(
        model_type="leaves",
        batch_size=4,
        tqdm_enable=True
    )
    outputs, labels = model(example_sentences)
    print("Raw outputs:", outputs)
    print("Predicted labels:", labels)
