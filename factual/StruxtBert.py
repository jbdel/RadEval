import numpy as np
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm

from factual.constants import leaves_mapping, upper_mapping, leaves_with_statuses_mapping, upper_with_statuses_mapping


class StruxtBert(nn.Module):
    def __init__(self, model_id_or_path, mapping, tqdm_enable=False):
        super().__init__()

        self.mapping = mapping
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained(model_id_or_path, num_labels=len(mapping))
        self.model.to(self.device)  # move model to GPU if available
        self.model.eval()
        self.tqdm_enable = tqdm_enable

        self.tokenizer = BertTokenizer.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-general'
        )

    def map_predictions_to_labels(self, outputs):
        # invert the mapping so we can look up labels by index (1-based index)
        inverted_mapping = {v: k for k, v in self.mapping.items()}

        all_labels = []
        for output in outputs:
            predicted_labels = [inverted_mapping[i] for i, pred in enumerate(output) if pred == 1]
            all_labels.append(predicted_labels)

        return all_labels

    def forward(self, sentences, batch_size=4):

        # create batches of sentences
        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

        with torch.no_grad():
            outputs = []
            for batch in tqdm(batches, desc="Predicting", disable=not self.tqdm_enable):
                inputs = self.tokenizer.batch_encode_plus(
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
                preds = torch.sigmoid(logits) > 0.5
                outputs.append(preds.cpu().numpy())

        outputs = np.concatenate(outputs, axis=0).astype(int)

        # map predictions to labels
        return outputs, self.map_predictions_to_labels(outputs)


if __name__ == "__main__":
    model = "StanfordAIMI/CXR-BERT-Leaves-Diseases-Only"
    outputs, _ = StruxtBert(model_id_or_path=model, mapping=leaves_mapping, tqdm_enable=True)(
        sentences=[
            "Layering pleural effusions",
            "Moderate pulmonary edema.",
            "Chronic fracture and dislocation involving the left humeral surgical neck and glenoid.",
            "Stable cardiomegaly.",
        ],
    )
    print(outputs)
