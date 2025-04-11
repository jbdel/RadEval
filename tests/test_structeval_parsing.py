import pytest
import json
import re
import nltk

nltk.download("punkt", quiet=True)

def clean_numbered_list(text):
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
        ['No pleural effusions or pneumothoraces.sentence.', 'mass is 3.5 cm.', 'sentence.']
    ]

    for i, ref in enumerate(refs):
        parsed = clean_numbered_list(ref)
        # print(parsed)
        assert parsed == expected_outputs[i], f"Failed on example {i}. Got {parsed}, expected {expected_outputs[i]}"

if __name__ == "__main__":
    test_radeval()
