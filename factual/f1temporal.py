# Temporal Entity F1
# Adopted from https://github.com/X-iZhang/Libra/blob/main/libra/eval/temporal_f1.py

import re
import stanza
import argparse
from typing import List, Union



# Initialize the pipeline with the radiology NER model explicitly specified
nlp = stanza.Pipeline(
    lang='en',
    package='radiology',
    processors={'tokenize': 'default', 'ner': 'radiology'},
    logging_level='ERROR',  # Only output warnings or more severe messages
    verbose=False  # Suppress additional information during pipeline initialization
)

# Keywords used for radiology-related entity extraction
# Reference: Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing (CVPR2023) 
# https://arxiv.org/pdf/2301.04558

KEYWORDS = {
    "bigger", "change", "cleared", "constant", "decrease", "decreased", "decreasing", "elevated", "elevation", 
    "enlarged", "enlargement", "enlarging", "expanded", "greater", "growing", "improved", "improvement", 
    "improving", "increase", "increased", "increasing", "larger", "new", "persistence", "persistent", 
    "persisting", "progression", "progressive", "reduced", "removal", "resolution", "resolved", "resolving", 
    "smaller", "stability", "stable", "stably", "unchanged", "unfolded", "worse", "worsen", "worsened", 
    "worsening", "unaltered"
}

def clean_text(text: str) -> str:
    """
    Clean the input text by removing special characters and redundant spaces or newlines.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    # Remove special characters and redundant newlines
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a single space
    text = re.sub(r'[_-]+', ' ', text)  # Replace underscores and dashes with spaces
    text = re.sub(r'\(___, __, __\)', '', text)  # Remove irrelevant underscore patterns
    text = re.sub(r'---, ---, ---', '', text)  # Remove dashed patterns
    text = re.sub(r'\(__, __, ___\)', '', text)  # Remove similar underscore patterns
    text = re.sub(r'[_-]+', ' ', text)  # Replace underscores and dashes again (if any remain)
    text = re.sub(r'[^\w\s.,:;()-]', '', text)  # Remove non-alphanumeric characters except common punctuation
    
    # Remove extra spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def extract_entities(text: str, keywords: set) -> set:
    """
    Extract entities from the given text based on Stanza NER and provided keywords.

    Args:
        text (str): Input text.
        keywords (set): Set of keywords to extract entities.

    Returns:
        set: Set of matched entities found in the text.
    """
    # Use Stanza NER to extract entities tagged as "OBSERVATION" or "OBSERVATION_MODIFIER"
    doc = nlp(text)
    stanza_entities = {ent.text.lower() for ent in doc.entities if ent.type in {"OBSERVATION", "OBSERVATION_MODIFIER"}}

    # Filter Stanza entities to include only those present in keywords
    matched_stanza_entities = {entity for entity in stanza_entities if entity in keywords}

    # Clean the text before extracting entities
    text = clean_text(text)

    # Create a regex pattern that matches any of the keywords as whole words
    pattern = r'\b(' + '|'.join(re.escape(word) for word in keywords) + r')\b'

    # Find all matches using regex
    keyword_matches = {match.group().lower() for match in re.finditer(pattern, text.lower())}

    # Combine Stanza entities and regex matches
    return matched_stanza_entities | keyword_matches

def calculate_tem_score(prediction_text: str, reference_text: Union[str, List[str]], epsilon: float = 1e-10) -> float:
    """
    Calculate the Temporal Entity Matching (TEM) score (similar to F1-score).

    Args:
        reference_text (Union[str, List[str]]): Reference text or a list of reference texts.
        prediction_text (str): Prediction text.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: TEM score.
    """
    if isinstance(reference_text, list):
        reference_entities = set()
        for ref in reference_text:
            reference_entities.update(extract_entities(ref, KEYWORDS))
    else:
        reference_entities = extract_entities(reference_text, KEYWORDS)

    prediction_entities = extract_entities(prediction_text, KEYWORDS)

    if len(reference_entities) == 0:
        if len(prediction_entities) == 0:
            return {
                "f1": 1.0,
                "prediction_entities": prediction_entities,
                "reference_entities": reference_entities
            }  # Perfect match when both are empty
        else:
            return {
                "f1": epsilon,
                "prediction_entities": prediction_entities,
                "reference_entities": reference_entities
            }  # Minimal score when reference is empty but prediction is not

    # Calculate intersection of entities
    true_positives = len(prediction_entities & reference_entities) 

    # Calculate precision and recall with epsilon to avoid division by zero
    precision = (true_positives + epsilon) / (len(prediction_entities) + epsilon)
    recall = (true_positives + epsilon) / (len(reference_entities) + epsilon)

    # Calculate TEM score (F1 score)
    tem_score = (2 * precision * recall) / (precision + recall + epsilon)
    
    return {
        "f1": tem_score,
        "prediction_entities": prediction_entities,
        "reference_entities": reference_entities
    }

def F1Temporal(predictions: List[str], references: List[Union[str, List[str]]], epsilon: float = 1e-10) -> dict:
    """
    Calculate the average TEM score over a list of reference and prediction texts.

    Args:
        references (List[Union[str, List[str]]]): List of reference texts or lists of reference texts.
        predictions (List[str]): List of prediction texts.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: Average TEM score.
    """
    assert len(references) == len(predictions), "Reference and prediction lists must have the same length."
    
    tem_scores = []
    prediction_entities = []
    reference_entities = []

    for pred, ref in zip(predictions, references):
        result = calculate_tem_score(pred, ref, epsilon)
        tem_scores.append(result["f1"])
        prediction_entities.append(result["prediction_entities"])
        reference_entities.append(result["reference_entities"])

    average_f1 = sum(tem_scores) / len(tem_scores)

    return {
        "f1": average_f1,
        "prediction_entities": prediction_entities,
        "reference_entities": reference_entities
    }
