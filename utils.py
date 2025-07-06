import re
import nltk

nltk.download("punkt_tab", quiet=True)

def clean_numbered_list(text):
    """
    Clean a report if it's a numbered list by:
    1. Adding proper spacing between numbered items
    2. Removing the numbered list markers
    3. Adding spaces after periods between sentences
    """
    # First, separate numbered items that are stuck together without spaces
    # Example: "textx.2. text2" -> "texty. 2. text2"
    text = re.sub(r'\.(\d+\.)', r'. \1', text)
    
    # Handle patterns where there's no period between numbered entries
    # Example: "1. item1 2. item2" -> "1. item1. 2. item2"
    text = re.sub(r'(\d+\.\s*[^.]+?)\s+(?=\d+\.)', r'\1. ', text)
    
    # Then remove the numbered list markers
    # But avoid removing decimal numbers in measurements like "3.5 cm"
    text = re.sub(r'(?<!\d)\d+\.\s*', '', text)
    
    # Add spaces after periods between sentences if missing
    # Example: "sentence1.sentence2" -> "sentence1. sentence2"
    # But don't split decimal numbers like "3.5 cm"
    text = re.sub(r'\.([A-Za-z])', r'. \1', text)
    return nltk.sent_tokenize(text)