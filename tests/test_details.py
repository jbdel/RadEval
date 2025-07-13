from RadEval import RadEval
import pytest

def test_do_details():
    """
    This function is used to run the test for RadEval with do_details=True.
    It is useful for debugging and ensuring that the test runs correctly.
    """
    refs = [
        "Increased mild pulmonary edema and left basal atelectasis.",
    ]

    hyps = [
        "No pleural effusions or pneumothoraces.",
    ]

    # Instantiate RadEval with desired configurations
    evaluator = RadEval(do_radgraph=True,
                      do_green=False,
                      do_bleu=True,
                      do_rouge=True,
                      do_bertscore=True,
                      do_srr_bert=True,
                      do_chexbert=True,
                      do_temporal=True,
                      do_ratescore=True,
                      do_radcliq=True,
                      do_radeval_bertsore=True,
                      do_details=True)
    
    # Run the evaluation
    results = evaluator(refs=refs, hyps=hyps)
    
    # 1. Check that all expected evaluation metric keys are present
    expected_keys = {'radgraph', 'bleu', 'bertscore', 'rouge', 'srr_bert', 
                    'chexbert', 'ratescore', 'radcliq-v1', 'temporal_f1', 'radeval_bertsore'}
    actual_keys = set(results.keys())
    assert expected_keys == actual_keys, f"Key mismatch. Expected: {expected_keys}, Actual: {actual_keys}"
    
    # 2. Check radgraph detailed information structure (key functionality of do_details=True)
    radgraph_result = results['radgraph']
    
    # Check basic score fields
    assert 'radgraph_simple' in radgraph_result
    assert 'radgraph_partial' in radgraph_result
    assert 'radgraph_complete' in radgraph_result
    assert 'reward_list' in radgraph_result
    
    # Check detailed annotation fields (core functionality of do_details=True)
    assert 'hypothesis_annotation_lists' in radgraph_result, "Missing hypothesis_annotation_lists"
    assert 'reference_annotation_lists' in radgraph_result, "Missing reference_annotation_lists"
    
    # Check annotation list structure
    hyp_annotations = radgraph_result['hypothesis_annotation_lists']
    ref_annotations = radgraph_result['reference_annotation_lists']
    
    assert len(hyp_annotations) == len(hyps), f"Hypothesis annotation count mismatch: {len(hyp_annotations)} vs {len(hyps)}"
    assert len(ref_annotations) == len(refs), f"Reference annotation count mismatch: {len(ref_annotations)} vs {len(refs)}"
    
    # Check detailed annotation structure
    for i, annotation in enumerate(hyp_annotations):
        assert 'text' in annotation, f"Hypothesis annotation {i} missing text field"
        assert 'entities' in annotation, f"Hypothesis annotation {i} missing entities field"
        assert 'data_source' in annotation, f"Hypothesis annotation {i} missing data_source field"
        assert 'data_split' in annotation, f"Hypothesis annotation {i} missing data_split field"
        assert isinstance(annotation['entities'], dict), f"Hypothesis annotation {i} entities is not a dictionary"
    
    for i, annotation in enumerate(ref_annotations):
        assert 'text' in annotation, f"Reference annotation {i} missing text field"
        assert 'entities' in annotation, f"Reference annotation {i} missing entities field"
        assert 'data_source' in annotation, f"Reference annotation {i} missing data_source field"
        assert 'data_split' in annotation, f"Reference annotation {i} missing data_split field"
        assert isinstance(annotation['entities'], dict), f"Reference annotation {i} entities is not a dictionary"
    
    # 3. Check basic structure and data types for each metric
    
    # BLEU scores
    bleu_result = results['bleu']
    for key in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']:
        assert key in bleu_result, f"BLEU missing {key}"
        assert isinstance(bleu_result[key], (int, float)), f"BLEU {key} is not a numeric type"
    
    # BERTScore
    bertscore_result = results['bertscore']
    assert 'mean_score' in bertscore_result
    assert 'sample_scores' in bertscore_result
    assert isinstance(bertscore_result['sample_scores'], list)
    assert len(bertscore_result['sample_scores']) == len(hyps)
    
    # ROUGE
    rouge_result = results['rouge']
    for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
        assert rouge_type in rouge_result
        assert 'mean_score' in rouge_result[rouge_type]
        assert 'sample_scores' in rouge_result[rouge_type]
    
    # CheXbert
    chexbert_result = results['chexbert']
    assert 'label_scores_f1-score' in chexbert_result
    assert 'chexbert-5' in chexbert_result['label_scores_f1-score']
    assert 'chexbert_all' in chexbert_result['label_scores_f1-score']
    
    # RateScore
    ratescore_result = results['ratescore']
    assert 'f1-score' in ratescore_result
    assert 'hyps_pairs' in ratescore_result
    assert 'refs_pairs' in ratescore_result
    assert isinstance(ratescore_result['hyps_pairs'], list)
    assert isinstance(ratescore_result['refs_pairs'], list)
    
    # Temporal F1
    temporal_result = results['temporal_f1']
    assert 'f1-score' in temporal_result
    assert 'hyps_entities' in temporal_result
    assert 'refs_entities' in temporal_result
    
    # 4. Functional validation: ensure do_details=True actually generates detailed information
    # Check that entity annotations contain expected medical terms
    hyp_text = hyp_annotations[0]['text']
    ref_text = ref_annotations[0]['text']
    
    assert "pleural" in hyp_text.lower() or "pneumothoraces" in hyp_text.lower(), "Hypothesis text should contain relevant medical terms"
    assert "edema" in ref_text.lower() or "atelectasis" in ref_text.lower(), "Reference text should contain relevant medical terms"
    
    # Check that entity dictionaries contain medical entities
    hyp_entities = hyp_annotations[0]['entities']
    ref_entities = ref_annotations[0]['entities']
    
    assert len(hyp_entities) > 0, "Hypothesis annotation should contain entities"
    assert len(ref_entities) > 0, "Reference annotation should contain entities"
    
    # Check entity structure
    for entity_id, entity in hyp_entities.items():
        assert 'tokens' in entity, f"Entity {entity_id} missing tokens"
        assert 'label' in entity, f"Entity {entity_id} missing label"
        assert 'start_ix' in entity, f"Entity {entity_id} missing start_ix"
        assert 'end_ix' in entity, f"Entity {entity_id} missing end_ix"
        assert 'relations' in entity, f"Entity {entity_id} missing relations"
    
    return True

if __name__ == "__main__":
    # Run the test
    test_do_details()