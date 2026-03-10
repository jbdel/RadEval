import json
from radgraph import RadGraph


def compute_f1(test, retrieved):
    """Computes F1 between test/retrieved report's entities or relations."""
    tp = len(test & retrieved)
    fp = len(retrieved) - tp
    fn = len(test) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0


def extract_entities(output):
    """Extracts set of (tokens, label) from a RadGraph output dict."""
    return {(tuple(ent["tokens"]), ent["label"]) for ent in output.get("entities", {}).values()}


def extract_relations(output):
    """Extracts set of (src, tgt, relation) from a RadGraph output dict."""
    rels = set()
    entities = output.get("entities", {})
    for ent in entities.values():
        src = (tuple(ent["tokens"]), ent["label"])
        for rel_type, tgt_idx in ent.get("relations", []):
            tgt_ent = entities.get(tgt_idx)
            if tgt_ent:
                tgt = (tuple(tgt_ent["tokens"]), tgt_ent["label"])
                rels.add((src, tgt, rel_type))
    return rels


def compute_radgraph_scores(refs, hyps, model_name='radgraph'):
    """
    Computes combined RadGraph F1 scores for each pair of reference and hypothesis reports.
    Returns:
      List of floats: (entity_f1 + relation_f1)/2 per report.
    """
    # Initialize RadGraph model
    rad = RadGraph(model_type=model_name)

    # Perform inference
    gt_outputs = rad(refs)
    pred_outputs = rad(hyps)

    scores = []
    for i in range(len(gt_outputs)):
        gt_out = gt_outputs[str(i)]
        pred_out = pred_outputs[str(i)]
        
        gt_ents = extract_entities(gt_out)
        pred_ents = extract_entities(pred_out)
        gt_rels = extract_relations(gt_out)
        pred_rels = extract_relations(pred_out)

        ent_f1 = compute_f1(gt_ents, pred_ents)
        rel_f1 = compute_f1(gt_rels, pred_rels)
        scores.append((ent_f1 + rel_f1) / 2)

    return scores


if __name__ == '__main__':
    # Example usage
    refs = [
        "No evidence of pneumothorax following chest tube removal.",
        "There is a left pleural effusion."
    ]
    hyps = [
        "No pneumothorax detected.",
        "Left pleural effusion is present."
    ]

    combined_scores = compute_radgraph_scores(refs, hyps)
    print(combined_scores)  # e.g., [1.0, 1.0]
    from radgraph import F1RadGraph
    f1_radgraph = F1RadGraph(model_type="radgraph", reward_level="simple")
    f1_scores = f1_radgraph(refs, hyps,)
    print(f1_scores)