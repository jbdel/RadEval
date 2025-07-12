from RadEval.nlg.bertscore.bertscore import BertScore

def radcliq_bertscore(refs, hyps):
    bertscore_scorer = BertScore(model_type='distilroberta-base',
                                rescale_with_baseline=True,
                                idf=False,
                                num_layers=None)
    print(bertscore_scorer)
    avg, scores = bertscore_scorer(refs, hyps)
    return scores