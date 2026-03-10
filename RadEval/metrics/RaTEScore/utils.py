import torch
import torch.nn.functional as F
import medspacy
nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_conte"])

def sentence_split(text_list):
    """
    split sentences by medspacy
    """
    clean_text_list = [] 
    is_start_list = []

    for text in text_list:

        doc = nlp(text)

        is_start = 1

        for sent in doc.sents:
            sent = str(sent).strip()
            # # check if the sentence has no words
            if len(sent.split()) == 0:
                continue
            if len(sent) < 3:
                continue
            is_start_list.append(is_start)
            clean_text_list.append(sent)
            is_start = 0

    return clean_text_list, is_start_list

def post_process(tokenized_text, predicted_entities, tokenizer):
    entity_spans = []
    start = end = None
    entity_type = None

    for i, (token, label) in enumerate(zip(tokenized_text, predicted_entities[:len(tokenized_text)])):
        if token in ["[CLS]", "[SEP]"]:
            continue
        if label != "O" and i < len(predicted_entities) - 1:
            if label.startswith("B-") and predicted_entities[i+1].startswith("I-"):
                start = i
                entity_type = label[2:]
            elif label.startswith("B-") and predicted_entities[i+1].startswith("B-"):
                start = i
                end = i
                entity_spans.append((start, end, label[2:]))
                start = i
                entity_type = label[2:]
            elif label.startswith("B-") and predicted_entities[i+1].startswith("O"):
                start = i
                end = i
                entity_spans.append((start, end, label[2:]))
                start = end = None
                entity_type = None
            elif label.startswith("I-") and predicted_entities[i+1].startswith("B-"):
                end = i
                if start is not None:
                    entity_spans.append((start, end, entity_type))
                start = i
                entity_type = label[2:]
            elif label.startswith("I-") and predicted_entities[i+1].startswith("O"):
                end = i
                if start is not None:
                    entity_spans.append((start, end, entity_type))
                start = end = None
                entity_type = None

    # 处理最后一个实体
    if start is not None and end is None:
        end = len(tokenized_text) - 2
        entity_spans.append((start, end, entity_type))

    # 输出结果
    save_pair = []
    for start, end, entity_type in entity_spans:
        entity_str = tokenizer.convert_tokens_to_string(tokenized_text[start:end+1])
        # print(f"实体: {entity_str}, 类型: {entity_type}")
        save_pair.append((entity_str, entity_type))

    return save_pair


def topk_similarity(embeddings1, embeddings2, k=1):
    """
    Compute the top-k similarity between two sets of embeddings using PyTorch.
    """

    ### Normalize the embeddings to use cosine similarity
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    
    topk_values = []
    topk_indices = []

    ### Iterate over each embedding in the first set
    for emb1 in embeddings1:
        
        ### Calculate cosine similarity between this embedding and all embeddings in the second set
        similarities = torch.matmul(embeddings2, emb1)

        ### Find the top-k highest similarity values
        values, indices = torch.topk(similarities, k, largest=True)

        topk_values.append(values[0])
        topk_indices.append(indices[0])

    return topk_indices, topk_values

def compute(gt_embeds_word, pred_embeds_word, gt_types, pred_types, weight_matrix):
    neg_class = [('NON-DISEASE', 'DISEASE'), 
                 ('NON-ABNORMALITY', 'ABNORMALITY'), 
                 ('DISEASE', 'NON-DISEASE'), 
                ('ABNORMALITY', 'NON-ABNORMALITY'),
                ('NON-DISEASE', 'ABNORMALITY'),
                ('NON-ABNORMALITY', 'DISEASE'),
                ('DISEASE', 'NON-ABNORMALITY'),
                ('ABNORMALITY', 'NON-DISEASE'),]
    neg_weight = weight_matrix[("NEG", "WEIGHT")]
    topk_indices, topk_values = topk_similarity(gt_embeds_word, pred_embeds_word, k=1)   

    
    for i in range(len(topk_indices)):
        topk_indices[i] = topk_indices[i].cpu().numpy().tolist()
        topk_values[i] = topk_values[i].cpu().numpy().tolist()
        
    # map the indices to type
    topk_map = [pred_types[i] for i in topk_indices]
    
    weight_score = [weight_matrix[(gt_type, pred_type)] for gt_type, pred_type in zip(gt_types, topk_map)]
    type_score = [neg_weight if (gt_type, pred_type) in neg_class else 1 for gt_type, pred_type in zip(gt_types, topk_map)]
    
    weighted_avg_score = 0
    weighted_sum = 0
    for score, weight, type in zip(topk_values, weight_score, type_score):
        weighted_avg_score += score*weight*type
        weighted_sum += weight
    if weighted_sum != 0:
        RaTE = weighted_avg_score/weighted_sum
    else:
        RaTE = 0
    
    return RaTE