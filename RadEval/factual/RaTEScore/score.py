import torch
import medspacy
nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_context"])

from .utils import sentence_split, post_process

def run_ner(texts, idx2label, tokenizer, model, device, batch_size):
    
    clean_text_list, is_start_list = sentence_split(texts)
    
    predicted_labels = []

    for i in range(0, len(clean_text_list), batch_size):
        batch_text = clean_text_list[i:i+batch_size]

        inputs = tokenizer(batch_text, 
                        max_length=512,
                        padding=True, 
                        truncation=True, 
                        return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_labels.extend(torch.argmax(outputs.logits, dim=2).tolist())

    inputs = tokenizer(clean_text_list, 
                        max_length=512,
                        padding=True, 
                        truncation=True, 
                        return_tensors="pt")
    
    save_pairs = []
    
    pad_token_id = tokenizer.pad_token_id

    for i, is_start in enumerate(is_start_list):

        predicted_entities = [idx2label[label] for label in predicted_labels[i]]

        non_pad_mask = inputs["input_ids"][i] != pad_token_id
        non_pad_length = non_pad_mask.sum().item()
        non_pad_input_ids = inputs["input_ids"][i][:non_pad_length]
        
        tokenized_text = tokenizer.convert_ids_to_tokens(non_pad_input_ids)

        if is_start:
            save_pair = post_process(tokenized_text, predicted_entities, tokenizer)
        else:
            save_pair = post_process(tokenized_text, predicted_entities, tokenizer)
            save_pairs[-1].extend(save_pair)
            continue
        
        save_pairs.append(save_pair)

    return save_pairs


def process_embedding(pair, eval_tokenizer, eval_model, device):
    entities = [pair[0] for pair in pair]
    types = [pair[1] for pair in pair]
    
    if len(entities) == 0:
        embeds_word = torch.tensor([])
    else:
        embeds_word = torch.tensor([]).to(device)
        
        with torch.no_grad():
            # tokenize the queries   
            encoded = eval_tokenizer(
                entities, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=30,
            ).to(device)
            
            # encode the queries (use the [CLS] last hidden states as the representations)
            embeds_word = torch.cat((embeds_word.to('cpu'),
                                    eval_model(**encoded).last_hidden_state[:, 0, :].to('cpu')), dim=0)
    
    return embeds_word, types

