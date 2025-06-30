import io

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile

# step 1: Setup constant
model_name = "StanfordAIMI/CheXagent-2-3b"
dtype = torch.bfloat16
device = "cuda"


# step 3: Download image from URL, save to a local file, and prepare path list
url = "https://huggingface.co/IAMJB/interpret-cxr-impression-baseline/resolve/main/effusions-bibasal.jpg"
resp = requests.get(url)
resp.raise_for_status()

# Use a NamedTemporaryFile so it lives on disk
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
    tmpfile.write(resp.content)
    local_path = tmpfile.name  # this is a real file path on disk

paths = [local_path]

# step 2: Load Processor and Model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
model = model.to(dtype)
model.eval()

# step 3: Inference
query = tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': "Generate the Findings section"}])
conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
output = model.generate(
    input_ids.to(device), do_sample=False, num_beams=1, temperature=1., top_p=1., use_cache=True,
    max_new_tokens=512
)[0]
response = tokenizer.decode(output[input_ids.size(1):-1])
print(response)