import torch
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor, VisionEncoderDecoderModel, GenerationConfig
import requests

mode = "impression"
# Model
model = VisionEncoderDecoderModel.from_pretrained(f"IAMJB/chexpert-mimic-cxr-{mode}-baseline").eval()
tokenizer = BertTokenizer.from_pretrained(f"IAMJB/chexpert-mimic-cxr-{mode}-baseline")
image_processor = ViTImageProcessor.from_pretrained(f"IAMJB/chexpert-mimic-cxr-{mode}-baseline")
#
# Dataset
generation_args = {
   "bos_token_id": model.config.bos_token_id,
   "eos_token_id": model.config.eos_token_id,
   "pad_token_id": model.config.pad_token_id,
   "num_return_sequences": 1,
   "max_length": 128,
   "use_cache": True,
   "beam_width": 2,
}

model.to("cuda")
#
# Inference


findings_file= open("/fss/jb/run_rrg_models/data/mimic-cxr/impression/test.impression.tok").readlines()

lines = []
for line in findings_file:
    line = line.strip()
    if line:
        lines.append(line)

images_file = open("/fss/jb/run_rrg_models/data/mimic-cxr/impression/test.image.tok").readlines()


images= []
for line in images_file:
    line = line.strip()
    if line:
        images.append(line)

assert len(lines) == len(images)

for i in range(20):
    with torch.no_grad():
        curr_images = (images[i].split(","))
        frontal_image_path = "images/" + curr_images[0]
        frontal_image = Image.open(frontal_image_path)
        pixel_values = image_processor(frontal_image, return_tensors="pt").pixel_values
        # Generate predictions
        generated_ids = model.generate(
            pixel_values.to("cuda"),
            generation_config=GenerationConfig(
                **{**generation_args, "decoder_start_token_id": tokenizer.cls_token_id})
        )
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(generated_texts)

        import csv
        with open("chexplus.csv", "a") as f:
    
            writer = csv.writer(f)
            if i == 0:
                writer.writerow(["type","ground_truth", "prediction"])
            writer.writerow(["impression", lines[i], generated_texts[0]])
        