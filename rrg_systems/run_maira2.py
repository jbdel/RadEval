from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch

model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)

device = torch.device("cuda")
model = model.eval()
model = model.to(device)
from PIL import Image


findings_file= open("/fss/jb/run_rrg_models/data/mimic-cxr/findings/test.findings.tok").readlines()

lines = []
for line in findings_file:
    line = line.strip()
    if line:
        lines.append(line)

images_file = open("/fss/jb/run_rrg_models/data/mimic-cxr/findings/test.image.tok").readlines()


images= []
for line in images_file:
    line = line.strip()
    if line:
        images.append(line)

assert len(lines) == len(images)

for i in range(10):
    curr_images = (images[i].split(","))
    frontal_image_path = "images/" + curr_images[0]
    frontal_image = Image.open(frontal_image_path)
    if len(curr_images) > 1:
        lateral_image_path = "images/" + curr_images[1]
        lateral_image = Image.open(lateral_image_path)

    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=frontal_image,
        current_lateral=lateral_image if len(curr_images) > 1 else None,
        prior_frontal=None,  # Our example has no prior
        indication=None,
        technique=None,
        comparison=None,
        prior_report=None,  # Our example has no prior
        return_tensors="pt",
        get_grounding=False,  # For this example we generate a non-grounded report
    )

    processed_inputs = processed_inputs.to(device)
    with torch.no_grad():
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=300,  # Set to 450 for grounded reporting
            use_cache=True,
        )
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
    decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    print("Parsed prediction:", prediction)
    # write to csv both the prediction and the ground truth
    # use csv library to write to csv
    import csv
    with open("maira2.csv", "a") as f:
   
        writer = csv.writer(f)
        if i == 0:
            writer.writerow(["type","ground_truth", "prediction"])
        writer.writerow(["findings", lines[i], prediction])
    