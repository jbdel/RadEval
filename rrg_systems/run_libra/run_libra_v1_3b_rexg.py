# --- Silence Hugging Face warnings ---
from transformers.utils import logging
logging.set_verbosity_error()

# --- Import necessary libraries ---
from libra.eval.run_libra import load_model
from libra.eval import libra_eval
from torch import cuda
import os
import json
import pandas as pd
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id")

# --- Set up the model ---
# model_path = "X-iZhang/libra-v1.0-7b"
model_path = "X-iZhang/libra-v1.0-3b"
# model_path = "X-iZhang/Med-CXRGen-F"
# model_path = "X-iZhang/Med-CXRGen-I"


reuse_model = load_model(model_path)
print('Load model success')

warning_message = (
    "WARNING: The libra family of models is only trained on frontal chest X-ray images."
)
warnings.warn(warning_message)

# --- Choose prompt based on model type ---
if "Med-CXRGen-I" in model_path:
    prompt = "Provide a detailed description of the impression in the radiology image."
else:
    prompt = "Provide a detailed description of the findings in the radiology image."

# --- Define inference parameters ---
num_beams = 5
length_penalty = 2.0
max_new_tokens = 256
num_return_sequences = 3

# Define the output directory for the model predictions
output_file = "./RadEval/rrg_systems/run_libra/answers/libra.v1.3b.chex.test.findings.tok"

# load the ground truth findings file
findings_file= open("/mnt/primary/RadEval/rrg_systems/mimic-cxr/test.findings.tok").readlines()

lines = []
for line in findings_file:
    line = line.strip()
    if line:
        lines.append(line)

# Load the image dataset
images_file = open("/mnt/primary/RadEval/rrg_systems/mimic-cxr/test.findings.image.libra.tok").readlines()

images= []
for line in images_file:
    line = line.strip()
    if line:
        images.append(line)

# Make sure the number of images matches the number of lines in the findings file
assert len(lines) == len(images)

with open(output_file, "w") as f:
    for i in tqdm(range(len(images)), desc=f"Generating with {model_path}"):
        curr_images = images[i].split(",")
        frontal_image_path = curr_images[0]  # Libra only uses single frontal image for inference

        # Run inference
        findings = libra_eval(
            libra_model=reuse_model,
            image_file=[frontal_image_path,frontal_image_path],         # Dummy previous image for libra models
            query=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            top_p=None,
            temperature=None
        )
        # Clean up the findings text for the \n character
        findings = findings.replace("\n", "\\n").strip()
        f.write(findings + "\n")

# --- Check the output file ---
with open(output_file, "r") as f_check:
    output_lines = f_check.readlines()

assert len(output_lines) == len(lines), \
    f"Mismatch: wrote {len(output_lines)} lines, but expected {len(lines)}"

print(f"Saved {len(output_lines)} predictions to {output_file}")