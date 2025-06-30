from libra.eval import libra_eval

# Path to your current chest X-ray image
image_path = "images/mimic-cxr-images-512/files/p10/p10046166/s50051329/abea5eb9-b7c32823-3a14c5ca-77868030-69c83139.jpg"

# Run inference to get the Impression section
impression = libra_eval(
    model_path="X-iZhang/Med-CXRGen-I",      # impression-section model
    image_file=[image_path],                 # single image
    query="Generate the Impression section of the radiology report.",
    conv_mode="libra_v0",                    # v0 conv_mode for Med-CXRGen-I
    max_new_tokens=128,                      # adjust length as needed
    temperature=0.1,                         # low temp for deterministic output
    top_p=0.9
)

print("=== Impression ===")
print(impression)
