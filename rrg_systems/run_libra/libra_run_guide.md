# Libra Chest X-ray Report Generation Tool

## Environment Setup

```bash
git clone https://github.com/X-iZhang/Libra.git
cd Libra
```

```bash
conda create -n libra python=3.10 -y
conda activate libra
```
Or use environment.yaml
```bash
conda env create -f environment.yaml
conda activate libra
```

## Running Model Examples

```bash
python run_libra_v1_7b.py 
```

## Path Configuration
```python
# Define the output directory for the model predictions
output_file = "../RadEval/rrg_systems/run_libra/answers/libra.v1.7b.mimic.test.findings.tok"

# Load the ground truth findings file (ensure the same number of studies)
findings_file= open("/fss/jb/run_rrg_models/data/mimic-cxr/findings/test.findings.tok").readlines()

# Load the image dataset
images_file = open("/fss/jb/run_rrg_models/data/mimic-cxr/findings/test.image.tok").readlines()
```

🪧:

The same procedure applies for `run_libra_v1_3b.py`, `run_libra_v0_F.py`, and `run_libra_v0_I.py`

⚠️:

`run_libra_v0_I.py` can only generate `Impression` sections.

`run_libra_v1_7b.py`, `run_libra_v1_3b.py`, and `run_libra_v0_F.py` can only generate `Findings` sections.