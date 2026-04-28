from pathlib import Path
from setuptools import setup, find_packages

setup(
    name='RadEval',
    version='2.1.0',
    author='Jean-Benoit Delbrouck, Justin Xu, Xi Zhang, Dave Van Veen',
    maintainer='Xi Zhang, JB Delbrouck',
    url='https://github.com/jbdel/RadEval',
    project_urls={
        'Bug Reports': 'https://github.com/jbdel/RadEval/issues',
        'Source': 'https://github.com/jbdel/RadEval',
        'Documentation': 'https://github.com/jbdel/RadEval/blob/main/README.md',
    },
    license='MIT',
    description='All-in-one metrics for evaluating AI-generated radiology text',
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords=[
        'radiology',
        'evaluation',
        'natural language processing',
        'radiology report',
        'medical NLP',
        'clinical text generation',
        'LLM',
        'bioNLP',
        'chexbert',
        'radgraph',
        'medical AI',
        'reinforcement learning',
        'reward model',
    ],
    python_requires='>=3.11',
    install_requires=[
        'torch>=2.3',
        'torchvision',
        'transformers>=5.0',
        # Vendored BERTScore (RadEval/metrics/bertscore/_vendor/) replaces
        # the `bert-score` PyPI package; no dependency required.
        # Vendored RadGraph inference (RadEval/metrics/radgraph/_vendor/) —
        # a patched subset of Stanford-AIMI/radgraph 0.1.18 that runs on
        # transformers 5.x. See RadEval/metrics/radgraph/_vendor/__init__.py
        # for details. No external `radgraph` PyPI dependency is required.
        'rouge_score',
        'scikit-learn>=1.8.0',
        'numpy<3',
        'medspacy',
        'stanza',
        'pillow',
        'sentencepiece',
        'datasets>=2.19',
        'accelerate>=1.1',
        'pandas',
        'rich',
        'pyyaml',
        'appdirs',
        'filelock',
        'dotmap',
        'h5py',
        'jsonpickle',
        'nltk',
        # Required by the GREEN-radllama2-7b remote modeling file, which
        # `import matplotlib.pyplot` at module load (via the HF Hub
        # `trust_remote_code=True` path used by the GREEN metric).
        'matplotlib',
        'huggingface_hub>=1.0',
    ],
    extras_require={
        'api': [
            'google-genai',
            'openai',
            'tenacity',
        ],
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "RadEval.metrics.SRRBert": ["*.json"],
        "RadEval.metrics.bertscore._vendor": [
            "rescale_baseline/*/*.tsv",
        ],
    },
    zip_safe=False,
)
