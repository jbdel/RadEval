from setuptools import setup, find_packages

setup(
    name='RadEval',
    version='0.0.1',
    author='Jean-Benoit Delbrouck',
    license='MIT',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    install_requires=[
        'torch==2.3',
        'transformers==4.39.0',
        'radgraph',
        'rouge_score',
        'bert-score==0.3.13',
        'scikit-learn',
        'numpy<2',
    ],
    packages=find_packages(),
    zip_safe=False)
