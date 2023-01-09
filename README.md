# PromptEHR
[![PyPI version](https://badge.fury.io/py/transtab.svg)](https://badge.fury.io/py/promptehr)
[![Downloads](https://pepy.tech/badge/promptehr)](https://pepy.tech/project/promptehr)
![GitHub Repo stars](https://img.shields.io/github/stars/ryanwangzf/promptehr)
![GitHub Repo forks](https://img.shields.io/github/forks/ryanwangzf/promptehr)

Wang, Zifeng and Sun, Jimeng. (2022). PromptEHR: Conditional Electronic Healthcare Records Generation with Prompt Learning. EMNLP'22.

# News
- [2023/01/08] `PromptEHR` is now integrated into [`PyTrial`](https://github.com/RyanWangZf/PyTrial) with a complete [documentation](https://pytrial.readthedocs.io/en/latest/trial_simulation/sequence/promptehr.html) and [example](https://colab.research.google.com/drive/1EbzLdSwTrbgsEgz8z70qzTLQWiPWlyRm?usp=sharing), please check! New version with bugs fixed is also released!


# Usage

Get pretrained PromptEHR model (learned on MIMIC-III sequence EHRs) in three lines:

```python
from promptehr import PromptEHR

model = PromptEHR()

model.from_pretrained()
```

A jupyter example is available at https://github.com/RyanWangZf/PromptEHR/blob/main/example/demo_promptehr.ipynb.



# How to install

Install the correct `PyTorch` version by referring to https://pytorch.org/get-started/locally/.

Then try to install `PromptEHR` by

```bash
pip install git+https://github.com/RyanWangZf/PromptEHR.git
```

or

```bash
pip install promptehr
```



# Load demo synthetic EHRs (generated by PromptEHR)

```python
from promptehr import load_synthetic_data
data = load_synthetic_data()
```



# Use PromptEHR for generation

```python
from promptehr import SequencePatient
from promptehr import load_synthetic_data
from promptehr import PromptEHR

# init model
model = PromptEHR()
model.from_pretrained()

# load input data
demo = load_synthetic_data(n_sample=1000) # we have 10,000 samples in total

# build the standard input data for train or test PromptEHR models
seqdata = SequencePatient(data={'v':demo['visit'], 'y':demo['y'], 'x':demo['feature'],},
    metadata={
        'visit':{'mode':'dense'},
        'label':{'mode':'tensor'}, 
        'voc':demo['voc'],
        'max_visit':20,
        }
    )
# you can try to fit on this data by
# model.fit(seqdata)

# start generate
# n: the target total number of samples to generate
# n_per_sample: based on each sample, how many fake samples will be generated
# the output will have the same format of `SequencePatient`
fake_data = model.predict(seqdata, n=1000, n_per_sample=10)
```

