# Clip-retrieval Exploration

Here we will reproduce the clip-retrieval's official repo documentation, further exploring to our liking.

## Installation

In order to install clip-retrieval, first setup a CUDA-ready pytorch virtual environment. Here are the steps to do so with `conda`.

```bash
    conda update conda
    conda create -n clipenv
    conda activate clipenv
    (clipenv) conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
    (clipenv) pip install clip-retrieval notebook img2dataset
```

## Usage

Simply activate your environment and set it as the jupyter notebook kernel. Then, open the notebooks on your frontend of choice.
