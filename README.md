# deep-learning

## local machine setup

Setting up pythong environments is painful. Here's what I've got 

- Install Miniconda to help manage pythong environments 
- Run `conda create --name virtualEnv1 python==3.11`
- `conda activate virtualEnv1`

At this point conda sends you into your python virtual environment where you can begin to install python dependedncies using pip

E.g 
`pip install torch`

Launch a conda/ jupyter notebook

```
import torch

if torch.cuda.is_available():
    print("GPU is available.")
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("GPU is not available; using CPU.")

```
### Optionally
- install cuda if you have a GPU
`conda install cuda -c nvdia/label/cuda-12.8.0`

## Running workloads remotely 

I'll be using GPUs in [google collab](https://colab.research.google.com). Onnce signed in you can go to runtime and switch CPU to GPU of your choice. I will be trying out the NVDIA blackwell H100. 

Run the same script as before
```
import torch

if torch.cuda.is_available():
    print("GPU is available.")
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("GPU is not available; using CPU.")
```

```
GPU is available.
GPU device name: NVIDIA A100-SXM4-80GB
Number of GPUs: 1
```

We will be trying out multi-threaded GPUs eventually, but 1 suffices for now.