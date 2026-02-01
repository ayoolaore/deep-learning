# deep-learning

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