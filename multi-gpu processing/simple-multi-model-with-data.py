#!/usr/bin/env python
# coding: utf-8

# This is a simple PyTorch demo illustraing training on FashionMNIS dataset and the use of DataLoader.

# ### Loading Data

# In[25]:


import torch
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.multiprocessing as mp # this library is used to spawn multiple processes for distributed training, allowing us to run the training loop on multiple GPUs in parallel. Each process will handle a portion of the training data and update the model parameters independently.
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP # this is a wrapper that will help us parallelize our model across multiple GPUs
from torch.distributed import init_process_group, destroy_process_group
import os


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    start_time = time.time()
    os.environ["MASTER_ADDR"] = "localhost" # Sets the IP address of the master node (the node that coordinates the training). This is most useful when training across multiple machines, but we set it to localhost for single machine multi-GPU training.
    os.environ["MASTER_PORT"] = "12355" # Sets the port on the master node for communication. 
    torch.cuda.set_device(rank) # Sets the current GPU device for this process based on its rank. Each process will be assigned a different GPU.
    init_process_group(backend="nccl", rank=rank, world_size=world_size) # this line allows communication between the GPU hive. The "nccl" backend is NVIDIA's Collective Communications Library, a communication protocol for NVIDIA GPUs.
    print(f"Time taken to setup DDP for rank {rank}: {time.time() - start_time:.2f} seconds")

def prepare_dataloader(dataset: Dataset, batch_size: int):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset) # The distributed sampler allows us to partition the dataset across multiple processes, ensuring that each process gets a different subset of the data for training. This is crucial for distributed training to ensure that we are not training on the same data in each process, which would lead to inefficient training and poor model performance.
        # If we don't specify a sampler, the DataLoader uses RandomSampler by default.
    )

    for X, y in dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return dataloader

class Trainer:
    def __init__(
        self,
        model: NeuralNetwork,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[gpu_id]) # Wraps the model for distributed training , allowing it to be trainied across GPUs and synchronizes the gradients during backpropagation. The device_ids argument specifies which GPU to use for this process.

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss


    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch) # This allows DataLoader know it is now in a new epoch and should shuffle the data differently for each epoch. 
        for batch_idx, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            if batch_idx % 100 == 0:
                loss, current = loss.item(), (batch_idx + 1) * len(source)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(self.train_data.dataset):>5d}]")

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            print(f"Starting Epoch {epoch} on GPU{self.gpu_id}")
            start_time = time.time()
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            print(f"GPU{self.gpu_id}: Time taken for epoch {epoch}: {time.time() - start_time:.2f} seconds")

def test(dataloader, model, gpu_id):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(gpu_id), y.to(gpu_id)
            pred = model(X)
            test_loss += nn.CrossEntropyLoss()(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss = test_loss / num_batches
    correct = correct / size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def load_train_objs():
    # train_set = training_data  # load your dataset
    model = NeuralNetwork()  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, optimizer


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    model, optimizer = load_train_objs()
    start_time = time.time()
    train_set= datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    train_data = prepare_dataloader(train_set, batch_size)
    print(f"Time taken to prepare dataloader: {time.time() - start_time:.2f} seconds")
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    start_time = time.time()
    trainer.train(total_epochs)
    print(f"GPU{rank}: Total Training time: {time.time() - start_time:.2f} seconds")
    torch.distributed.barrier()
    if rank == 0:
        test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        start_time = time.time()
        test(test_dataloader, trainer.model, rank)
        print(f"GPU{rank}: Time taken to test the model: {time.time() - start_time:.2f} seconds")
        torch.save(trainer.model.module.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")
    destroy_process_group()





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs',  default=10, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    total_epochs = args.total_epochs
    save_every = args.save_every 
    batch_size = args.batch_size 
    world_size = torch.cuda.device_count() # Get the number of available GPUs. This will determine how many processes we need to spawn for distributed training. As opposed to setting device = 0 for single GPU training.
    start_time = time.time()
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size) # here we are spawning multiple processes (one for each GPU) and running the main function in each process using world_size to determine how many processes to spawn. Each process will have a unique rank (from 0 to world_size-1) that is passed to the main function. This allows each process to know which GPU it should use.
    print(f"Total time taken for training across all GPUs: {time.time() - start_time:.2f} seconds")

    # Inference after training
    classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
    ]

    device = 0 if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
    x, y = test_data[0][0], test_data[0][1]
    print(y)
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')





