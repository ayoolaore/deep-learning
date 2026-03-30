import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp # this library is used to spawn multiple processes for distributed training, allowing us to run the training loop on multiple GPUs in parallel. Each process will handle a portion of the training data and update the model parameters independently.
from torch.utils.data.distributed import DistributedSampler # This is a sampler that restricts data loading to a subset of the dataset for distributed training. It ensures that each process gets a different subset of the data, which is crucial for efficient training across multiple GPUs.
from torch.nn.parallel import DistributedDataParallel as DDP # this is a wrapper that will help us parallelize our model across multiple GPUs
from torch.distributed import init_process_group, destroy_process_group # init_process_group is used to initialize the default distributed process group, which is necessary for distributed training. It sets up the communication backend and initializes the process group based on the specified parameters. destroy_process_group is used to clean up the process group after training is complete.
import os


# ddp_setup is a helper function that initializes the process group for distributed training. It sets the device for each process and initializes the process group using the NCCL backend, which is optimized for NVIDIA GPUs.
# Once the process is distributed across GPUs, this is the first function that will be called in each process to set up the distributed environment. It ensures that each process is aware of its rank and the total number of processes, which is crucial for coordinating the training across multiple GPUs.
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost" # Sets the IP address of the master node (the node that coordinates the training). This is most useful when training across multiple machines, but we set it to localhost for single machine multi-GPU training.
    os.environ["MASTER_PORT"] = "12355" # Sets the port on the master node for communication. 
    torch.cuda.set_device(rank) # Sets the current GPU device for this process based on its rank. Each process will be assigned a different GPU.
    init_process_group(backend="nccl", rank=rank, world_size=world_size) # this line allows communication between the GPU hive. The "nccl" backend is NVIDIA's Collective Communications Library, a communication protocol for NVIDIA GPUs.
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id]) # Wraps the model for distributed training , allowing it to be trainied across GPUs and synchronizes the gradients during backpropagation. The device_ids argument specifies which GPU to use for this process.

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch) # This allows DataLoader know it is now in a new epoch and should shuffle the data differently for each epoch. 
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            print(f"Starting Epoch {epoch} on GPU{self.gpu_id}")
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset) # The distributed sampler allows us to partition the dataset across multiple processes, ensuring that each process gets a different subset of the data for training. This is crucial for distributed training to ensure that we are not training on the same data in each process, which would lead to inefficient training and poor model performance.
        # If we don't specify a sampler, the DataLoader uses RandomSampler by default.
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group() # this line is used to clean up the process group after training is complete. It ensures that all processes are properly terminated and resources are released.


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count() # Get the number of available GPUs. This will determine how many processes we need to spawn for distributed training. As opposed to setting device = 0 for single GPU training.
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size) # here we are spawning multiple processes (one for each GPU) and running the main function in each process using world_size to determine how many processes to spawn. Each process will have a unique rank (from 0 to world_size-1) that is passed to the main function. This allows each process to know which GPU it should use.
