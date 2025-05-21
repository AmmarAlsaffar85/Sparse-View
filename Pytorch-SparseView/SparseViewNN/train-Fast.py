#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train-Fast.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/training_porschestators_DownFactor10.hdf5 --model Swin2SR_1block
#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train-Fast.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/training_porschestators_DownFactor10.hdf5 --model UNet
#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/training_porschestators.hdf5 --model UNet_Org
#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/training_porschestators.hdf5 --model ODConvSR-9-96
#python3 /zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-Swin/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/train.py --trainingdata /lhome/alsaffar/NewProjects/SparseViewInterpolation/training_porschestators.hdf5 --model ResNet

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import h5py
import time
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils import seeding, create_dir, epoch_time
from sklearn.model_selection import train_test_split
from data import DriveDataset
#from models.load_models import load_model
import matplotlib.pyplot as plt
import random
from models.load_models import ModelLoader
# Assume DriveDataset and other necessary functions (seeding, create_dir, epoch_time, load_model) are defined

def train(model, loader, optimizer, device, rank):
    model.train()
    total_loss = torch.tensor(0.0).to(device)
    total_samples = torch.tensor(0).to(device)

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        batch_size = x.size(0)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)  # Using PyTorch's MSE loss
        loss.backward()
        optimizer.step()

        #For example, if you have two batches: batch1 has 64 samples, loss 2.0 (average), batch2 has 64 samples, loss 3.0. Then
        #the average per batch is (2 + 3)/2 = 2.5. But the true average over samples is (64*2 + 64*3)/(128) = (2.5).
        total_loss += loss.detach() * batch_size
        total_samples += batch_size

    #in PyTorch, you can't all_reduce Python scalars. You need to use tensors.
    #So first, I'll modify the `train` and `evaluate` functions to compute the loss correctly over all samples.
    #Once that's done, in the DDP scenario, each process will compute its own total_loss and
    #total_samples. Then, we can all-reduce these two values across all processes to get the
    #global total_loss and total_samples.
    # All-reduce total_loss and total_samples
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    #But wait, `total_loss` and `total_samples` are tensors on the device. After all_reduce, they
    #hold the sum across all processes. Then, dividing gives the global average loss, so all_reduce is like summing all the
    #losses form all the processes
    epoch_loss = (total_loss / total_samples).item()
    return epoch_loss

def evaluate(model, loader, device, rank):
    model.eval()
    total_loss = torch.tensor(0.0).to(device)
    total_samples = torch.tensor(0).to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            batch_size = x.size(0)

            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)

            total_loss += loss.detach() * batch_size
            total_samples += batch_size

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    epoch_loss = (total_loss / total_samples).item()
    return epoch_loss
    
def load_checkpoint(model, optimizer, scheduler, path, rank):
    # Load on all ranks
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # Adjust for multi-node if needed
    checkpoint = torch.load(path, map_location=map_location)
    
    model.module.load_state_dict(checkpoint['model_state_dict'])
    
    
    # optimizer, and scheduler
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore RNG states
    torch.set_rng_state(checkpoint['rng_state']['torch'])
    torch.cuda.set_rng_state(checkpoint['rng_state']['cuda'])
    np.random.set_state(checkpoint['rng_state']['numpy'])
    random.setstate(checkpoint['rng_state']['python'])
    
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    return start_epoch

def main(rank, world_size, args):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    torch.cuda.set_device(rank)
    #**Adjust Device Handling**: Each process should use its own device (e.g., `device =torch.device('cuda', rank)`).
    device = torch.device('cuda', rank)
    
    seeding(42 + rank)  # Ensure different seeds per process

    # Load datasets
    AllData_input = h5py.File(args.trainingdata, 'r')['training'][:,:,:]
    AllData_label = h5py.File(args.trainingdata, 'r')['label'][:,:,:]

    training, testing, training_label, testing_label = train_test_split(
        AllData_input, AllData_label, test_size=0.20, random_state=42
    )

    # Hyperparameters
    #the batch size in the original code is 67. With DDP, each process will process
    #a batch of size 67, so the effective batch size is 67 * world_size. If the user wants the
    #effective batch size to remain 67, they need to adjust the per-process batch size to 67 /
    #world_size, but this may not be an integer. So perhaps the code should take the per-GPU
    #batch size and multiply by the number of GPUs to get the effective batch size. But in the
    #original code, the user set batch_size=67. So in DDP, each process will use batch_size=67,
    #leading to an effective batch size of 67 * world_size. 
    batch_size = 210 #was 170
    num_epochs = 100
    lr = 1e-4
    checkpoint_path = "/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/files/checkpoint_"+args.model+"_CompleteData.pth"

    # Create datasets and samplers
    train_dataset = DriveDataset(training, training_label)
    valid_dataset = DriveDataset(testing, testing_label)

    #But during evaluation, each GPU processes a part of the validation set, and
    #then the loss is averaged across all. However, in the original code, the evaluation is done
    #on the entire dataset.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    #The data loaders need to use `DistributedSampler` to ensure each GPU processes a unique subset of the data
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=2,
        pin_memory=True
    )

    # Model, Optimizer, Scheduler
    loader = ModelLoader()
    model = loader.get_model(args.model).to(device)
    #model = load_model(args.model).to(device)
    #Now, the model. After moving the model to the device, wrap it with DDP
    # **Model Wrapping**: Wrap the model with `torch.nn.parallel.DistributedDataParallel`. This will handle gradient synchronization across GPUs.
    model = DDP(model, device_ids=[rank], output_device=rank)
    #the optimizer should be created after wrapping the model with DDP
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    best_valid_loss = float('inf')
    H = {"train_loss": [], "test_loss": []}
    
    # Load checkpoint if resuming   
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, rank)
        dist.barrier()  # Wait for all ranks to load
        
        
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, device, rank)
        valid_loss = evaluate(model, valid_loader, device, rank)

        scheduler.step(valid_loss)
        
        # **Checkpoint Saving**: Only save the model from the main process (rank 0) to avoid redundancy.
        if rank == 0:
            H["train_loss"].append(train_loss)
            H["test_loss"].append(valid_loss)

            if valid_loss < best_valid_loss:
                print(f"Valid loss improved from {best_valid_loss:.5f} to {valid_loss:.5f}")
                best_valid_loss = valid_loss
                #when saving the checkpoint, we should save `model.module.state_dict()` instead of
                #`model.state_dict()`, because DDP wraps the original model as `model.module`.
                #If the state dictionary contains tensors on different devices (e.g., some on cuda:0 and others on cuda:1), this can cause the error when loading the model.
                #Solution:
                #Ensure all tensors in the state dictionary are on the same device before saving the model. You can do this by moving the model to a specific device before saving
                model = model.to('cuda:0')  # Move the model to cuda:0
                #torch.save(model.module.state_dict(), checkpoint_path)
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'rng_state': {
                                   'torch': torch.get_rng_state(),
                                   'cuda': torch.cuda.get_rng_state(),
                                   'numpy': np.random.get_state(),
                                   'python': random.getstate()
                                   }
                           }, checkpoint_path)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            #Similarly, printing should be done only by rank 0 to avoid cluttering the output.
            #So any print statements (like `data_str = f"Valid loss improved..."`) should be done only by
            #rank 0.
            print(f'Epoch {epoch+1:02}: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}, Val Loss: {valid_loss:.3f}')
    
    dist.destroy_process_group()
    
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig('/zhome/alsaffar/ipvs/ImportantCodes/ImportantCodes/SparseViewInterpolationMethod-MultiModel-Fast/Pytorch-UNet-SparseViewInterpolation/SparseViewNN/lossCurve'+args.model)
    #plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainingdata', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--resume', type=str, default=None,
                    help='Path to the checkpoint file to resume training')
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'              #
    os.environ['MASTER_PORT'] = '8888'
    #os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7" 
    world_size = torch.cuda.device_count()
    
    #the `torch.multiprocessing.spawn` function starts the processes, and each runs the
    #`main` function with their respective `rank`.
    # *Process Spawning**: Use `torch.multiprocessing.spawn` to launch multiple processes, one per GPU.
    # with DDP, each process will run the main function with a different rank.
    # So, the main function should be inside a function that is called via `mp.spawn`.
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    #IMPORTANT
    #`torch.multiprocessing.spawn(fn, args=(),nprocs=1, join=True, daemon=False, ...)`.
    #The `fn` is called as `fn(i, *args)` for each i in 0 to
    #nprocs-1.
    #So when we call `mp.spawn(main, args=(world_size, args), nprocs=world_size)`, each
    #process will call `main(rank, world_size, args)`.
    
    


