# ddp_comparison_train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import time
import datetime

def setup_ddp(local_rank, global_rank, world_size, master_addr, master_port, backend):
    """Initializes the distributed environment for a single process."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    print(f"Rank {global_rank} (Local {local_rank}): Initializing process group with backend '{backend}'. Master: {master_addr}:{master_port}, WorldSize: {world_size}")
    dist.init_process_group(backend, rank=global_rank, world_size=world_size)
    torch.cuda.set_device(local_rank) 
    print(f"Rank {global_rank} (Local {local_rank}): Process group initialized. Device: cuda:{local_rank}")

def cleanup_ddp():
    dist.destroy_process_group()

def get_imagenet100_loader(data_root, batch_size_per_gpu, world_size, global_rank, num_workers=4):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset_path = os.path.join(data_root, 'train')
    if not os.path.isdir(train_dataset_path):
        print(f"Warning (Rank {global_rank}): Training data path '{train_dataset_path}' not found. Please check --imagenet-root.")
        # Attempting to use data_root directly if train/ subdirectory is missing.
        # This might be the case if data_root already points to the directory containing class folders.
        if os.path.isdir(data_root) and any(os.path.isdir(os.path.join(data_root, d)) for d in os.listdir(data_root)):
             print(f"Warning (Rank {global_rank}): Using '{data_root}' as the training dataset path.")
             train_dataset_path = data_root
        else:
            raise FileNotFoundError(f"Training data not found at '{os.path.join(data_root, 'train')}' or '{data_root}'. Please verify --imagenet-root.")


    train_dataset = ImageFolder(root=train_dataset_path, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader

def train_worker(local_rank, node_rank, num_gpus_per_node, world_size, args):
    global_rank = node_rank * num_gpus_per_node + local_rank
    
    setup_ddp(local_rank, global_rank, world_size, args.master_addr, args.master_port, args.backend)

    if global_rank == 0: print(f"Creating {args.model_name} model...")
    model = getattr(torchvision.models, args.model_name)(num_classes=100) # ImageNet100
    model = model.to(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    if global_rank == 0: print(f"Model created and wrapped with DDP.")

    criterion = nn.CrossEntropyLoss().to(local_rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr * world_size, momentum=0.9)

    if global_rank == 0: print(f"Preparing DataLoader on rank 0...")
    train_loader = get_imagenet100_loader(args.imagenet_root, args.batch_size, world_size, global_rank, num_workers=args.num_workers)
    if global_rank == 0: print(f"DataLoader ready. Batches per epoch: {len(train_loader)}")

    model.train()
    
    if global_rank == 0:
        print(f"\nStarting DDP training for {args.num_epochs} epochs to measure throughput...")
        print(f"World Size: {world_size}, Batch Size per GPU: {args.batch_size}, Total Effective Batch Size: {args.batch_size * world_size}")

    total_images_processed_after_warmup = 0
    total_time_train_batches_after_warmup = 0.0
    warmup_iterations = args.warmup_iters
    
    overall_start_time = time.time()
    current_iteration = 0

    for epoch in range(args.num_epochs):
        train_loader.sampler.set_epoch(epoch) # Important for shuffling with DistributedSampler
        
        # tqdm progress bar only on global rank 0 process for the first local rank
        if global_rank == 0 and local_rank == 0:
            pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.num_epochs} [GPU{local_rank}]")
        else:
            pbar_train = enumerate(train_loader)

        for i, (images, labels) in pbar_train:
            iter_start_time = time.time()

            images = images.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize(device=local_rank) # Sync for timing
            iter_end_time = time.time()

            if current_iteration >= warmup_iterations:
                total_images_processed_after_warmup += images.size(0) # Count images processed by this GPU
                total_time_train_batches_after_warmup += (iter_end_time - iter_start_time)
            
            current_iteration += 1

            if global_rank == 0 and local_rank == 0 and (i + 1) % args.log_interval == 0:
                current_batch_time = iter_end_time - iter_start_time
                # Throughput for this batch on this GPU, then multiply by world_size for total
                current_iter_speed_total = (images.size(0) * world_size) / current_batch_time if current_batch_time > 0 else 0
                pbar_train.set_postfix({'loss': loss.item(), 'iter_img/s_total': f'{current_iter_speed_total:.2f}'})
        
        if global_rank == 0 and local_rank == 0:
            print(f"Epoch {epoch+1} finished on GPU {local_rank}.")

    # Aggregate timing information from all ranks for accuracy
    # Each rank processed 'total_images_processed_after_warmup' on its own batches
    # And spent 'total_time_train_batches_after_warmup' on those.
    # We need to sum images and take max time (or avg time - max is more conservative for throughput)
    
    images_tensor = torch.tensor([float(total_images_processed_after_warmup)], device=local_rank)
    time_tensor = torch.tensor([float(total_time_train_batches_after_warmup)], device=local_rank)

    if world_size > 1:
        dist.all_reduce(images_tensor, op=dist.ReduceOp.SUM) # Sum of all images processed by all GPUs
        dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)   # Max time taken by any GPU for its batches (bottleneck)
                                                            # Or use dist.ReduceOp.SUM and divide by world_size for average batch time

    total_images_all_gpus = images_tensor.item()
    max_time_across_gpus = time_tensor.item() # If using MAX
    # avg_time_across_gpus = time_tensor.item() / world_size # If using SUM for time_tensor

    if global_rank == 0 and local_rank == 0: # Only print summary once
        overall_script_time = time.time() - overall_start_time
        print(f"\n--- DDP Throughput Summary (Aggregated) ---")
        if max_time_across_gpus > 0 and total_images_all_gpus > 0: # Ensure we have valid numbers
            # Effective throughput is total images processed by all GPUs divided by the time the slowest path took
            avg_throughput_total = total_images_all_gpus / max_time_across_gpus
            print(f"Average Iteration Speed (Total, Post-Warmup): {avg_throughput_total:.2f} images/sec")
            print(f"Average Iteration Speed per GPU (Post-Warmup): {avg_throughput_total / world_size:.2f} images/sec")
        else:
            print("Not enough iterations post-warmup or no images processed to calculate average throughput.")
        print(f"Total images summed across GPUs (post-warmup): {total_images_all_gpus}")
        print(f"Max batch processing time on any rank (post-warmup): {max_time_across_gpus:.2f} seconds")
        print(f"Overall script run time: {str(datetime.timedelta(seconds=int(overall_script_time)))}")

    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDP ImageNet-100 Training for Comparison')
    parser.add_argument('--master-addr', type=str, required=True, help='IP address of the DDP master node')
    parser.add_argument('--master-port', type=int, default=29500, help='Port for DDP store communication')
    parser.add_argument('--node-rank', type=int, required=True, help='Rank of this node (0, 1, ...)')
    parser.add_argument('--num-nodes', type=int, required=True, help='Total number of nodes')
    parser.add_argument('--num-gpus-per-node', type=int, default=1, help='Number of GPUs to use on this node')

    parser.add_argument('--model-name', type=str, default='resnet50', help='Model name (e.g., resnet50, resnet152)')
    parser.add_argument('--imagenet-root', type=str, default='/home/data', help='Path to ImageNet-100 dataset (parent of train/)')
    parser.add_argument('--batch-size', type=int, default=64, help="Per-GPU batch size")
    parser.add_argument('--num-epochs', type=int, default=3, help='Number of epochs for throughput test (after warmup)')
    parser.add_argument('--lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval for tqdm')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'], help='DDP backend')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers per GPU process')
    parser.add_argument('--warmup-iters', type=int, default=20, help="Number of initial iterations to skip for throughput calculation")
    
    args = parser.parse_args()

    if args.num_gpus_per_node > torch.cuda.device_count():
        print(f"Warning: Requested {args.num_gpus_per_node} GPUs, but only {torch.cuda.device_count()} available. Using available count.")
        args.num_gpus_per_node = torch.cuda.device_count()
    
    if args.num_gpus_per_node == 0:
        print("Error: No GPUs specified or available for this node. Exiting.")
        exit(1)

    world_size = args.num_nodes * args.num_gpus_per_node

    print(f"--- DDP Setup on Node Rank {args.node_rank} ---")
    print(f"Master Address: {args.master_addr}:{args.master_port}")
    print(f"World Size: {world_size} ({args.num_nodes} nodes x {args.num_gpus_per_node} GPUs/node)")
    print(f"Number of GPUs on this node: {args.num_gpus_per_node}")

    mp.spawn(train_worker,
             args=(args.node_rank, args.num_gpus_per_node, world_size, args),
             nprocs=args.num_gpus_per_node, # Launch one process per GPU on this node
             join=True)