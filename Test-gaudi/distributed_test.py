import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import habana_frameworks.torch.core as htcore

import habana_dataloader

device = torch.device('hpu')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    #distributed package for HCCL
    import habana_frameworks.torch.distributed.hccl
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.fc1   = nn.Linear(784, 1024)
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512, 10)

    def forward(self, x):

        out = x.view(-1,28*28)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

def train_(net,criterion,optimizer,trainloader,device):

    net.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(trainloader):

        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, targets)

        loss.backward()
        
        # API call to trigger execution
        htcore.mark_step()
        
        optimizer.step()

        # API call to trigger execution
        htcore.mark_step()
        

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = train_loss/(batch_idx+1)
    train_acc = 100.0*(correct/total)
    print("Training loss is {} and training accuracy is {}".format(train_loss,train_acc))



def train(rank,world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    epochs = 200
    batch_size = 512
    lr = 0.01
    milestones = [10,15]
    load_path = './data'
    save_path = './checkpoints'
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root=load_path, train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                            sampler=DistributedSampler(trainset, shuffle=True), pin_memory=True)
    testset = torchvision.datasets.MNIST(root=load_path, train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                            sampler=DistributedSampler(testset, shuffle=True), pin_memory=True)

    
    model = SimpleModel().to(device)
    ddp_model = DDP(model)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, epochs+1):
        print("=====================================================================")
        print("Epoch : {}".format(epoch))
        train_(ddp_model,criterion,optimizer,trainloader,device)
        
    cleanup()

def run(code, world_size):
    mp.spawn(code,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    world_size = 8
    # os.environ["MAX_WAIT_ATTEMPTS"] = "50"
    # os.environ['HCL_CPU_AFFINITY'] = '1'
    run(train, world_size)
            