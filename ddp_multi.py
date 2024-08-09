import torch
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
import os

def ddp_setup(rank,world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12335'
    init_process_group(backend='nccl',rank=rank,world_size=world_size)

class Trainer:

    def __init__(self,model,train_loader,optimizer,gpu_id,save_every) -> None:
        #self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.loader = train_loader
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.model = DDP(model,device_ids=[self.gpu_id])

    def train_loop(self,max_epochs):

        for epoch in range(max_epochs):

            epoch_loss = 0
            self.model.train()
            for X,y in self.loader:
                X = X.to(self.gpu_id)
                y = y.to(self.gpu_id)

                logits = self.model(X)
                loss = torch.nn.CrossEntropyLoss()(logits,y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss+=loss

            print(f'Epoch : {epoch} | Epoch loss: {epoch_loss}')


            if self.gpu_id ==0 and epoch% self.save_every==0:
                self._save_checkpoints(epoch)

        
    def _save_checkpoints(self,epoch):

        ckp = self.model.module.state_dict()
        PATH = 'checkpoint.pth'
        torch.save(ckp,PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    
                
def load_train_objs():
    train_set = datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1,5,3),
        torch.nn.Flatten(1),
        torch.nn.Linear(3380,64),
        torch.nn.Linear(64,10))
    
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    return train_set,model,optimizer

def prepare_dataloader(dataset,batch_size:int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank,world_size,total_epochs,save_every,batch_size):
    ddp_setup(rank,world_size=world_size)
    dataset,model,optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset,batch_size)
    trainer = Trainer(model,train_data,optimizer,rank,save_every)
    trainer.train_loop(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    world_size = torch.cuda.device_count()
    mp.spawn(main,args=(world_size,total_epochs,save_every,batch_size),nprocs=world_size)



