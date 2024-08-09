import torch
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
import os

def ddp_setup():

    init_process_group(backend='nccl')

class Trainer:

    def __init__(self,model,train_loader,optimizer,save_every,snapshot_path) -> None:
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.loader = train_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print("Loading snapshop")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model,device_ids=[self.gpu_id])

    def _load_snapshot(self,snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f'Resuming training from snapshot at Epoch {self.epochs_run}')


    def train_loop(self,max_epochs):

        for epoch in range(self.epochs_run,max_epochs):

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

        
    def _save_snapshot(self,epoch):
        snapshot = {}
        snapshot['MODEL_STATE'] = self.model.module.state_dict()
        snapshot['EPOCHS_RUN'] = epoch
        #PATH = 'checkpoint.pth'
        torch.save(snapshot,'snapshot.pth')
        print(f"Epoch {epoch} | Training checkpoint saved at snapshot.pth")

    
                
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

def main(total_epochs,save_every,snapshot_path = 'snapshot.pth'):
    ddp_setup()
    dataset,model,optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset,batch_size=64)
    trainer = Trainer(model,train_data,optimizer,save_every,snapshot_path)
    trainer.train_loop(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    
    main(total_epochs,save_every)



