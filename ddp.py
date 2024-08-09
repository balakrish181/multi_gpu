import torch
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


class Trainer:

    def __init__(self,model,train_loader,optimizer,gpu_id,save_every) -> None:
        #self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.loader = train_loader
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.save_every = save_every

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


            if epoch% self.save_every==0:
                self._save_checkpoints(epoch)

        
    def _save_checkpoints(self,epoch):

        ckp = self.model.state_dict()
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
        pin_memory=True,shuffle=True
    )
import sys
def main(device,total_epochs,save_every,batch_size):
    dataset,model,optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset,batch_size)
    #print(len(train_data))
    #sys.exit()
    trainer = Trainer(model,train_data,optimizer,device,save_every)
    trainer.train_loop(total_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)



