import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

mean,std=(0.5,),(0.5,)

tf=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize(mean,std)])

train=datasets.FashionMNIST('~/.pytorch/FMNIST',download=True,train=True,transform=tf)
test=datasets.FashionMNIST('~/.pytorch/FMNIST',download=True,train=False,transform=tf)

train_loader=torch.utils.data.DataLoader(train,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(test,batch_size=64,shuffle=False)

model=nn.Sequential(nn.Linear(784,128),
                    nn.ReLU(),
                    nn.Linear(128,64),
                    nn.ReLU(),
                    nn.Linear(64,10),
                    nn.LogSoftmax(dim=1))

criterion=nn.NLLLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

n_ep=20
train_losses=[]
valid_losses=[]

for ep in range(n_ep):
    train_loss=0

    for imgs,labels in train_loader:
        optimizer.zero_grad()
        out=model(imgs.view(imgs.shape[0],-1))
        loss=criterion(out,labels)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        #print(f"Batch: {batch_num}, Loss: {loss.item()}")

    train_losses.append(train_loss/len(train_loader))

    valid_loss=0
    for imgs,labels in test_loader:

        out=model(imgs.view(imgs.shape[0],-1))
        loss=criterion(out,labels)
        valid_loss+=loss.item()

    valid_losses.append(valid_loss/len(test_loader))
    print(f"Ep: {ep}, Training loss: {train_loss/len(train_loader)}, Valid loss: {valid_loss/len(test_loader)}")

correct=0
total=0

predictions=[]

with torch.no_grad():
    for imgs,labels in test_loader:
        out=model(imgs.view(imgs.shape[0],-1))
        _, pred=torch.max(out.data,1)
        predictions.append(pred.cpu().detach().numpy())
        total+=labels.size(0)
        correct+=(pred==labels).sum().item()

print('Accuracy: %d %%' % (100 * correct / total))

y_pred=np.concatenate(predictions).ravel()
y_test=test.targets.cpu().detach().numpy()
report=classification_report(y_pred,y_test,output_dict=True)
pd.DataFrame(report).transpose().to_csv('classification_report.csv')

print('Report Saved!')

torch.save(model.state_dict(),'save_model.pth')

print('Model Saved!')

plt.figure()
plt.plot(train_losses,label='train_loss')
plt.plot(valid_losses,'r',label='valid_loss')
plt.legend()
plt.savefig('train_valid_result.png',dpi=350)

print('Train_Valid Plot Saved!')
