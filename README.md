# fmnist_torch

### fmnist dataset info

https://github.com/zalandoresearch/fashion-mnist

training set: 60,000 samples

test set: 10,000 samples

loading data:

```
mean,std=(0.5,),(0.5,)

tf=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize(mean,std)])

train=datasets.FashionMNIST('~/.pytorch/FMNIST',download=True,train=True,transform=tf)
test=datasets.FashionMNIST('~/.pytorch/FMNIST',download=True,train=False,transform=tf)

train_loader=torch.utils.data.DataLoader(train,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(test,batch_size=64,shuffle=False)
```

### create git repo

git new repo, copy HTTPS address

in local folder:

    git init

    git add .

    git commit -m 'linear'

    git remote add origin https://github.com/ha5ha6/fmnist_torch.git

    git pull origin main --allow-unrelated-histories (cuz remote has README.md)

    git branch -M main (change branch from master to main)    

    git push -u origin main

after made some change:

    git add .

    git commit -m 'some change'

    git push -u origin main
