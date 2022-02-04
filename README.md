# fmnist_torch

### fmnist dataset info

https://github.com/zalandoresearch/fashion-mnist

training set: 60,000 samples [1,28,28]

test set: 10,000 samples

labels:

| Label | Description |
|-------|-------------|
|   0   | T-shirt/top |
|   1   |   Trouser   |
|   2   |   Pullover  |
|   3   |    Dress    |
|   4   |     Coat    |
|   5   |    Sandal   |
|   6   |    Shirt    |
|   7   |   Sneaker   |
|   8   |     Bag     |
|   9   |  Ankle boot |

loading data:

```python
mean,std=(0.5,),(0.5,)

tf=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize(mean,std)])

train=datasets.FashionMNIST('~/.pytorch/FMNIST',download=True,train=True,transform=tf)
test=datasets.FashionMNIST('~/.pytorch/FMNIST',download=True,train=False,transform=tf)

train_loader=torch.utils.data.DataLoader(train,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(test,batch_size=64,shuffle=False)
```

to read each sample from **train**

```python
len(train) #60,000

type(train[0]) #tuple

len(train[0]) #2

train[0][0] #shape: torch.Size([1, 28, 28])

train[0][1] #label int
```

to read each sample from **train_loader**

```python
img,label=next(iter(train_loader))

img.shape,img.view(img.shape[0],-1).shape

#(torch.Size([64, 1, 28, 28]), torch.Size([64, 784]))

#show first image
plt.imshow(img[0][0],cmap='gray')

#show first label
label[0] #tensor(5)
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
