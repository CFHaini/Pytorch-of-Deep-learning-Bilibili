import numpy as np
import matplotlib.pyplot as plt
import torch

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w1=torch.Tensor([1.0])
w1.requires_grad=True
w2=torch.Tensor([1.0])
w2.requires_grad=True
b=torch.Tensor([1.0])
b.requires_grad=True

def forward(x):
    return w1*x**2+w2*x+b
def loss(x,y):
    y_pre=forward(x)
    return (y_pre-y)**2
before_training=forward(4)

xl=[]
yl=[]

for epoch in range(2000):
    xl.append(epoch)
    l=loss(1,2)
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()
        with torch.no_grad():
            w1-=0.01*w1.grad
            w2-=0.01*w2.grad
            b-=0.01*w2.grad

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('Epoch: ',epoch,l.item())
    yl.append(l.item())

print('Predict (before training)',4, before_training.item())
print('Predict (after training)',4,forward(4).item())

fig=plt.figure()
ax=fig.add_subplot()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.plot(xl,yl)
plt.show()