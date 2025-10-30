import numpy
import matplotlib.pyplot as plt 

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

def forward(x,w):
    return x*w

def loss(xs,ys,w):
    cost=0
    for x,y in zip(xs,ys):
        y_pre=forward(x,w)
        cost=cost+(y_pre-y)**2
    return cost/(len(xs))

def gradient(xs,ys,w):
    val=0
    for x,y in zip(xs,ys):
        val=val+x*(x*w-y)
    val=val*2/len(xs)
    return val

w=1

print(f'Predict (before training), 4, {forward(4,w)}')

x=[]
y=[]

for epoch in range(100):
    cost_val=loss(x_data,y_data,w)
    grad_val=gradient(x_data,y_data,w)
    w=w-0.01*grad_val
    x.append(epoch)
    y.append(cost_val)
    print(f'Epoch={epoch} cost_val={cost_val} grad_val={grad_val} w={w}')
print(f'Predict (after training), 4, {forward(4,w)}')

plt.plot(x,y)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()