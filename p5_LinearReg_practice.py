import numpy as np
import matplotlib.pyplot as plt
import torch

x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):

        super(LinearModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred

model=LinearModel()

criterion = torch.nn.MSELoss(size_average=False)

optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred =model(x_data)
    loss =criterion(y_pred,y_data)
    print(y_pred)
    print("y_pred的类型是:",type(y_pred))
    print("loss的类型是:",type(loss))

    print("训练轮次是:",epoch,"loss.item()=",loss.item())
    optimizer.zero_grad()
    
    loss.backward()

    optimizer.step()

print("w= ",model.linear.weight.item())
print("b= ",model.linear.bias.item())

x_test =torch.Tensor([4.0])
y_test= model(x_test)

print("y_test的类型是:",type(y_test))

print("y_pred = ",y_test.item())

print(type(torch.nn.Linear))