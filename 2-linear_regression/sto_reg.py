import torch
import matplotlib.pyplot as plt

w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)

X = torch.arange(-3,3,0.1).view(-1,1)
f = -3*X 

plt.plot(X.numpy(), f.numpy())
plt.show()

Y = f+0.1*torch.randn(X.size())

plt.plot(X.numpy(), f.numpy())
plt.plot(X.numpy(), Y.numpy(), 'ro')
plt.show()

def forward(x):
    y=w*x + b
    return y 

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

lr = 0.1 # learning rate 
LOSS = []
for epoch in range(4):
    Yhat = forward(X)
    LOSS.append(criterion(Yhat,Y))
    for x,y in zip(X,Y):
        yhat = forward(x)
        loss = criterion(yhat,y)
        loss.backward()
        w.data = w.data - lr*w.grad.data
        b.data = w.data - lr*b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

plt.plot(LOSS)
plt.show()
