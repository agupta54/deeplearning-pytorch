# Linear Regression in Pytorch 

```python
import torch 

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)

def forward(x):
    y = w*x + b 
    return y 
x = torch.tensor([[1.0]])

yhat = forward(x)
yhat:tensor([[1.0]])
    
from torch.nn import Linear 
model = Linear(in_features=1, out_features=1)
y = model(x)

torch.manual_seed(1)
model = Linear(in_features=1, out_features=1)
print(list(model.parameters()))

import torch.nn as nn 
class LR(nn.module):
    def __init__(self, in_size, out_size):
        super(LR,self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        out=self.linear(x) 
        return out 
```



In pytorch regression has following steps - 

* Training 
* Forward Step 

We assume a linear relation between X and Y. Each point in Y has an associated noise. The higher the variance of the noise the more dispersed is the data from a linear pattern.  

```python
import torch 
w = torch.tensor(-10.0, requires_grad=True)
X = torch.arange(-3,3,0.1).view(-1,1)
f = -3*X
plt.plot(X.numpy(),f.numpy())
plt.show()
Y = f+0.1*torch.rand(X.size())

plt.plot(X.numpy(), f.numpy())
plt.plot(X.numpy(), Y.numpy(), 'ro')

def forward(x):
    y = w*x
    return y 

def criterion(yhat, y):
    return torch.mean((yhat-y)**2)

lr = 0.1
LOSS = []
for epoch in range(4):
    Yhat=forward(X)
    loss = criterion(Yhay, Y)
    loss.backward()
    w.grad 
    w.data = w.data - lr*w.grad.data 
    w.grad.data.zero_()
    LOSS.append(loss)
    
```

