import torch as t 
import torch.nn as nn
from optimizers import SGD, Adam
import torch.optim
from model import MLP

xdata = t.tensor([[1.0],[2.0],[3.0]])
ydata = xdata*3 

model = nn.Sequential(
    nn.Linear(1,1, bias=False)
)
for p in model.parameters():
    print("starting param ")
    print(p)
# p = limodel.parameters())

# opt = t.optim.SGD(model.parameters(), lr=1e-3)
# opt.step()
optimizer = Adam(model.parameters(), lr=1e-3, b1=0.9, b2=0.999, ep=10e-8)
for i in range(10):
    f = t.norm((model(xdata)-ydata)**2)
    f.backward()
    optimizer.step()
    optimizer.zero_grad()

for m in model.parameters():
    print("new params are ")
    print(m)
    # print("\n gradient clear?")
    # print(m.grad)

    #try to cross check implementation by running pytorch version and our version at the same time