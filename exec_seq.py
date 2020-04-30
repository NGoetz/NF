import torch
from binNF.normalizing_flows.manager import *
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp

def camel(x):
    return torch.exp( -((x[:,0]-0.75)**2+(x[:,1]-0.75)**2)/(0.2**2))+torch.exp( -((x[:,0]-0.25)**2+(x[:,1]-0.25)**2)/(0.2**2))

n_flow = 2      # number of dimensions


# We define our NormalizingFlow object 
NF =  PWLinManager(n_flow=n_flow)

NF.create_model(n_pass_through=1,n_cells=2, n_bins=25, NN=[10,10,10,10,10], roll_step=1)
optim = torch.optim.Adamax(NF._model.parameters(),lr=3e-4, weight_decay=1e-7) 

history=NF._train_variance_forward_seq(camel,optim,"./logs/tmp/",9000,500,0,True, True,True)


print('Initial loss')
print(NF.int_loss)
print('Initial variance')
print(NF.int_var)
print('Epoch of best result')
print(NF.best_epoch)
print('Best loss')
print(NF.best_loss)
print('Best variance')
print(NF.best_var)
      
"""
losses=[]
for key, value in NF.history.items():
    losses.append(value["loss"])

fig = plt.figure(figsize=(12, 4))
a1=fig.add_subplot(131)
plt.plot(losses)

a1.title.set_text('Loss')
a2=fig.add_subplot(132)
plt.plot(np.sqrt(np.exp(losses)))
a2.title.set_text('Standard Deviation')
plt.show(block=True)

w = torch.empty((12100,2)) 
torch.nn.init.uniform_(w,0,1)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

Y=NF.format_input(w,dev)
X=NF.model(Y)
X=X.data.cpu().numpy()
fig = plt.figure(figsize=(12, 6))
a3=fig.add_subplot(111)

plt.hist2d(X[:,0],X[:,1],bins=25)
axes = plt.gca()
axes.set_xlim([-0,1]) 
axes.set_ylim([-0,1])



a3.title.set_text('Point histogram (PDF)')
a3.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    left=True,         # ticks along the top edge are off
    labelbottom=True,
    labelleft=True)
a3.set_aspect(aspect=1.)
plt.show(block=True)

"""