import torch
from binNF.normalizing_flows.manager import *
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp

TF_CPP_MIN_LOG_LEVEL="2"


def step(x):
    a=torch.zeros_like(x[:,0])
    b=torch.ones_like(x[:,0])
   
    return torch.where(torch.max(abs(x[:,0]), abs(x[:,1]))>1,a,b)

def g(x):
   
    q=torch.max(torch.abs(x),dim=-1).values
  
    a=torch.zeros_like(q)
    b=torch.ones_like(q)
   
    return torch.where(q<0.5,a,b)

def gaussian(x):
    return torch.exp( -((x[:,0]-0.5)**2+(x[:,1]-0.5)**2)/(0.3**2)) 

def camel(x):
    return torch.exp( -((x[:,0]-0.75)**2+(x[:,1]-0.75)**2)/(0.2**2))+torch.exp( -((x[:,0]-0.25)**2+(x[:,1]-0.25)**2)/(0.2**2))

def gaussianb(x):
    return torch.exp( -(x)**2)[:,0]

def gaussianbnp(x):
    return np.exp( -((x[:,0]+1)**2+(x[:,1])**2) )

def gaussiannp(x):
    return np.exp( -(x[:,0])**2 )

def con(x):
    y=torch.empty(x.shape[0])
    return y.fill_(5)

def sin(x):
    return 2+torch.sin(x[:,1])

def lin(x):
    return 0.2*x[:,0]+0.5

def sinnp(x):
    return 2+np.sin(x[:,1])

#TODO: FIX EARLY STOPPING ISSUES
def run():
    torch.multiprocessing.freeze_support()
    n_flow = 2      # number of dimensions
    
    
    # We define our NormalizingFlow object 
    NF =  PWLinManager(n_flow=n_flow)
    NF.create_model(n_pass_through=1,n_cells=2, n_bins=25, NN=[10,10,10,10,10], roll_step=1)
    
    #shm=torch.zeros(6)
    
    #shm.share_memory_()
    mp.spawn(NF._train_variance_forward,args=(camel,"./logs/tmp2/",3e-4, 1e-7,3000,400,0,True, True,True,6,),nprocs=6, daemon=True)
    
    checkpoint = torch.load("./logs/tmp2/loss-model.txt")
   
   
    NF.model.load_state_dict(checkpoint['model_state_dict'])
    int_loss = checkpoint['int_loss']
    int_var=checkpoint['int_var']
    end_loss=checkpoint['end_loss']
    end_var=checkpoint['end_var']
    NF.model.eval()
    print('Initial loss')
    print(int_loss)
    print('Initial variance')
    print(int_var)
    
    print('End loss')
    print(end_loss)
    print('End variance')
    print(end_var)
    
    
    w = torch.empty((12100,2)) 
    torch.nn.init.uniform_(w,0,1)
   
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    Y=NF.format_input(w, dev)
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


if __name__ == '__main__':
    run()


