import torch
from binNF.normalizing_flows.manager import *
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from multiprocessing import Process,Queue,Manager
ex = Experiment()

#ex.observers.append(MongoObserver(url='92.194.61.224:27017',
                                  #db_name='MY_DB'))

ex.observers.append(FileStorageObserver('logs/sacred/runs1'))


def camel(x):
    return torch.exp( -((x[:,0]-0.75)**2+(x[:,1]-0.75)**2)/(0.2**2))+torch.exp( -((x[:,0]-0.25)**2+(x[:,1]-0.25)**2)/(0.2**2))

@ex.config
def cfg():
    m=Manager()
    n_flow=2
    n_pass_through=1
    n_cells=2
    n_bins=25
    NN_length=5
    NN_width=10
    lr=3e-4
    weight_decay=1e-7
    batch_size=9000
    epoch_length=5000
    f=camel
    logdir='logs/sacred/runs1'
    q=m.Queue()
    dev=0
    
    


@ex.capture
def run(_run,n_flow,n_pass_through, n_cells, n_bins, NN_length, NN_width, lr, weight_decay, batch_size, epoch_length, f, logdir,q,dev):
    
    # We define our NormalizingFlow object 
    NF =  PWLinManager(n_flow=n_flow)

    NF.create_model(n_pass_through=n_pass_through,n_cells=n_cells, n_bins=n_bins, NN=[NN_width]*NN_length, roll_step=1)
    optim = torch.optim.Adamax(NF._model.parameters(),lr=lr, weight_decay=weight_decay) 

    NF._train_variance_forward_seq(f,optim,logdir,batch_size,epoch_length,0,True, False,True,_run,dev)


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
    #s=q.get()
    #if(s[0]<NF.best_var.tolist()):
     #   q.put(s)
    #else:
    q.put((NF.best_var.tolist(), _run._id,))
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
"""
    w = torch.empty((12100,2)) 
    torch.nn.init.uniform_(w,0,1)

    dev = torch.device("cuda:"+str(dev)) if torch.cuda.is_available() else torch.device("cpu")

    Y=NF.format_input(w,dev)
    X=NF.model(Y)
    X=X.data.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    a3=fig.add_subplot(111)
    try:
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
        if(NF.best_var.tolist()<0.02):
            plt.savefig(logdir+'/'+str(_run._id)+'pdf.png', dpi=50)
            ex.add_artifact(logdir+'/'+str(_run._id)+'pdf.png')
        else:
            print("plot not necessary")
    except:
        print("plot not possible")
    #return NF.best_var
    return NF.best_var.tolist()

   


@ex.automain
def main(_run):
    return(run(_run), _run._id)
    