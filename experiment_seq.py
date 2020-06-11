import torch
from binNF.normalizing_flows.manager import *
import matplotlib
from pathlib import Path
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from multiprocessing import Process,Queue,Manager
import datetime
ex = Experiment()
LOGDIR='logs/sacred/mfruns1'

#ex.observers.append(MongoObserver(url='92.194.61.224:27017',
                                  #db_name='MY_DB'))




def camel(x):
    return torch.exp( -((x[:,0]-0.75)**2+(x[:,1]-0.75)**2)/(0.2**2))+torch.exp( -((x[:,0]-0.25)**2+(x[:,1]-0.25)**2)/(0.2**2))

def ex_init(logdir):
    LOGDIR=logdir
    ex.observers.append(FileStorageObserver(LOGDIR))

@ex.config
def cfg():
    m=Manager()
    n_flow=2
    n_cells=2
    n_bins=7
    NN_length=5
    NN_width=11
    lr=2e-3
    weight_decay=5e-07
    batch_size=80000
    epoch_length=5000
    f=camel
    logdir='logs/sacred/mfruns1'
    q=m.Queue()
    dev=0
    gn=1
    gw=1
    internal_id=0
    log=True

    
    


@ex.capture
def run(_run,n_flow, n_cells, n_bins, NN_length, NN_width, lr, weight_decay, batch_size, epoch_length, f, logdir,q,dev, internal_id, log):
    start_time=datetime.datetime.utcnow()
    
    # We define our NormalizingFlow object 
    NF =  PWQuadManager(n_flow=n_flow)
    
    NF.create_model(n_cells=n_cells, n_bins=n_bins, NN=[NN_width]*NN_length, dev=dev)

    optim = torch.optim.Adamax(NF._model.parameters(),lr=lr, weight_decay=weight_decay) 

    NF._train_variance_forward_seq(f,optim,logdir,batch_size,epoch_length,0,True, False,True,_run,dev,log)
    end_time=datetime.datetime.utcnow()
    
    print('Initial loss')
    print(NF.int_loss)
    print('Epoch of best result')
    print(NF.best_epoch)
    print('Best loss')
    print(NF.best_loss)
    print('Best loss relative')
    print(NF.best_loss_rel)
    
    
    if(_run!=None):
            _run.log_scalar("training.best_loss", NF.best_loss.tolist(), 0)
            _run.log_scalar("training.best_loss_rel", (NF.best_loss_rel).tolist(), 0)
            _run.log_scalar("training.best_loss_var", NF.best_var,0)
            _run.log_scalar("training.best_epoch", NF.best_epoch, 0)
            _run.log_scalar("training.time", (end_time-start_time).total_seconds(), 0)
            _run.log_scalar("training.best_func_count", NF.best_func_count, 0)
            _run.log_scalar("training.b_varJ", (NF.varJ).tolist(), 0)
            _run.log_scalar("training.b_DKL", (NF.DKL).tolist(), 0)
            
    
    q.put((NF.best_loss.tolist(), _run._id,NF.best_loss_rel.tolist(),NF.best_func_count, NF.varJ.tolist(),
           NF.DKL.tolist(),NF.best_var, NF.best_epoch,"NIS",(end_time-start_time).total_seconds(),internal_id))
    pass
"""
    w = torch.empty((12100,2)) 
    torch.nn.init.uniform_(w,0,1)

    Y=NF.format_input(w,dev)
    X=NF.model(Y)
    X=X.data.cpu().numpy()
    
    try:
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
        if(NF.best_var.tolist()<0.02):
            plt.savefig(logdir+'/'+str(_run._id)+'pdf.png', dpi=50)
            ex.add_artifact(logdir+'/'+str(_run._id)+'pdf.png')
        else:
            print("plot not necessary")
    except:
        print("plot not possible")
    """
    #return NF.best_var
    

   


@ex.automain
def main(_run):
    return(run(_run), _run._id)
    