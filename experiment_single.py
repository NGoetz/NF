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
ex = Experiment()
LOGDIR='logs/sacred/single'
ex.observers.append(FileStorageObserver(LOGDIR))





def create_fun(gn, gw):
    
    if(gn==1):
        def f(x):
                return torch.exp(-torch.sum((x-0.5)**2/(gw**2),-1))
        return f
    
    if(gn==2):
        def f(x):
                return torch.exp(-torch.sum((x-0.25)**2/(gw**2),-1))+torch.exp(-torch.sum((x-0.75)**2/(gw**2),-1))
        return f
    
    if(gn==4):
        def f(x):
            shift=torch.ones_like(x)*0.25
            shift1=shift.clone()*3
            lim=int((shift.shape[1]/2))
            shift2=torch.cat((shift[:,:lim],shift1[:,lim:]),-1)
            shift3=torch.cat((shift1[:,:lim],shift[:,lim:]),-1)
            return torch.exp(-torch.sum((x-shift)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift1)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift2)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift3)**2/(gw**2),-1))
        return f
    
    if(gn==8):
        def f(x):
            shift=torch.ones_like(x)*0.25#000
            shift1=shift.clone()*3#111
            lim=int((shift.shape[1]/3))
            shift2=torch.cat((shift[:,:lim],shift1[:,lim:2*lim],shift[:,2*lim:]),-1) #010
            shift3=torch.cat((shift1[:,:lim],shift[:,lim:]),-1)#100
            shift4=torch.cat((shift1[:,:lim],shift1[:,lim:2*lim],shift[:,2*lim:]),-1) #110
            shift5=torch.cat((shift[:,:2*lim],shift1[:,2*lim:]),-1) #001
            shift6=torch.cat((shift1[:,:lim],shift[:,lim:2*lim],shift1[:,2*lim:]),-1) #101
            shift7=torch.cat((shift[:,:lim],shift1[:,lim:]),-1) #011
            return torch.exp(-torch.sum((x-shift)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift1)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift2)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift3)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift4)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift5)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift6)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift7)**2/(gw**2),-1))
        return f

@ex.config
def cfg():
    m=Manager()
    n_flow=9
    n_cells=8
    n_bins=7
    NN_length=5
    NN_width=11
    lr=2e-3
    weight_decay=5e-07
    batch_size=80000
    epoch_length=2000
    f=create_fun(1,0.22)
    logdir=LOGDIR
    q=m.Queue()
    dev=0
    gn=1
    gw=1
    internal_id=0

    
    


@ex.capture
def run(_run,n_flow, n_cells, n_bins, NN_length, NN_width, lr, weight_decay, batch_size, epoch_length, f, logdir,q,dev, internal_id):
    
    
    # We define our NormalizingFlow object 
    NF =  PWQuadManager(n_flow=n_flow)
    
    NF.create_model(n_cells=n_cells, n_bins=n_bins, NN=[NN_width]*NN_length, dev=dev)

    optim = torch.optim.Adamax(NF._model.parameters(),lr=lr, weight_decay=weight_decay) 

    NF._train_variance_forward_seq(f,optim,logdir,batch_size,epoch_length,0,True, False,True,_run,dev)
    
    w = torch.empty(batch_size, NF.n_flow)
    torch.nn.init.uniform_(w).to(dev)
    XJ = NF.best_model(NF.format_input(w,dev))
    X = (XJ[:, :-1])
    fXJ = torch.mul(f(X), XJ[:, -1])
    loss = torch.mean(fXJ**2)
    loss_rel=loss/NF.int_loss
    print('Initial loss')
    print(NF.int_loss)
    print('Epoch of best result')
    print(NF.best_epoch)
    print('Best loss')
    print(NF.best_loss)
    print('Best loss relative')
    print(NF.best_loss_rel)
    print('Validation loss')
    print(loss)
    print('Validation loss relative')
    print(loss_rel)
    
    if(_run!=None):
            _run.log_scalar("training.a_val_loss", loss.tolist(), 0)
            _run.log_scalar("training.a_val_loss_rel", (loss_rel).tolist(), 0)
            _run.log_scalar("training.a_val_loss_var", NF.best_var,0)
            _run.log_scalar("training.b_varJ", (NF.varJ).tolist(), 0)
            _run.log_scalar("training.b_DKL", (NF.DKL).tolist(), 0)
            
    
    q.put((loss.tolist(), _run._id,loss_rel.tolist(),NF.best_func_count, NF.varJ.tolist(),
           NF.DKL.tolist(),NF.best_var, NF.best_epoch,"NIS",NF.best_time,internal_id))
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
    