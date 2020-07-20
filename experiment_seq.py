import torch
from nisrep.normalizing_flows.manager import *
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
def csq(x):
    ret=torch.cos(x[:,0])**2
    
    return ret


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
    f=csq
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

    sig,sig_err=NF._train_variance_forward_seq(f,optim,logdir,batch_size,epoch_length,0,True, False,True,_run,dev,log)
    end_time=datetime.datetime.utcnow()
    
    print('RESULT')
    print(str(sig) + " +/- " +str(sig_err))
    sig=sig/(2.56819*10**(-9)) #GeV**2 -> pb
    sig_err=sig_err/(2.56819*10**(-9))
    print(str(sig) + " +/- " +str(sig_err))
    print('Initial loss')
    print(NF.int_loss)
    print('Epoch of best result')
    print(NF.best_epoch)
    print('Best loss')
    print(NF.best_loss)
    print('Best loss relative')
    print(NF.best_loss_rel)
    
    
    if(_run!=None):
            _run.log_scalar("training.a_integral", sig.tolist(), 0)
            _run.log_scalar("training.a_error", (sig_err).tolist(), 0)
            _run.log_scalar("training.best_loss", NF.best_loss.tolist(), 0)
            _run.log_scalar("training.best_loss_rel", (NF.best_loss_rel).tolist(), 0)
            _run.log_scalar("training.best_loss_var", NF.best_var,0)
            _run.log_scalar("training.best_epoch", NF.best_epoch, 0)
            _run.log_scalar("training.time", (end_time-start_time).total_seconds(), 0)
            _run.log_scalar("training.best_func_count", NF.best_func_count, 0)
            _run.log_scalar("training.b_varJ", (NF.varJ).tolist(), 0)
            _run.log_scalar("training.b_DKL", (NF.DKL).tolist(), 0)
            
    
    q.put((NF.best_loss.tolist(), _run._id,NF.best_loss_rel.tolist(),NF.best_func_count, NF.varJ.tolist(),
           NF.DKL.tolist(),NF.best_var, NF.best_epoch,"NIS",(end_time-start_time).total_seconds(),internal_id,sig.tolist(), 
           sig_err.tolist()))
    pass

    

   


@ex.automain
def main(_run):
    return(run(_run), _run._id)
    