import numpy as np
import vegas
from sacred import Experiment
from sacred.observers import FileStorageObserver
from multiprocessing import Process,Queue,Manager
import datetime
import copy
exv = Experiment()
LOGDIR='logs/sacred/mfruns1'


#exv.observers.append(FileStorageObserver('logs/sacred/mfruns1'))

def exv_init(logdir):
    LOGDIR=logdir
    exv.observers.append(FileStorageObserver(LOGDIR))


def camel(x):
    return torch.exp( -((x[:,0]-0.75)**2+(x[:,1]-0.75)**2)/(0.2**2))+torch.exp( -((x[:,0]-0.25)**2+(x[:,1]-0.25)**2)/(0.2**2))

@exv.config
def cfg():
    m=Manager()
    n_flow=2
    n_pass_through=1
    n_cells=2
    n_bins=7
    NN_length=5
    NN_width=11
    lr=2e-3
    weight_decay=5e-07
    batch_size=80000
    epoch_length=10000
    f=camel
    logdir='logs/sacred/mfruns1'
    q=m.Queue()
    dev=0
    ngrid=40
    gn=1
    gw=1
    
    


@exv.capture
def run(_run,n_flow,n_pass_through, n_cells, n_bins, NN_length, NN_width, lr, weight_decay, batch_size, epoch_length, f, logdir,q,dev, ngrid):
    start_time=datetime.datetime.utcnow()
    m = vegas.AdaptiveMap([[0, 1]]*n_flow, ninc=ngrid)
    #we gave NIS 3 000 000 evaluations -> 10 iterations * 300000 y's  ... 5 iterations till adaption
    ny = np.int(30000*np.sqrt(n_flow/2))
    y = np.random.uniform(0., 1., (ny, n_flow)) 
    x = np.empty(y.shape, float)           
    jac = np.empty(y.shape[0], float)
    f2 = np.empty(y.shape[0], float)
    loss_int=0
    best_loss=1000
    epch=1
    best_time=start_time
    for j in range(ny): 
        loss_int+=f(y[j])**2/ny

    for itn in range(10):                    # 7 iterations to adapt
        m.map(y, x, jac)                     # compute x's and jac
        loss=0
        for j in range(ny):                  # compute training data
            f2[j] = (jac[j] * f(x[j])) ** 2

        loss=np.mean(f2)
        
        m.add_training_data(y, f2)           # adapt
        m.adapt(alpha=1)
        if(loss<best_loss or itn==0):
            mapper=copy.deepcopy(m)
            best_loss=loss
            epch=itn+1
            
            
    best_time=datetime.datetime.utcnow()-start_time
    y = np.random.uniform(0., 1., (ny, n_flow)) 
    x = np.empty(y.shape, float) 
    mapper.map(y, x, jac)
    for j in range(ny):                  # compute training data
            f2[j] = (jac[j] * f(x[j])) ** 2
    loss=np.mean(f2)
    lossvar=np.var(f2)/ny
    


   
    lossrel=loss/loss_int
   
    print('Initial loss')
    print(loss_int)
    print('Epoch of best result')
    print(0)
    print('Best loss')
    print(loss)
    print('Best loss relative')
    print(lossrel)
    print('Validation loss')
    print(loss)
    print('Validation loss relative')
    print(lossrel)
    
    if(_run!=None):
            _run.log_scalar("training.a_val_loss", loss, 0)
            _run.log_scalar("training.a_val_loss_rel", lossrel, 0)
            _run.log_scalar("training.a_val_loss_var", lossvar,0)
            _run.log_scalar("training.b_varJ", 0, 0)
            _run.log_scalar("training.b_DKL", 0, 0)
            _run.log_scalar("training.best_time", (best_time).total_seconds(), 0)
            _run.log_scalar("training.best_func_count", epch*30000*np.sqrt(n_flow), 0)
            _run.log_scalar("training.best_iterations", epch, 0)
            
    
    q.put((loss, _run._id,lossrel,np.int(epch*30000*np.sqrt(n_flow)), 0,
           0,lossvar, epch,"VEGAS",(best_time).total_seconds(),))
    pass


@exv.automain
def main(_run):
    return(run(_run), _run._id)
    