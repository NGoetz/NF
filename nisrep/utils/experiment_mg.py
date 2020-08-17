import torch
from nisrep.normalizing_flows.manager import *
import datetime
import pprint
import os
import random 
import gc
import numpy as np

epoch_length=800

def pro(para):
    
   
    n_flow=para["n_flow"]
    n_bins=para["n_bins"]
    NN_width=para["NN_width"]
    NN_length=para["NN_length"]
    dev=para["dev"]
    lr=para["lr"]
    wd=para["weight_decay"]
    var_n=para["var_n"]
    batch_size=para["batch_size"]
    pt=para["pt"]
    q=para["q"]
    internal_id=para["id"]
    f=para["f"]
    dev=para["dev"]
    logdir=para["logdir"]+"/"+str(internal_id)
    log=para["log"]
  
    start_time=datetime.datetime.utcnow()
    if not os.path.exists(logdir):
                os.mkdir(logdir)
    filename=logdir+'/log.txt'
    file=open(filename,"w+") 
    file.write("Started \n")
    file.write("Batchsize: "+ str(batch_size) + " n_bins: " +str(n_bins) + " NN_length: "+str(NN_length) + 
          " NN_width: "+str(NN_width)+'\n')
    file.write("LR: "+str(lr)+" weight decay: "+str(wd)+ " preburn_time: "+str(pt)+'\n')
    file.write("-----------"+'\n')
    # We define our NormalizingFlow object 
    NF =  PWQuadManager(n_flow=n_flow)
    
    file.write("Create Model \n")
    file.write("Device: "+str(dev)+"\n" )
    NF.create_model(n_cells=2, n_bins=n_bins, NN=[NN_width]*NN_length, dev=dev) #cells automatically adapted
   
    file.write("Model Created \n")
    optim = torch.optim.Adamax(NF._model.parameters(),lr=lr, weight_decay=wd) 
    file.write("Start training \n")
      
        
    
    sig,sig_err=NF._train_variance_forward_seq(f,optim,log, logdir,batch_size,epoch_length,
                                           pretty_progressbar=False,save_best=True,run=None,dev=dev,
                                               integrate=True,mini_batch_size=batch_size, preburn_time=pt)
     
    file.write("End training \n")
    end_time=datetime.datetime.utcnow()
    file.write("{0:5E}  +/- {1:3E}  \n".format(sig,sig_err))
    sig=sig/(2.56819*10**(-9)) #GeV**2 -> pb
    sig_err=sig_err/(2.56819*10**(-9))
    
    
    dev = torch.device("cuda:"+str(dev)) if torch.cuda.is_available() else torch.device("cpu")
    w = torch.empty(var_n, NF.n_flow, device=dev)
    torch.nn.init.uniform_(w)
    
    Y=NF.format_input(w, dev=dev)
    NF.best_model.to(dev)
    
    X=NF.best_model(Y)
    v_var=torch.var(f(X[:,:-1])*X[:,-1]).detach()
    w_max=torch.max(f(X[:,:-1])*X[:,-1]).cpu().detach().tolist()
    w_mean=torch.mean(f(X[:,:-1])*X[:,-1]).cpu().detach().tolist()
    
    bl=NF.best_loss.detach().tolist()
    blr=NF.best_loss_rel.detach().tolist()
    bfc=NF.best_func_count
    vJ=NF.varJ.detach().tolist()
    DKL=NF.DKL.detach().tolist()
    bv=NF.best_var
    be=NF.best_epoch
    q.put((bl, None,blr,bfc, vJ,
           DKL,bv,be ,"NIS",(end_time-start_time).total_seconds(),internal_id,sig, 
           sig_err,v_var.tolist()))
    
   
        
   


    
    file.write("Final Variance: {0:5E} \n".format(v_var))
    file.write("{0:5E}  +/- {1:3E} pb \n".format(sig,sig_err))
    
    w = torch.empty(int(var_n/2), NF.n_flow,device=dev)
    std=torch.zeros((20,))
    mean=torch.zeros((20,))
    for i in range(20):
        torch.nn.init.uniform_(w)
        Y=NF.format_input(w, dev=dev)
        X=NF.best_model(Y)
        std[i]=torch.std(f(X[:,:-1])*X[:,-1])
        mean[i]=torch.mean(f(X[:,:-1])*X[:,-1])
    sig=torch.mean(mean)
    sig_err=torch.mean(std/np.sqrt(20))
    file.write("Post training integrate:"+'\n')
    file.write("{0:5E}  +/- {1:3E} pb \n".format(sig/(2.56819*10**(-9)),sig_err/(2.56819*10**(-9))))
    file.write("Unweighting efficiency: "+str(w_mean/w_max)+'\n')
    file.write("Duration:"+'\n')
    file.write(str((end_time-start_time).total_seconds())+'\n')
    file.write("-----------"+'\n')
    file.write('Initial loss'+'\n')
    file.write(str(NF.int_loss.tolist())+'\n')
    file.write('Best loss'+'\n')
    file.write(str(NF.best_loss.tolist())+'\n')
    file.write('Best loss relative'+'\n')
    file.write(str(NF.best_loss_rel.tolist())+'\n')
    file.write('Evaluations'+'\n')
    file.write(str(NF.best_func_count)+'\n')
    file.write('Epoch'+'\n')
    file.write(str(NF.best_epoch)+'\n')
    file.write("---------------"+'\n')
    file.write(pprint.pformat(para))
    del NF,X,w,v_var,para,optim,sig,sig_err,Y,w_max,w_mean
    
   

    file.close()
    gc.collect()
   
    return