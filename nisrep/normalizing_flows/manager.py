import torch as torch
import copy
import os
import yaml
import numpy as np
from tqdm.autonotebook import tqdm
from .layers.coupling_cells import  AffineCoupling, PWLin, PWQuad
from .layers.layers import *
from .misc import tqdm_recycled
from statistics import mean
import time
import gc
import datetime
import math



TF_CPP_MIN_LOG_LEVEL="2"

def get_bin(x, n=0):
    """
    Get the binary representation of x.

    Parameters
    ----------
    x : int
    n : int
        Minimum number of digits. If x needs less digits in binary, the rest
        is filled with zeros.

    Returns
    -------
    list of binary digits
    """
    y=format(x, 'b').zfill(n)
    return [int(i) for i in str(y)]
    

def normal(x,mu, sigma, n_flow):
    return (torch.exp(-torch.sum((x-mu)**2/(2*sigma**2),-1)))/(sigma*np.sqrt((2*np.pi)**n_flow))

class ModelAPI():

    @property
    def model(self):
        if self._model is not None:
            return self._model
        else:
            raise AttributeError("No model was instantiated")


class BasicManager(ModelAPI):
    """Basic training methods"""
    

    def __init__(self,
                  n_flow=2,*args):
    
        self.n_flow = n_flow
        self._model = None
        self._inverse_model = None
    
        self.optimizer_object = None

    
    def _train_variance_forward_seq(self, f, optimizer_object, log=True, logdir=None, batch_size = 10000, epochs=10,
                                    epoch_start=0,
                               pretty_progressbar=True,  
                                    save_best=True, run=None,dev=0,mini_batch_size=2000, integrate=False,
                                    preburn_time=75, kill_counter=7, impr_ratio=1e-2):
        """Train the model using the integrand variance as loss and compute the Jacobian in the forward pass
        (fixed latent space sample mapped to a phase space sample)

        """
        
        dev = torch.device("cuda:"+str(dev)) if torch.cuda.is_available() else torch.device("cpu")
        
        if preburn_time>10:
            check_time=preburn_time
        else:
            check_time=50
        
        if(mini_batch_size>batch_size):
            mini_batch_size=batch_size
        n_minibatches = int(batch_size/mini_batch_size)
        
        if(run!=None and log): 
            filename=logdir+"/"+str(_run._id)+"/torch"
            filename2=filename+"_int"
            if not os.path.exists(logdir+"/"+str(_run._id)):
                os.mkdir(logdir+"/"+str(_run._id))
   
        else:
            filename=logdir+"/torch"
            filename2=filename+"_int"
            if not os.path.exists(logdir):
                os.makedirs(logdir)
   
        try:
            if(log):
                file= open(filename,"w+")
                file2=open(filename2,"w+")
                file2.close()
                file.close()
                torch.save({
                    'model_state_dict': self.best_model.state_dict()
                    },filename2)

        except:
            print("Torch save not possible")
            
        integ=torch.zeros((epochs+1,),device=dev)
        err=torch.zeros((epochs+1,),device=dev)
        
         
        
        # Instantiate a pretty progress bar if needed
        if pretty_progressbar:
            epoch_progress = tqdm(range(epoch_start,epoch_start+epochs), leave=False, desc="Loss: {0:.3e} | Epoch".format(0.))
            # Instantiate a pretty progress bar for the minibatch loop if it is not trivial
            if n_minibatches>1:
                minibatch_progress = tqdm_recycled(range(n_minibatches), leave=False, desc="Step")
            else:
                minibatch_progress = range(n_minibatches)

        else:
            epoch_progress = range(epoch_start, epoch_start+epochs)
            minibatch_progress = range(n_minibatches)

        i=0
        self.model.to(dev)
        self.best_loss=0
       
        self.best_var=0
        maxf=0
        
        while (i<self.n_flow):
            w = torch.empty(2*mini_batch_size, self.n_flow).to(torch.double)
     
            torch.nn.init.uniform_(w).to(dev)  # Generate a batch of points in latent space
       
      
       
        
            dkl=torch.nn.KLDivLoss(reduction='batchmean')
            fres=f(w)
            integ[0]+=torch.sum(fres)/(self.n_flow*2*mini_batch_size)
            err[0]+=torch.var(fres)/(self.n_flow)
            if torch.max(fres)>maxf:
                maxf=torch.max(fres) 
            
            self.best_loss+=torch.var(fres/maxf).detach()/self.n_flow
           
            
            i=i+1
            
            
            self.best_var+=float((torch.var((fres)**2)/2*mini_batch_size).detach())
        
            
        
        
        if save_best or log:
            
            
            
            XJ = self.model(  # Pass through the model
                self.format_input(  # Append a unit Jacobian to each point
                    w,dev
                )
              )
            X=XJ[:,:-1]
            J = XJ[:,-1]
            self.varJ=torch.mean(J**2).detach()

            self.DKL=dkl(torch.log(X+torch.ones_like(X).fill_(1e-45)).to(dev),w.to(dev)).detach()
            

            self.best_model=copy.deepcopy(self.model)
           

            self.best_epoch=0
            self.best_time=0
            self.best_loss_rel=torch.ones_like(self.best_loss)
            self.func_count=2*mini_batch_size*self.n_flow/(batch_size/mini_batch_size)
            self.best_func_count=2*mini_batch_size*self.n_flow/(batch_size/mini_batch_size)
            del XJ,X,J
        if(run!=None and log):
            run.log_scalar("training.int_loss", self.best_loss.tolist(), 0)
       

        self.int_loss=self.best_loss
           
    
        
        stale_save=1000
        preburner=True
      
        counter=0
        last_loss=1000
        #print("START")
        for i in epoch_progress:
            loss = 0
            loss2=0
            loss3=0
            var = 0
            optimizer_object.zero_grad()
            
            for j in minibatch_progress:
                
                w = torch.empty(mini_batch_size, self.n_flow).to(dev).to(torch.double)
                torch.nn.init.uniform_(w)
               
                
                XJ = self.model(                                            # Pass through the model
                    self.format_input(                                      # Append a unit Jacobian to each point
                        w,dev
                    )
                )

                
                # Separate the points and their Jacobians:
                # This sample is fixed, we optimize the Jacobian
                X = (XJ[:, :-1]).detach()

               
                if(preburner):
                    fres=f(w)
                    fXJ=torch.mul(fres, XJ[:, -1])/maxf
                    
                    integ[i+1]+=torch.mean(fres)/n_minibatches
                    err[i+1]+=torch.var(fres)/n_minibatches
                                        
                else:
                    fres=torch.mul(f(X), XJ[:, -1])
                    fXJ = fres/maxf
                    
                    integ[i+1]+=torch.mean(fres.detach())/n_minibatches
                    err[i+1]+=torch.var(fres.detach())/n_minibatches
                    
               
                    
                if(save_best):
                    self.func_count=self.func_count+1
                
                loss+=torch.var(fXJ)
                
                
                # The Monte Carlo integrand is fXJ: we minimize its variance up to the constant term
                #in newer versions, the variance estimator is used in order to increase
                #stability for flat distributions
                
                var+=float(torch.var(fXJ**2)/mini_batch_size) # variance of the mean is variance/N
                del X, fXJ, XJ
                gc.collect()
                
      
           
           
            loss=loss/n_minibatches
           
            
            loss.backward()
            
            optimizer_object.step()
            
            
            # Update the progress bar
            if pretty_progressbar:
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))
            if(run!=None and log):       
                run.log_scalar("training.loss", loss.tolist(), i)
                run.log_scalar("training.loss_rel",(loss/self.int_loss).tolist(),i)
               

            if (save_best or log):     
                self.best_func_count=self.func_count*mini_batch_size
            if (save_best or log) and (loss < self.best_loss) and not preburner:
                self.best_loss = loss
                self.best_var=var
                self.best_loss_rel=loss/self.int_loss
                self.best_model=copy.deepcopy(self.model)
                self.best_epoch=i
                
                
                if(run!=None):
                    self.best_time=(datetime.datetime.utcnow()-run.start_time).total_seconds()
                else:
                    self.best_time=0
                
                    
            if loss<last_loss:
                counter=0
            else:
                counter+=1
                if counter>kill_counter and preburner:
                    counter=0
                    preburner=False
                elif counter>kill_counter:
                    break
            last_loss=loss
            if ( i%check_time==0) and i>(preburn_time+1) and float(self.best_loss/stale_save)>(1-impr_ratio) and not preburner:
                break
            elif  i%check_time==0 and not preburner and (self.best_loss<self.int_loss or i>300):
                
                stale_save=self.best_loss 
           
           
        
            if preburner and ((loss<0.25*self.best_loss) or i>preburn_time):
               
                preburner=False
            
            
         
       
        endpoint=i+1
        with torch.no_grad():
            if(integrate and endpoint<epochs-1):
                model=self.best_model.eval()
                for s in range(endpoint,epochs):
                    for t in range(n_minibatches):
                        w = torch.empty(mini_batch_size, self.n_flow).to(dev).to(torch.double)
                        torch.nn.init.uniform_(w)

                        XJ = model(self.format_input(w,dev)).detach() 
                        fres=torch.mul(f(XJ[:,:-1]), XJ[:, -1])
                        integ[s+1]+=torch.mean(fres.detach())/n_minibatches
                        err[s+1]+=torch.var(fres.detach())/n_minibatches
                    
                    self.best_func_count=self.best_func_count+1
       
       
        self.integ_tot=torch.sum(integ/err)/torch.sum(1/err)
        self.err_tot=torch.sqrt(1/torch.sum(1/err))/math.sqrt(epochs)
        
        
        
        if(run!=None and integrate):
            run.log_scalar("training.integ", self.integ_tot.tolist(), 0)
            run.log_scalar("training.err", self.err_tot.tolist(), 0)

        try:
            if (log):
                torch.save({
                    'best_epoch': self.best_epoch,
                    'best_loss': self.best_loss,
                    'int_loss': self.int_loss,
                    'best_loss_rel' : self.best_loss_rel,
                    'best_func_count' : self.best_func_count,
                    'model_state_dict': self.best_model.state_dict(),
                    'integ' : self.integ_tot,
                    'err' : self.err_tot
                    },filename)
        except:
            print("Torch save not possible")
        
        del loss, var, w, optimizer_object
        
        if(integrate):
            return (self.integ_tot.detach().tolist(),self.err_tot.detach().tolist())
        else:
            return (0,0)
    
  
    
    
    
   
class AffineManager(BasicManager):
    
    """A manager for normalizing flows with affine coupling cells interleaved with rolling layers that
    apply cyclic permutations on the variables. All cells have the same number of pass through variables and the
    same step size in the cyclic permutation.
    Each coupling cell has a fully connected NN with a fixed number of layers (depth) of fixed size (width)

    Hyperparameters:
    - n_pass_through
    - n_bins
    - nn_width
    - nn_depth
    - roll_step
    """

    format_input = AddJacobian()
    

    def create_model(self,
                     n_pass_through,
                     n_cells,
                     NN,
                     roll_step
                     ):
  
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        
        self._model = torch.nn.Sequential()
        
        
        for i_cell in range(n_cells):#create coupling cells
            self._model.add_module(str(i_cell),
                AffineCoupling(flow_size=self.n_flow, pass_through_size=n_pass_through,
                                                NN_layers=NN)
            )
            self._model.add_module("roll",RollLayer(roll_step)) #add roll layer
        
       
        w = torch.empty(10, self.n_flow).to(dev)
        torch.nn.init.uniform_(w)
        # Do one pass forward:
        self._model(self.format_input(w, dev))  
        
        
class PWLinManager(BasicManager):
    
    """A manager for normalizing flows with piecewise-linear coupling cells interleaved with rolling layers that
    apply cyclic permutations on the variables. All cells have the same number of pass through variables and the
    same step size in the cyclic permutation.
    Each coupling cell has a fully connected NN with a fixed number of layers (depth) of fixed size (width)

    Hyperparameters:
    - n_pass_through
    - n_bins
    - nn_width
    - nn_depth
    - roll_step
    """

    format_input = AddJacobian()
    

    def create_model(self,
                     n_pass_through,
                     n_cells,
                     n_bins,
                     NN,
                     roll_step
                     ):
       
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self._model = torch.nn.Sequential()
        
        
        for i_cell in range(n_cells):#create coupling cells
            self._model.add_module(str(i_cell),
                PWLin(flow_size=self.n_flow, pass_through_size=n_pass_through,n_bins=n_bins,
                                                NN_layers=NN)
            )
            self._model.add_module("roll",RollLayer(roll_step)) #add roll layer
        self._model.to(dev)
        self.best_model=self.model
       
        w = torch.empty(5, self.n_flow).to(torch.double)
        torch.nn.init.uniform_(w).to(dev)
        # Do one pass forward:
        self._model(self.format_input(w,dev))   
        
        
class PWQuadManager(BasicManager):
    
    """A manager for normalizing flows with piecewise-quadratic coupling cells interleaved with masking layers that
    mask subsets of the varialbes. The cells have different numbers of pass through variables.
    Each coupling cell has a fully connected NN with a fixed number of layers (depth) of fixed size (width)

    Hyperparameters:
    - n_bins
    - nn_width
    - nn_depth
    - n_cells
    """

    format_input = AddJacobian()
    

    def create_model(self,
                     n_cells,
                     n_bins,
                     NN,
                     dev=0
                     ):
    
        
        if(n_cells<2*np.ceil(np.log2(self.n_flow)) and n_cells<self.n_flow):
            
            if(self.n_flow<=6):
                n_cells=self.n_flow
            elif self.n_flow==7:
                n_cells=6
            else:
                n_cells=int(2*np.ceil(np.log2(self.n_flow)))
            print("Adjusted # coupling cells to "+str(n_cells))
        dev = torch.device("cuda:"+str(dev)) if torch.cuda.is_available() else torch.device("cpu")
       
        
        self._model = torch.nn.Sequential()
        if(self.n_flow<=7):
            if(self.n_flow<=6):
                roll_step=1
                n_pass_through=1
            elif (self.n_flow==7):
                roll_step=1
                n_pass_through=2

            for i_cell in range(n_cells):#create coupling cells
                self._model.add_module(str(i_cell),
                    PWQuad(flow_size=self.n_flow, pass_through_size=n_pass_through,n_bins=n_bins,
                                                    NN_layers=NN)
                )
                if(i_cell<n_cells-1):
                    self._model.add_module("roll"+str(i_cell),RollLayer(roll_step)) #add roll layer
                else:
                     self._model.add_module("roll"+str(i_cell),RollLayer(self.n_flow-((n_cells-1)%self.n_flow)))
                    
                     
                
        else:
            roll_step=1
            n_pass_through=int(self.n_flow/2)
            dims=torch.arange(self.n_flow)
            n=len(get_bin(self.n_flow-1,0))
            dims_bin=torch.ones(self.n_flow,n)
            
            dims_bin=torch.IntTensor(list(map(get_bin, dims,[n]*self.n_flow))).to(dev)
            
            for i_cells in range(2*n):#create coupling cells
                masker=MaskLayer(dims_bin,i_cells,dev)
                self._model.add_module("mask"+str(i_cells),masker)
                self._model.add_module(str(i_cells),
                    PWQuad(flow_size=self.n_flow, pass_through_size=masker.pass_through,n_bins=n_bins,
                                                    NN_layers=NN)
                )

                self._model.add_module("demask"+str(i_cells),DeMaskLayer(masker.feeder,masker.trafoer))
            for i_cells in range(n_cells-2*n):
                self._model.add_module(str(i_cells+2*n),
                    PWQuad(flow_size=self.n_flow, pass_through_size=n_pass_through,n_bins=n_bins,
                                                    NN_layers=NN)
                )
                if(i_cells<n_cells-2*n-1):
                    self._model.add_module("roll"+str(i_cells+2*n),RollLayer(roll_step)) #add roll layer
                else:
                    self._model.add_module("roll"+str(i_cells+2*n),RollLayer(self.n_flow-((n_cells-2*n-1)%self.n_flow)))
                    
       
        
        self._model.to(dev).to(torch.double)
        self.best_model=self.model
          
        w = torch.empty(5, self.n_flow).to(torch.double)
        torch.nn.init.uniform_(w)
       
        
        # Do one pass forward:
        
        self._model(self.format_input(w,dev)) 
       
        return  
    

    
