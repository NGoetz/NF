import torch as torch
import copy
import os
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from .layers.coupling_cells import  AffineCoupling, PWLin, PWQuad
from .layers.layers import *
from .misc import tqdm_recycled
from statistics import mean
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import datetime




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
    #return format(x, 'b').zfill(n)

def tanp(x):
    return 150*(1+((torch.tan((x-0.5)*np.pi))**2))*np.pi   #derivative for jacobian

def atanp(x):
    return (1/150)*(np.pi/(x**2+np.pi**2))   #derivative for jacobian

def normal(x,mu, sigma, n_flow):
    return (torch.exp(-torch.sum((x-mu)**2/(2*sigma**2),-1)))/(sigma*np.sqrt((2*np.pi)**n_flow))

class ModelAPI():

    @property
    def model(self):
        if self._model is not None:
            return self._model
        else:
            raise AttributeError("No model was instantiated")
    
    def save_weights(self, *, logdir, prefix=""):
        """Save the current weights"""
        filename = os.path.join(logdir, "model_info", prefix, "weights.h5")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.model.state_dict,filename)


class BasicManager(ModelAPI):
    """Basic training methods"""
    

    def __init__(self,
                 *,
                 n_flow:int):
    
        self.n_flow = n_flow
        self._model = None
        self._inverse_model = None
    
        self.optimizer_object = None
        os.environ['MASTER_ADDR'] = '127.0.1.1'
        os.environ['MASTER_PORT'] = '2222'

    
    def _train_variance_forward_seq(self, f, optimizer_object, logdir, batch_size = 10000, epochs=10, epoch_start=0,
                                logging=True, pretty_progressbar=True,  
                                    save_best=True, _run=None,n=0,log=True,mini_batch_size=20000,
                                **train_opts):
        """Train the model using the integrand variance as loss and compute the Jacobian in the forward pass
        (fixed latent space sample mapped to a phase space sample)
        

        Args:
            f ():
            batch_size ():
            epochs ():
            epoch_start ():
            logging ():
            pretty_progressbar ():
            optimizer_object ():
            logdir():
            **train_opts ():

        Returns: None

        """
        
        dev = torch.device("cuda:"+str(n)) if torch.cuda.is_available() else torch.device("cpu")
        
        assert mini_batch_size<=batch_size, "The minibatch size must be smaller than the batch size"
        n_minibatches = int(batch_size/mini_batch_size)
        
        if(_run!=None and log): 
            filename=logdir+"/"+str(_run._id)+"/torch"
            filename2=filename+"_int"
            if not os.path.exists(logdir+"/"+str(_run._id)):
                os.mkdir(logdir+"/"+str(_run._id))
   
        else:
            filename=logdir+"/torch"
            filename2=filename+"_int"
            if not os.path.exists(logdir):
                os.mkdir(logdir)
   
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
         
        #writer=SummaryWriter(log_dir=logdir)
        
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
        """
#######################################
        # Keep track of metric history if needed
        if logging:
            history = {}
        """
        i=0
        self.model.to(dev)
        self.best_loss=0
       
        self.best_var=0
        maxf=0
        #meanf=0
        while (i<self.n_flow):
            w = torch.empty(2*mini_batch_size, self.n_flow).to(torch.double)
       # 
        
            torch.nn.init.uniform_(w).to(dev)  # Generate a batch of points in latent space
       
       # Y=torch.tan((w-0.5)*(np.pi))
       # """
        # Run the model once
        #torch.nn.init.normal_(w, std=10)
        
            dkl=torch.nn.KLDivLoss(reduction='batchmean')
            if torch.max(f(w))>maxf:
                maxf=torch.max(f(w))
            #meanf_add=torch.mean(f(w))/maxf
            self.best_loss+=torch.var(f(w)/maxf).detach()
            
            i=i+1
            #meanf+=meanf_add
            
            self.best_var+=float((torch.var((f(w))**2)/2*mini_batch_size).detach())
        #meanf=meanf/(self.n_flow)
        
        #def g(x):
         #   return f(x)-meanf
            
        
        if save_best:
            
            
            #fz=f(X)
               
            XJ = self.model(  # Pass through the model
                self.format_input(  # Append a unit Jacobian to each point
                    w,dev
                )
              )
            X=XJ[:,:-1]
            J = XJ[:,-1]
            self.varJ=torch.mean(J**2).detach()
            #print(torch.var(w))
            #print(self.varJ)
            #print(torch.mean(J**2))
            #print(X)
            #print(torch.log(X))
            #print(w)
            #print(X)
            #self.DKL=dkl(torch.log(w).to(dev),X.to(dev)).detach()
            #print(dkl(torch.log(w).to(dev),XJ[:,:-1].to(dev)))
            #print(self.DKL)
            self.DKL=dkl(torch.log(X+torch.ones_like(X).fill_(1e-45)).to(dev),w.to(dev)).detach()
            #print(self.DKL)
            #print(self.DKL)
            self.best_loss=self.best_loss/self.n_flow
            
            
            
            #print("Int")
            #print(self.best_loss)
            self.best_model=copy.deepcopy(self.model)
           
            
            self.int_loss=self.best_loss
            #print(self.best_var)
            self.best_epoch=0
            self.best_time=0
            self.best_loss_rel=torch.ones_like(self.best_loss)
            self.func_count=1
            self.best_func_count=1
            del XJ,X,J
        if(_run!=None and log):
            _run.log_scalar("training.int_loss", self.best_loss.tolist(), 0)

       
           
    
        #torch.initial_seed()
        stale_save=1000
        preburner=True
      
        counter=0
        last_loss=1000
        for i in epoch_progress:
            loss = 0
            loss2=0
            loss3=0
            var = 0
            optimizer_object.zero_grad()
            
            for j in minibatch_progress:
               # if(i%3==0):
                

                #torch.nn.init.normal_(w, std=10)
                #"""
                w = torch.empty(mini_batch_size, self.n_flow).to(dev).to(torch.double)
                torch.nn.init.uniform_(w)
                #Y=150*torch.tan((w-0.5)*(np.pi))
                #"""
                #print("run")
                #w.requires_grad=True
                XJ = self.model(                                            # Pass through the model
                    self.format_input(                                      # Append a unit Jacobian to each point
                        #Y# Generate a batch of points in latent space
                        w,dev
                    )
                )

                #print(XJ)
                # Separate the points and their Jacobians:
                # This sample is fixed, we optimize the Jacobian
                X = (XJ[:, :-1]).detach()
                #Z=torch.atan(X/100)/(100*np.pi)+0.5


                #jacs=torch.mul(torch.abs(torch.prod(tanp(w),axis=-1)),torch.abs(torch.prod(atanp(X),axis=-1)))
                """
                fz=torch.mul(f(X).detach(),torch.abs(torch.prod(tanp(w),axis=-1)))
                """
                #fz=torch.div(f(X).detach(),normal(X,0,10,self.n_flow))
                #if(i<30 and 2*j>i):
               
                if(preburner):
                    fXJ=torch.mul(f(w), XJ[:, -1])/maxf
                    
                else:
                    fXJ = torch.mul(f(X), XJ[:, -1])/maxf
                    
                
                self.func_count=self.func_count+1
                
                loss+=torch.var(fXJ)
                
                # The Monte Carlo integrand is fXJ: we minimize its variance up to the constant term
                
                var+=float(torch.var(fXJ**2)/mini_batch_size) # variance of the mean is variance/N
                del X, fXJ, XJ
                
            
            loss=loss/n_minibatches
            
                
            
            
            loss.backward()
            
            optimizer_object.step()
            
            
            # Update the progress bar
            if pretty_progressbar:
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))
            if(_run!=None and log):       
                _run.log_scalar("training.loss", loss.tolist(), i)
                _run.log_scalar("training.loss_rel",(loss/self.int_loss).tolist(),i)
               

                
            """
            # Log the relevant data for internal use
            if logging:
                history[str(i)]={"loss":float(loss), "std":float(std)}

           
            writer.add_scalar('loss', float(loss),i)
            writer.add_scalar('std', float(std),i)
            """
            self.best_func_count=self.func_count*mini_batch_size
            if save_best and (loss < self.best_loss) and not preburner:
                self.best_loss = loss
                self.best_var=var
                self.best_loss_rel=loss/self.int_loss
                self.best_model=copy.deepcopy(self.model)
                self.best_epoch=i
                
                
                if(_run!=None):
                    self.best_time=(datetime.datetime.utcnow()-_run.start_time).total_seconds()
                else:
                    self.best_time=0
                
                    
            if loss<last_loss:
                counter=0
            else:
                counter+=1
                if counter>5 and preburner:
                    counter=0
                    preburner=False
                elif counter>5:
                    break
            last_loss=loss
            if i%50==0 and i>51 and float(self.best_loss/stale_save)>(1-1e-4) and not preburner:
                break
            elif i%50==0 and not preburner and (self.best_loss<self.int_loss or i>300):
                print(self.best_loss)
                print(stale_save)
                stale_save=self.best_loss 
            if _run!=None and self.best_time>300:
                break
           
        
            if preburner and ((loss<0.15*self.best_loss) or i>100):
                print("preburner finished")
                print(i)
                preburner=False
            """"""
            #if save_best and (loss > 1.75*self.best_loss*self.n_flow/2 or i-self.best_epoch>150) and i>pre_i+5:
            #    break   
           
            
            
        """
        writer.close()
        
        if logging:
            self.history=history
        """

        try:
            if (log):
                torch.save({
                    'best_epoch': self.best_epoch,
                    'best_loss': self.best_loss,
                    'int_loss': self.int_loss,
                    'best_loss_rel' : self.best_loss_rel,
                    'best_func_count' : self.best_func_count,
                    'model_state_dict': self.best_model.state_dict()
                    },filename)
        except:
            print("Torch save not possible")

        
        pass
    
  
            

    def _train_variance_forward(self,rank, f, logdir, lr=3e-4,reg=1e-7,batch_size = 10000, epochs=10, epoch_start=0,
                                logging=True, pretty_progressbar=True,  save_best=True, world_size=4, 
                                **train_opts):
        """Train the model using the integrand variance as loss and compute the Jacobian in the forward pass
        (fixed latent space sample mapped to a phase space sample)
        

        Args:
            f ():
            batch_size ():
            epochs ():
            epoch_start ():
            logging ():
            pretty_progressbar ():
            optimizer_object ():
            logdir():
            **train_opts ():

        Returns: history

        """
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename=logdir+"loss-model"+".txt"
        file= open(filename,"w+")
        file.close()
        optimizer_object = torch.optim.Adamax(self._model.parameters(),lr=lr, weight_decay=reg) 
        
        
        
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        
        w = torch.empty(batch_size, self.n_flow)
      
        torch.nn.init.uniform_(w).to(dev)  # Generate a batch of points in latent space
       
        self.model.to(dev)
        XJ = self.model(  # Pass through the model
            self.format_input(  # Append a unit Jacobian to each point
                w, dev
            )
          )
        
        
        
        if logging:
            X=XJ[:,:-1]
            #fz=torch.div(f(X),normal(X,0,10,self.n_flow))
            fz=f(X)
            J = XJ[:,-1]
            int_std = torch.std(fz*J)
            int_loss=torch.mean((fz*J)**2)
            best_model=self.model
            int_var=torch.var((fz*J)**2)
            best_std=int_std
            best_loss=int_loss
            best_var=int_var
            best_epoch=0
        if rank==0:
            writer=SummaryWriter(log_dir=logdir)
           
        
        
        if(dev==torch.device("cuda")):
            dist.init_process_group("nccl", rank=rank, init_method='env://', world_size=world_size)
        else:
            dist.init_process_group("gloo", rank=rank, init_method='env://', world_size=world_size)
         
        
        
        
        if dev == torch.device("cuda"):
           # n = torch.cuda.device_count()
            #if n>6:
            #    n=6
            #n=n//world_size
            device_ids = list(range(rank, (rank + 1)))
            self.model.to(device_ids[0])
            self.modelp=DDP(self.model, device_ids=device_ids)
            devp=torch.device("cuda:"+str(device_ids[0]))
        else:
            self.modelp=DDP(self.model, None)
            devp=dev
            
            
        # Instantiate a pretty progress bar if needed
        if pretty_progressbar:
            epoch_progress = tqdm(range(epoch_start,epoch_start+epochs), leave=False, desc="Loss: {0:.3e} | Epoch".format(0.))
        else:
            epoch_progress = range(epoch_start, epoch_start+epochs)
            
#######################################
        
            
            
        """
        shm[0]=0
        shm[1]=0
        shm[2]=0
        shm[3]=int_loss
        shm[4]=0
        shm[5]=0
        """
        #shm[6]=0
        torch.initial_seed()
       
        for i in epoch_progress:
          #  if(shm[6]==1):
           #     break
            """
            if(shm[4]==0):
                shm[0]=0
                shm[1]=0
                shm[2]=0
            """      
            loss = 0
            std = 0
            optimizer_object.zero_grad()

            
            w = torch.empty(batch_size, self.n_flow).to(devp)
            torch.nn.init.uniform_(w)

            XJ = self.modelp(                                            # Pass through the model
                self.format_input(                                      # Append a unit Jacobian to each point
                    w,devp
                )
            )


            # Separate the points and their Jacobians:
            # This sample is fixed, we optimize the Jacobian
            X = (XJ[:, :-1]).detach()
            



            fXJ = torch.mul(f(X), XJ[:, -1])

            # The Monte Carlo integrand is fXJ: we minimize its variance up to the constant term
            loss = torch.mean(fXJ**2)
            var=torch.var(fXJ**2)
            std= torch.std(fXJ)
            #shm[0]+=loss/world_size
            #shm[1]+=var/world_size
            #shm[2]+=std/world_size
            #shm[4]+=1
            

            loss.backward()

            optimizer_object.step()
            

            

            """
            if(shm[4]==world_size  and shm[0]>1e-20):
                shm[4]=0
                resloss=shm[0].clone()
                resstd=shm[2].clone()
            """   
            if logging and rank==0:
                writer.add_scalar('loss', float(loss),i)
                writer.add_scalar('std', float(std),i)

            # Update the progress bar
            if pretty_progressbar:
                #print(resloss)
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))
            #TODO: fix early stopping/saving of best model!
            """
                if logging and resloss < shm[3] :
                    shm[3]=resloss.clone()
                    best_std = shm[2].clone()
                    best_loss = resloss.clone()
                    best_var=shm[1].clone()
                    best_model=copy.deepcopy(self.modelp)
                    best_epoch=i
                    shm[5]=rank
                shm[0]=0
                shm[1]=0
                shm[2]=0
                #if(resloss>1.3*shm[3]): BUG!!
                #    shm[6]=1
         """ 
            
        if(rank==0):
            writer.close()


        if logging and rank==0:
            torch.save({
            #'best_epoch': best_epoch,
            'end_loss': loss,
            'end_var':var ,
            'int_loss': int_loss,
            'int_var':int_var,
            'model_state_dict': self.modelp.module.state_dict()
            },filename)


        dist.destroy_process_group()

        pass

      
    
    
    
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
    

    def create_model(self,*,
                     n_pass_through,
                     n_cells,
                     NN,
                     roll_step,
                     **opts
                     ):
        """

        Args:
            n_pass_through ():
            n_cells ():
            NN():
            roll_step ():
            **opts ():

        Returns:

        """
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(42)
        
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
    

    def create_model(self,*,
                     n_pass_through,
                     n_cells,
                     n_bins,
                     NN,
                     roll_step,
                     **opts
                     ):
        """

        Args:
            n_pass_through ():
            n_cells ():
            n_bins:
            NN():
            roll_step ():
            **opts ():

        Returns:

        """
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        #torch.manual_seed(42)
        
        self._model = torch.nn.Sequential()
        
        
        for i_cell in range(n_cells):#create coupling cells
            self._model.add_module(str(i_cell),
                PWLin(flow_size=self.n_flow, pass_through_size=n_pass_through,n_bins=n_bins,
                                                NN_layers=NN)
            )
            #self._model.add_module("batch", BatchLayer(self.n_flow))
            self._model.add_module("roll",RollLayer(roll_step)) #add roll layer
        self._model.to(dev)
        self.best_model=self.model
       
        w = torch.empty(5, self.n_flow).to(torch.double)
        torch.nn.init.uniform_(w).to(dev)
        # Do one pass forward:
        self._model(self.format_input(w,dev))   
        
        
class PWQuadManager(BasicManager):
    
    """A manager for normalizing flows with piecewise-quadratic coupling cells interleaved with rolling layers that
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
    

    def create_model(self,*,
                     #n_pass_through,
                     n_cells,
                     n_bins,
                     NN,
                     dev,
                     #roll_step,
                     **opts
                     ):
        """

        Args:
            n_cells ():
            n_bins:
            NN():
            **opts ():

        Returns:

        """
        if(n_cells<2*np.ceil(np.log2(self.n_flow)) and n_cells<self.n_flow):
            print("WARNING: Too few coupling cells!")
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
        #print(w.shape)
        torch.nn.init.uniform_(w).to(dev)
        ####
        """
        w=torch.arange(1.,self.n_flow+1)
        print(w.shape)
        w=torch.unsqueeze(w,0)
        print(w.shape)
        w=torch.cat((w,w),0)
        print(w)
        ##
        """
        # Do one pass forward:
        self._model(self.format_input(w,dev)) 
        
    

    
