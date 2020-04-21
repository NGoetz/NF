import torch as torch
import copy
import os
import yaml
import numpy as np
from tensorboard.plugins.hparams import api as hp
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from .layers.coupling_cells import  AffineCoupling
from .layers.layers import AddJacobian, RollLayer
from statistics import mean

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
        

    def _train_variance_forward(self, f, batch_size = 10000, epochs=10, epoch_start=0,
                                logging=True, pretty_progressbar=True,  save_best=True,
                                *, optimizer_object, logdir, **train_opts):
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
        self.optimizer_object = optimizer_object
        writer=SummaryWriter(log_dir=logdir)
        

        # Instantiate a pretty progress bar if needed
        if pretty_progressbar:
            epoch_progress = tqdm(range(epoch_start,epoch_start+epochs), leave=False, desc="Loss: {0:.3e} | Epoch".format(0.))
        else:
            epoch_progress = range(epoch_start, epoch_start+epochs)
            
#######################################
        # Keep track of metric history if needed
        if logging:
            history = {}
        w = torch.empty(batch_size, self.n_flow)
        """
        torch.nn.init.uniform_(w)  # Generate a batch of points in latent space
        Y=torch.tan((w-0.5)*(np.pi))
        """
        # Run the model once
        torch.nn.init.normal_(w, std=10)
        
        XJ = self.model(  # Pass through the model
            self.format_input(  # Append a unit Jacobian to each point
                w
            )
          )
       
        if save_best:
            X=XJ[:,:-1]
            fz=torch.div(f(X),normal(X,0,10,self.n_flow))
            J = XJ[:,-1]
            best_std = torch.std(fz*J)
            best_loss=torch.mean((fz*J)**2)
    
        
        for i in epoch_progress:
            loss = 0
            std = 0
            optimizer_object.zero_grad()
            
            #if(i%5==0):
            w = torch.empty(batch_size, self.n_flow)
            
            torch.nn.init.normal_(w, std=10)
            """
            torch.nn.init.uniform_(w)
            Y=150*torch.tan((w-0.5)*(np.pi))
            """
            
            XJ = self.model(                                            # Pass through the model
                self.format_input(                                      # Append a unit Jacobian to each point
                    #Y# Generate a batch of points in latent space
                    w
                )
            )

            
            # Separate the points and their Jacobians:
            # This sample is fixed, we optimize the Jacobian
            X = (XJ[:, :-1]).detach()
            #Z=torch.atan(X/100)/(100*np.pi)+0.5
           
            
            #jacs=torch.mul(torch.abs(torch.prod(tanp(w),axis=-1)),torch.abs(torch.prod(atanp(X),axis=-1)))
            """
            fz=torch.mul(f(X).detach(),torch.abs(torch.prod(tanp(w),axis=-1)))
            """
            fz=torch.div(f(X).detach(),normal(X,0,10,self.n_flow))
            
            fXJ = torch.mul(fz, XJ[:, -1])
            
            # The Monte Carlo integrand is fXJ: we minimize its variance up to the constant term
            loss = torch.mean(fXJ**2)
            var=torch.var(fXJ**2)
           
            loss.backward()
            
                
            optimizer_object.step()
            std= torch.std(fXJ)
            
            # Update the progress bar
            if pretty_progressbar:
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))

            # Log the relevant data for internal use
            if logging:
                history[str(i)]={"loss":float(loss), "std":float(std)}

           
            writer.add_scalar('loss', float(loss),i)
            writer.add_scalar('std', float(std),i)
            
            if save_best and loss < best_loss:
                best_std = std
                best_loss = loss
                self.best_model=copy.deepcopy(self.model)
             
            """
            if save_best and loss > 2*best_loss:
                break
            """
        
        writer.close()
        
        if logging:
            return history

      
    
    
    
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
        
        self._model = torch.nn.Sequential()
        
        for i_cell in range(n_cells):#create coupling cells
            self._model.add_module(str(i_cell),
                AffineCoupling(flow_size=self.n_flow, pass_through_size=n_pass_through,
                                                NN_layers=NN)
            )
            self._model.add_module("roll",RollLayer(roll_step)) #add roll layer
        
       
        w = torch.empty(1, self.n_flow)
        torch.nn.init.uniform_(w)
        # Do one pass forward:
        self._model(self.format_input(w))    
    
