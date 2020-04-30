import torch
import torch.nn.utils.weight_norm as weightNorm
import numpy as np
from .layers import Reshape


class AffineCoupling(torch.nn.Module):
    """
        Affine coupling cell
        Given the dimension of the input, the number of dimensions which should
        pass through without being transformed and a list of linear NN layers with
        their respective width, a fully connected NN with ReLU activation is constructed
    """
    def __init__(self,flow_size, pass_through_size, NN_layers):
        super(AffineCoupling, self).__init__()
        
        self.pass_through_size = pass_through_size
        self.flow_size = flow_size
        self.transform_size = flow_size - pass_through_size
        
        #the last layer has 2 outputs: scaling factor and translation
        sizes = NN_layers + [(2 * self.transform_size)] 
        NN_layers=[]
        NN_layers.append(torch.nn.Tanh())
        
        NN_layers.append(torch.nn.Linear(pass_through_size, sizes[0])) #size only one dim
       
        #NN_layers.append(torch.nn.BatchNorm1d(sizes[0]))
       
        NN_layers.append(torch.nn.ReLU())
        oldsize=sizes[0]
        
        for size in sizes[1:-1]:
            NN_layers.append(torch.nn.Linear(oldsize,size))
            #NN_layers.append(torch.nn.BatchNorm1d(size))
            
            NN_layers.append(torch.nn.ReLU())
            oldsize=size
       
         
        NN_layers.append((torch.nn.Linear(oldsize,sizes[-1])))
        #NN_layers.append(torch.nn.BatchNorm1d(sizes[-1]))
       
       # NN_layers.append(torch.nn.Sigmoid())
        
        NN_layers.append(Reshape(2, self.transform_size))
        #we construct a Sequential module from our NN
        self.NN = torch.nn.Sequential(*NN_layers)
        

    def forward(self, x):
        xA = x[:, :self.pass_through_size]
        xB = x[:, self.pass_through_size:self.flow_size] 
        
        shift_rescale = self.NN(xA) #The NN is evaluated on the pass-through dimensons
        z=torch.ones_like(shift_rescale[:,0])
        s0=shift_rescale[:,0]
        #s1=50*torch.tan(np.pi*(shift_rescale[:,1]-0.5))
        s1=8*shift_rescale[:,1]-4
        yB = torch.mul(xB, s0) + s1 #xB is transformed. 
        
        jacobian = x[:, self.flow_size] #the old jacobian is saved in the last row of the data...
        
        jacobian = jacobian*torch.prod(s0,1) #... and updated by the forward transformation
       
        return torch.cat((xA, yB, torch.unsqueeze(jacobian, 1)), dim=1)  #everything is packed in one tensor again


class PWLin(torch.nn.Module):
   
    def __init__(self,flow_size, pass_through_size, n_bins,NN_layers):
        super(PWLin, self).__init__()
        
        self.pass_through_size = pass_through_size
        self.flow_size = flow_size
        self.transform_size = flow_size - pass_through_size
        self.n_bins=n_bins
        
        #the last layer has the bins as output
        sizes = NN_layers + [( self.transform_size*self.n_bins)] 
        NN_layers=[]
        NN_layers.append(torch.nn.BatchNorm1d(pass_through_size))
        NN_layers.append(torch.nn.Linear(pass_through_size, sizes[0],bias=False))#size only one dim
        NN_layers.append(torch.nn.BatchNorm1d(sizes[0]))
        NN_layers.append(torch.nn.ReLU())
        oldsize=sizes[0]
        
        for size in sizes[1:-1]:
            NN_layers.append(torch.nn.Linear(oldsize,size,bias=False))
            NN_layers.append(torch.nn.BatchNorm1d(size))
            NN_layers.append(torch.nn.ReLU())
            oldsize=size
       
        
        NN_layers.append((torch.nn.Linear(oldsize,sizes[-1])))
        
        
        NN_layers.append(Reshape(self.transform_size, self.n_bins))
        #we construct a Sequential module from our NN
        self.NN = torch.nn.Sequential(*NN_layers)
        

    def forward(self, x):
        
        
        xA = x[:, :self.pass_through_size]
        xB = x[:, self.pass_through_size:self.flow_size]
        
        jacobian = torch.unsqueeze(x[:, self.flow_size], -1)
        Q = self.NN(xA)
        Q=torch.exp(Q)
        
        Qsum = torch.cumsum(Q, axis=-1) 
        
        Qnorms = torch.unsqueeze(Qsum[:, :, -1], axis=-1) 
        
        Q = Q/(Qnorms / self.n_bins) 
        
        Qsum = Qsum/Qnorms 
        Qsum=torch.cat((torch.unsqueeze(torch.zeros_like(Qsum[:,:,-1]),-1),Qsum),axis=-1)
        
        alphas = xB * self.n_bins   
       
        bins = torch.floor(alphas) 
        
        alphas = alphas-bins  
        alphas = alphas/self.n_bins 
        
        bins = bins.long()
        
        # Sum of the integrals of the bins
        cdf_int_part = torch.gather(Qsum, -1,torch.unsqueeze(bins, axis=-1)) 
        cdf_float_part = torch.gather(Q, -1, torch.unsqueeze(bins, axis=-1)) 
       
        cdf = torch.reshape((cdf_float_part * torch.unsqueeze(alphas, axis=-1)) + cdf_int_part, cdf_int_part.shape[:-1]) 
      
        jacobian = jacobian*torch.prod(cdf_float_part, axis=-2) 
        return torch.cat((xA, cdf, jacobian), axis=-1) 
        
    

    
