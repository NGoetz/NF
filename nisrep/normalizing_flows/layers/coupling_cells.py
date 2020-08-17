import torch
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
        NN_layers.append(torch.nn.BatchNorm1d(sizes[0]))
        
        NN_layers.append(torch.nn.Linear(pass_through_size, sizes[0])) #size only one dim
       
        NN_layers.append(torch.nn.BatchNorm1d(sizes[0]))
       
        NN_layers.append(torch.nn.ReLU())
        oldsize=sizes[0]
        
        for size in sizes[1:-1]:
            NN_layers.append(torch.nn.Linear(oldsize,size))
            NN_layers.append(torch.nn.BatchNorm1d(size))
            
            NN_layers.append(torch.nn.ReLU())
            oldsize=size
       
         
        NN_layers.append((torch.nn.Linear(oldsize,sizes[-1])))
       
        
        NN_layers.append(Reshape(2, self.transform_size))
        #we construct a Sequential module from our NN
        self.NN = torch.nn.Sequential(*NN_layers)
        

    def forward(self, x):
        xA = x[:, :self.pass_through_size]
        xB = x[:, self.pass_through_size:self.flow_size] 
        
        shift_rescale = self.NN(xA) #The NN is evaluated on the pass-through dimensons
        z=torch.ones_like(shift_rescale[:,0])
        s0=shift_rescale[:,0]
       
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
    
class PWQuad(torch.nn.Module):
   
    def __init__(self,flow_size, pass_through_size, n_bins,NN_layers, run=None):
        super(PWQuad, self).__init__()
        
        self.pass_through_size = pass_through_size
        self.flow_size = flow_size
        self.transform_size = flow_size - pass_through_size
        self.n_bins=n_bins
        self.run=run
        
        #the last layer has the bins as output
        sizes = NN_layers + [( self.transform_size*(2*self.n_bins+1))]
        self.NN=RectNN(pass_through_size,sizes,(self.transform_size,(2*self.n_bins+1))).NN
        
    def forward(self, x): 
        
        
        xA = x[:, :self.pass_through_size]
        xB = x[:, self.pass_through_size:self.flow_size]
        xB=torch.where(xB>(torch.ones_like(xB)-1e-6),(torch.ones_like(xB)-1e-6),xB)
        
        jacobian = torch.unsqueeze(x[:, self.flow_size], -1)
        
        Z=self.NN(xA)
       
        V=Z[:,:,:self.n_bins+1]
        
        W=Z[:,:,self.n_bins+1:]
        
        dev=W.device
        W=torch.exp(W)
        
        Wsum = torch.cumsum(W, axis=-1) 
        
        Wnorms = torch.unsqueeze(Wsum[:, :, -1], axis=-1) 
        
        W = W/Wnorms
        
        Wsum=Wsum/Wnorms
        
        
        V=torch.exp(V)
        
        Vsum=torch.cumsum(V, axis=-1)
        
        
        Vnorms=torch.cumsum(torch.mul((V[:,:,:-1]+V[:,:,1:])/2,W),axis=-1)
        
        Vnorms_tot=Vnorms[:, :, -1].clone() 
        V=torch.div(V,torch.unsqueeze(Vnorms_tot,axis=-1)) 
        Wsum2=torch.cat((torch.zeros([Wsum.shape[0],Wsum.shape[1],1]).to(dev).to(torch.double),Wsum),axis=-1)
        finder=torch.where(Wsum>torch.unsqueeze(xB,axis=-1),torch.zeros_like(Wsum),torch.ones_like(Wsum))
        
        div_ind=torch.unsqueeze(torch.argmax(torch.cat((torch.empty(Wsum.shape[0],Wsum.shape[1],1)
                                                        .fill_(1e-30).to(dev).to(torch.double),finder*Wsum),axis=-1),axis=-1),-1)
        
       
        
        alphas=torch.div((xB-torch.squeeze(torch.gather(Wsum2,-1,div_ind),axis=-1)),
                         torch.squeeze(torch.gather(W,-1,div_ind),axis=-1))
        
        VW=torch.cat((torch.zeros([V.shape[0],V.shape[1],1]).to(dev).to(torch.double),
                                  torch.cumsum(torch.mul((V[:,:,:-1]+V[:,:,1:])/2,W),axis=-1)),axis=-1)
        shift= torch.squeeze(torch.gather(VW,-1,div_ind),axis=-1)
        
        yB1=torch.mul((alphas**2)/2,torch.squeeze(torch.mul(torch.gather(V,-1, div_ind+1)-torch.gather(V,-1, div_ind),
                                                            torch.gather(W,-1,div_ind)),axis=-1))
        
        yB2=torch.mul(torch.mul(alphas,torch.squeeze(torch.gather(V,-1,div_ind),axis=-1)),
                                torch.squeeze(torch.gather(W,-1,div_ind),axis=-1))
       
       
        yB=yB1+yB2+shift
        
        
        jacobian=jacobian*torch.unsqueeze(torch.prod(torch.lerp(torch.squeeze(torch.gather(V,-1,div_ind),axis=-1),
                                                torch.squeeze(torch.gather(V,-1,div_ind+1),axis=-1),alphas), axis=-1),axis=-1)
       
        return torch.cat((xA, yB, jacobian), axis=-1) 
        
class RectNN(torch.nn.Module):
    #constructs a rectangular NN
    def __init__(self,pass_through_size,sizes,reshape):
        super(RectNN, self).__init__()
        NN_layers=[]
       
        NN_layers.append(torch.nn.BatchNorm1d(pass_through_size))
        NN_layers.append(torch.nn.Linear(pass_through_size, sizes[0],bias=False))
        NN_layers.append(torch.nn.BatchNorm1d(sizes[0]))
        NN_layers.append(torch.nn.ReLU())
        oldsize=sizes[0]
        
        for size in sizes[1:-1]:
            NN_layers.append(torch.nn.Linear(oldsize,size,bias=False))
            NN_layers.append(torch.nn.BatchNorm1d(size))
            NN_layers.append(torch.nn.ReLU())
            oldsize=size
       
        
        NN_layers.append((torch.nn.Linear(oldsize,sizes[-1])))
        
        
        NN_layers.append(Reshape(reshape[0], reshape[1]))
        #we construct a Sequential module from our NN
        self.NN = torch.nn.Sequential(*NN_layers)

    
