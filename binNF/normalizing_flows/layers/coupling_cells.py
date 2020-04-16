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
        sizes = NN_layers + [(1 * self.transform_size)] 
        NN_layers=[]
        NN_layers.append(torch.nn.Tanh())
        NN_layers.append(torch.nn.Linear(pass_through_size, sizes[0]))#size only one dim
        NN_layers.append(torch.nn.ReLU())
        oldsize=sizes[0]
        
        for size in sizes[1:-1]:
            NN_layers.append(torch.nn.Linear(oldsize,size))
            NN_layers.append(torch.nn.ReLU())
            oldsize=size
       
       # NN_layers.append(torch.nn.Tanh())  
        NN_layers.append((torch.nn.Linear(oldsize,sizes[-1])))
        NN_layers.append(torch.nn.Sigmoid())
        
        NN_layers.append(Reshape(1, self.transform_size))
        #we construct a Sequential module from our NN
        self.NN = torch.nn.Sequential(*NN_layers)
        

    def forward(self, x):
        xA = x[:, :self.pass_through_size]
        xB = x[:, self.pass_through_size:self.flow_size]  
        shift_rescale = self.NN(xA) #The NN is evaluated on the pass-through dimensons
        #shift_rescale[:, 0] = torch.exp(((shift_rescale[:, 0]))) 
        
        #print(shift_rescale[:,1])
        yB = torch.mul(xB, shift_rescale[:, 0])# + shift_rescale[:, 1] #xB is transformed. The translation is regularised
        
        jacobian = x[:, self.flow_size] #the old jacobian is saved in the last row of the data...
        
        jacobian = jacobian*torch.prod(shift_rescale[:, 0],1) #... and updated by the forward transformation
       
        return torch.cat((xA, yB, torch.unsqueeze(jacobian, 1)), dim=1)  #everything is packed in one tensor again
""" 

class PWLin(torch.nn.Module):
   
    def __init__(self,flow_size, pass_through_size, NN_layers):
        super(PWLin, self).__init__()
        
        self.pass_through_size = pass_through_size
        self.flow_size = flow_size
        self.transform_size = flow_size - pass_through_size
        self.bin_size=10
        
        #the last layer has 2 outputs: scaling factor and translation
        sizes = NN_layers + [(2 * self.transform_size)] 
        NN_layers=[]
        NN_layers.append(torch.nn.Linear(pass_through_size, sizes[0]))#size only one dim
        NN_layers.append(torch.nn.ReLU())
        oldsize=sizes[0]
        
        for size in sizes[1:-1]:
            NN_layers.append(torch.nn.Linear(oldsize,size))
            NN_layers.append(torch.nn.ReLU())
            oldsize=size
       
       
        NN_layers.append((torch.nn.Linear(oldsize,sizes[-1])))
        NN_layers.append(torch.nn.Softmax())
        
        NN_layers.append(Reshape(self.transform_size, self.bin_size))
        #we construct a Sequential module from our NN
        self.NN = torch.nn.Sequential(*NN_layers)
        

    def forward(self, x):
        xA = x[:, :self.pass_through_size]
        xB = x[:, self.pass_through_size:self.flow_size]
        
        jacobian = tf.unsqueeze(x[:, self.flow_size], -1)
        Q = self.NN(xA)
        Qsum = torch.cumsum(Q, axis=-1)
        Qnorms = torch.unsqueeze(Qsum[:, :, -1], axis=-1)
        Q /= Qnorms / self.n_bins
        Qsum /= Qnorms
        ###
        Qsum = tf.pad(Qsum, tf.constant([[0, 0], [0, 0], [1, 0]]))
        alphas = xB * self.n_bins
        bins = tf.math.floor(alphas)
        alphas -= bins
        alphas /= self.n_bins
        bins = tf.cast(bins, tf.int32)
        # Sum of the integrals of the bins
        cdf_int_part = tf.gather(Qsum, tf.expand_dims(bins, axis=-1), batch_dims=-1, axis=-1)
        cdf_float_part = tf.gather(Q, tf.expand_dims(bins, axis=-1), batch_dims=-1, axis=-1)
        cdf = tf.reshape((cdf_float_part * tf.expand_dims(alphas, axis=-1)) + cdf_int_part, cdf_int_part.shape[:-1])
        jacobian *= tf.reduce_prod(cdf_float_part, axis=-2)
        return tf.concat((xA, cdf, jacobian), axis=-1)
        ####
       
        shift_rescale = self.NN(xA) #The NN is evaluated on the pass-through dimensons
        shift_rescale[:, 0] = torch.exp(((shift_rescale[:, 0]))) 
        
        #print(shift_rescale[:,1])
        yB = torch.mul(xB, shift_rescale[:, 0]) + shift_rescale[:, 1] #xB is transformed. The translation is regularised
        
        jacobian = x[:, self.flow_size] #the old jacobian is saved in the last row of the data...
        
        jacobian = jacobian*torch.prod(shift_rescale[:, 0],1) #... and updated by the forward transformation
       
        return torch.cat((xA, yB, torch.unsqueeze(jacobian, 1)), dim=1)  #everything is packed in one tensor again
"""      

    
