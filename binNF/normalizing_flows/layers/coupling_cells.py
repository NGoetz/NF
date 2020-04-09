import torch
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
        NN_layers.append(torch.nn.Linear(pass_through_size, sizes[0]))#size only one dim
        NN_layers.append(torch.nn.ReLU())
        oldsize=sizes[0]
        
        for size in sizes[1:-1]:
            NN_layers.append(torch.nn.Linear(oldsize,size))
            NN_layers.append(torch.nn.ReLU())
            oldsize=size
       
        # Last layer will be exponentiated so it is regularised with Sigmoid
        NN_layers.append((torch.nn.Linear(oldsize,sizes[-1])))
        NN_layers.append(torch.nn.Sigmoid())
        NN_layers.append(Reshape(2, self.transform_size))
        #we construct a Sequential module from our NN
        self.NN = torch.nn.Sequential(*NN_layers)
        

    def forward(self, x):
        xA = x[:, :self.pass_through_size]
        xB = x[:, self.pass_through_size:self.flow_size]  
        shift_rescale = self.NN(xA) #The NN is evaluated on the pass-through dimensons
        
       
        shift_rescale[:, 1] = torch.exp((shift_rescale[:, 1])) 
        yB = torch.mul(xB, shift_rescale[:, 1]) + shift_rescale[:, 0] #xB is transformed. The translation is regularised
        jacobian = x[:, self.flow_size] #the old jacobian is saved in the last row of the data...
       
        jacobian = torch.mul(jacobian,torch.prod(shift_rescale[:, 1], 1)) #... and updated by the forward transformation
        
        return torch.cat((xA, yB, torch.unsqueeze(jacobian, 1)), dim=1)  #everything is packed in one tensor again
    
