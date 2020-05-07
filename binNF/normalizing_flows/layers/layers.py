import torch
import numpy as np


def tanp(input):
    return (1+((torch.tan((input-0.5)*np.pi))**2))*np.pi   #derivative for jacobian    




class Reshape(torch.nn.Module):
    """
        Reshaping layer
    """
    def __init__(self, *shapes: int):
        super().__init__()
        self.shapes: typing.Tuple[int] = shapes

    def forward(self, tensor):
        return torch.reshape(tensor.clone(), (tensor.shape[0], *self.shapes))

class AddJacobian(torch.nn.Module):
    """
        Class adding a row for the Jacobian values
    """
    def __init__(self, jacobian_value=torch.ones(1)):
        super(AddJacobian, self).__init__()
        self.jacobian_value = jacobian_value   

    
    def forward(self, input, dev):
        
        return torch.cat((input.to(dev), self.jacobian_value.expand(input.shape[0],1).to(dev)), dim=1)


class RollLayer(torch.nn.Module):
    """
        Layer which shifts the dimensions for performing the coupling permutations
        on different dimensions
    """
    def __init__(self, shift):
        super(RollLayer, self).__init__()
        self.shift = shift

   
    def forward(self, x):
        return torch.cat((torch.roll(x[:, :-1], self.shift, dims=-1), x[:, -1:]), axis=-1)
    
class BatchLayer(torch.nn.Module):
    """
        Layer which performs batch normalization on the coupling cell output
    """
    def __init__(self, n_flow):
        super(BatchLayer, self).__init__()
        self.batch=torch.nn.BatchNorm1d(n_flow)
        

   
    def forward(self, x):
        
        y1=self.batch(x[:,:-1])
        var=torch.var(x[:,:-1],1)
        derv=torch.prod(self.batch.running_mean)
        
        for i in range(x.shape[1]-1):
            derv=torch.div(derv,torch.sqrt(var[i]+1e-05))
        y2=x[:,-1]*derv
        y2=torch.unsqueeze(y2, axis=-1)
        print(torch.cat((y1,y2), axis=-1))
        return torch.cat((y1,y2), axis=-1)


