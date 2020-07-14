import torch
import numpy as np



def tanp(input):
    return (1+((torch.tan((input-0.5)*np.pi))**2))*np.pi   #derivative for jacobian    


class MaskLayer(torch.nn.Module):
    """
        Masking layer
    """
    def __init__(self, dims_bin, pos,dev):
        super().__init__()
     
        
        
        feed=pos%2
        trafo=(pos+1)%2
        
        pos=int(np.floor(pos/2))
        
        masker=dims_bin[:,pos]
        
        self.feeder=(masker == feed).nonzero().to(dev)
        self.trafoer=(masker == trafo).nonzero().to(dev)
        self.pass_through=self.feeder.shape[0]
       

    def forward(self, tensor):
     
        return torch.cat((torch.index_select(tensor, -1, self.feeder.view(-1)),
                          torch.index_select(tensor, -1, self.trafoer.view(-1)),
                          tensor[:,-1:]),-1)
    
class DeMaskLayer(torch.nn.Module):
    """
        DeMasking layer
    """
    def __init__(self, first, second):
        super().__init__()
        self.list_ind=torch.unsqueeze(torch.squeeze(torch.cat((first,second),0),-1),0)
        

    def forward(self, tensor):
     
        t=torch.ones_like(tensor[:,:-1],dtype=torch.int16)
        lister=self.list_ind*t
        ret=torch.empty_like(tensor[:,:-1])
        
        ret[:,torch.squeeze(self.list_ind,0)]=tensor[:,:-1]
        
        return torch.cat((ret,tensor[:,-1:]),-1)



class Reshape(torch.nn.Module):
    """
        Reshaping layer
    """
    def __init__(self, shapes1, shapes2):
        super(Reshape, self).__init__()
        self.shapes=(shapes1,shapes2)

    def forward(self, tensor):
        return torch.reshape(tensor.clone(), (tensor.shape[0], self.shapes[0], self.shapes[1])) 
    
class AddJacobian(torch.nn.Module):
    """
        Class adding a row for the Jacobian values
    """
    def __init__(self, jacobian_value=torch.ones(1)):
        super(AddJacobian, self).__init__()
        self.jacobian_value = jacobian_value   

    
    def forward(self, input, dev):
        
        return torch.cat((input.to(dev), self.jacobian_value.expand(input.shape[0],1).to(dev).to(torch.double)), dim=1)


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
        return torch.cat((y1,y2), axis=-1)


