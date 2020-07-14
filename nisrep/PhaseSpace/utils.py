import torch
import math

def set_square_t(inputt, square, negative=False):
        """Change the time component of this LorentzVector
        in such a way that self.square() = square.
        If negative is True, set the time component to be negative,
        else assume it is positive.
        """
        
        ret=torch.zeros_like(inputt)
       
        ret[:,0] = (rho2_t(inputt) + square) ** 0.5
        
        if negative: ret[:,0] *= -1
        ret[:,1:]=inputt[:,1:]
       
        return ret
    
def rho2_t(inputt):
        """Compute the radius squared."""
        
        return torch.sum(inputt[:,1:]*inputt[:,1:],-1)
    
def boostVector_t(inputt):
        
        if torch.min(inputt[:,0]) <= 0. or torch.min(square_t(inputt)) < 0.:
            logger.critical("Attempting to compute a boost vector from")
            logger.critical("%s (%.9e)" % (str(self), self.square()))
            raise InvalidOperation
        
        return inputt[:,1:]/inputt[:,0].unsqueeze(1)
    
def square_t(inputt):
    if(inputt.shape[1]==4 or inputt.shape[0]==4):
        
        return dot_t(inputt,inputt)
    else:
        
        return torch.sum(inputt*inputt,-1)
    
def dot_t(inputa,inputb):
    return inputa[:,0]*inputb[:,0] - inputa[:,1]*inputb[:,1] - inputa[:,2]*inputb[:,2] - inputa[:,3]*inputb[:,3]
    
def boost_t(inputt, boost_vector, gamma=-1.):
        """Transport self into the rest frame of the boost_vector in argument.
        This means that the following command, for any vector p=(E, px, py, pz)
            p.boost(-p.boostVector())
        transforms p to (M,0,0,0).
        """
        
        b2 = square_t(boost_vector)
        if gamma < 0.:
            gamma = 1.0 / torch.sqrt(1.0 - b2)
        inputt_space = inputt[:,1:]
       
        bp = torch.sum(inputt_space*boost_vector,-1)
        
        gamma2=torch.where(b2>0, (gamma-1.0)/b2,torch.zeros_like(b2))
        
        factor = gamma2*bp + gamma*inputt[:,0]
        
        inputt_space+= factor.unsqueeze(1)*boost_vector
        
        inputt[:,0] = gamma*(inputt[:,0] + bp)
        
        return inputt
    
def cosTheta_t(inputt):

        ptot =torch.sqrt(torch.dot(inputt[1:],inputt[1:]))
        assert (ptot > 0.)
        return inputt[3] / ptot

    

def phi_t(inputt):

    return torch.atan2(inputt[2], inputt[1])