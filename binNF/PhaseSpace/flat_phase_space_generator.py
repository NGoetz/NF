import sys
import os


import torch
import logging
import math
import copy
import datetime

logger = logging.getLogger('PhaseSpace')


def set_square_t(inputt, square, negative=False):
        """Change the time component of this LorentzVector
        in such a way that self.square() = square.
        If negative is True, set the time component to be negative,
        else assume it is positive.
        """
        ret=torch.zeros_like(inputt)
        ret[0] = (rho2_t(inputt) + square) ** 0.5
        if negative: ret[0] *= -1
        ret[1:]=inputt[1:]
        return ret
    
def rho2_t(inputt):
        """Compute the radius squared."""

        return torch.sum(inputt[1:]**2,-1)
    
def boostVector_t(inputt):

        if inputt[0] <= 0. or square_t(inputt) < 0.:
            logger.critical("Attempting to compute a boost vector from")
            logger.critical("%s (%.9e)" % (str(self), self.square()))
            raise InvalidOperation
        return inputt[1:]/inputt[0]
    
def square_t(inputt):
    if(inputt.shape[0]==4):
        return dot_t(inputt,inputt)
    else:
        return torch.dot(inputt,inputt)
    
def dot_t(inputa,inputb):
    return inputa[0]*inputb[0] - inputa[1]*inputb[1] - inputa[2]*inputb[2] - inputa[3]*inputb[3]
    
def boost_t(inputt, boost_vector, gamma=-1.):
        """Transport self into the rest frame of the boost_vector in argument.
        This means that the following command, for any vector p=(E, px, py, pz)
            p.boost(-p.boostVector())
        transforms p to (M,0,0,0).
        """
        
        b2 = square_t(boost_vector)
        if gamma < 0.:
            gamma = 1.0 / torch.sqrt(1.0 - b2)
        inputt_space = inputt[1:]
        bp = torch.dot(inputt_space,boost_vector)
        gamma2 = (gamma-1.0) / b2 if b2 > 0 else 0.
        factor = gamma2*bp + gamma*inputt[0]
       
        inputt_space+= factor*boost_vector
        inputt[0] = gamma*(inputt[0] + bp)
        return inputt
    
def cosTheta_t(inputt):

        ptot =torch.sqrt(torch.dot(inputt[1:],inputt[1:]))
        assert (ptot > 0.)
        return inputt[3] / ptot

    

def phi_t(inputt):

    return torch.atan2(inputt[2], inputt[1])

class Dimension(object):
    """ A dimension object specifying a specific integration dimension."""
    
    def __init__(self, name, folded=False):
        self.name   = name
        self.folded = folded
    
    def length(self):
        raise NotImplemented
    
    def random_sample(self):        
        raise NotImplemented

class DiscreteDimension(Dimension):
    """ A dimension object specifying a specific discrete integration dimension."""
    
    def __init__(self, name, values, **opts):
        try:
            self.normalized = opts.pop('normalized')
        except:
            self.normalized = False
        super(DiscreteDimension, self).__init__(name, **opts)
        assert(isinstance(values, list))
        self.values = values
    
    def length(self):
        if normalized:
            return 1.0/float(len(values))
        else:
            return 1.0
    
    def random_sample(self):
        return np.int64(random.choice(values))
        
class ContinuousDimension(Dimension):
    """ A dimension object specifying a specific discrete integration dimension."""
    
    def __init__(self, name, lower_bound=0.0, upper_bound=1.0, **opts):
        super(ContinuousDimension, self).__init__(name, **opts)
        assert(upper_bound>lower_bound)
        self.lower_bound  = lower_bound
        self.upper_bound  = upper_bound 

    def length(self):
        return (self.upper_bound-self.lower_bound)

    def random_sample(self):
        return np.float64(self.lower_bound+random.random()*(self.upper_bound-self.lower_bound))

class DimensionList(list):
    """A DimensionList."""

    def __init__(self, *args, **opts):
        super(DimensionList, self).__init__(*args, **opts)

    def volume(self):
        """ Returns the volue of the complete list of dimensions."""
        vol = 1.0
        for d in self:
            vol *= d.length()
        return vol
    
    def append(self, arg, **opts):
        """ Type-checking. """
        assert(isinstance(arg, Dimension))
        super(DimensionList, self).append(arg, **opts)
        
    def get_discrete_dimensions(self):
        """ Access all discrete dimensions. """
        return DimensionList(d for d in self if isinstance(d, DiscreteDimension))
    
    def get_continuous_dimensions(self):
        """ Access all discrete dimensions. """
        return DimensionList(d for d in self if isinstance(d, ContinuousDimension))
    
    def random_sample(self):
        return np.array([d.random_sample() for d in self])


#=========================================================================================
# Phase space generation
#=========================================================================================

class VirtualPhaseSpaceGenerator(object):

    def __init__(self, initial_masses, final_masses,
                 beam_Es):
        
        dev = torch.device("cuda:"+str(4)) if torch.cuda.is_available() else torch.device("cpu")
        self.initial_masses  = initial_masses
        self.masses_t        = torch.tensor(final_masses,requires_grad=False, dtype=torch.double, device=dev)
        self.n_initial       = len(initial_masses)
        self.n_final         = len(final_masses)
        self.beam_Es         = beam_Es
        self.collider_energy = sum(beam_Es)
        self.dimensions      = self.get_dimensions()
        self.dim_ordered_names = [d.name for d in self.dimensions]

        self.dim_name_to_position = dict((d.name,i) for i, d in enumerate(self.dimensions))
        self.position_to_dim_name = dict((v,k) for (k,v) in self.dim_name_to_position.items())
      
    def generateKinematics(self, E_cm, random_variables):
        """Generate a phase-space point with fixed center of mass energy."""

        raise NotImplementedError
    
    def get_PS_point(self, random_variables):
        """Generate a complete PS point, including Bjorken x's,
        dictating a specific choice of incoming particle's momenta."""

        raise NotImplementedError

    def nDimPhaseSpace(self):
        """Return the number of random numbers required to produce
        a given multiplicity final state."""

        if self.n_final == 1:
            return 0
        return 3*self.n_final - 4

    def get_dimensions(self):
        """Generate a list of dimensions for this integrand."""
        
        dims = DimensionList()


        # Add the phase-space dimensions
        dims.extend([ ContinuousDimension('x_%d'%i,lower_bound=0.0, upper_bound=1.0) 
                                     for i in range(1, self.nDimPhaseSpace()+1) ])
        
        return dims


class FlatInvertiblePhasespace(VirtualPhaseSpaceGenerator):
    """Implementation following S. Platzer, arxiv:1308.2922"""

    # This parameter defines a thin layer around the boundary of the unit hypercube
    # of the random variables generating the phase-space,
    # so as to avoid extrema which are an issue in most PS generators.
    epsilon_border = 1e-10

    # The lowest value that the center of mass energy can take.
    # We take here 1 GeV, as anyway below this non-perturbative effects dominate
    # and factorization does not make sense anymore
    absolute_Ecm_min = 1.

   

    def __init__(self, *args, **opts):
        super(FlatInvertiblePhasespace, self).__init__(*args, **opts)
        if self.n_initial == 1:
            raise InvalidCmd("This basic generator does not support decay topologies.")

    def get_dimensions(self):
        """ Make sure the collider setup is supported."""

        
        if self.beam_Es[0]!=self.beam_Es[1]:
            raise InvalidCmd(
                "This basic generator only supports colliders with incoming beams equally energetic.")

        return super(FlatInvertiblePhasespace,self).get_dimensions()

    @staticmethod
    def get_flatWeights(E_cm, n, mass=None):
        """ Return the phase-space volume for a n massless final states.
        Vol(E_cm, n) = (pi/2)^(n-1) *  (E_cm^2)^(n-2) / ((n-1)!*(n-2)!)
        """
        if n==1: 
            # The jacobian from \delta(s_hat - m_final**2) present in 2->1 convolution
            # must typically be accounted for in the MC integration framework since we
            # don't have access to that here, so we just return 1.
            return 1.

        return math.pow((math.pi/2.0),n-1)*\
            (math.pow((E_cm**2),n-2)/(math.factorial(n-1)*math.factorial(n-2)))
    
   
    
    def massless_map(self,x,exp):

        return (x**(exp))*((exp+1)-(exp)*x)
        
    def bisect_vec(self,v_t, target=1.e-16, maxLevel=60):
        """Solve v = (n+2) * u^(n+1) - (n+1) * u^(n+2) for u. Vectorized"""
        
        exp=torch.arange(self.n_final-2,0,step=-1).to(v_t.device)
        level = 0
        left  = torch.zeros_like(v_t)
        right = torch.ones_like(v_t)
            
        checkV = torch.ones_like(v_t)*-1
        u =torch.ones_like(v_t)*-1
        
        while (level < maxLevel):
            u = (left + right) * (0.5**(level + 1))
           
            checkV = self.massless_map(u,exp)
            
           
            left *= 2.
            right *= 2.
            con=torch.ones_like(left)*0.5
            adder=torch.where(v_t<=checkV, con*-1.,con)
           
           
            left=left+(adder+0.5)
            right=right+(adder-0.5)
            
            level += 1
           

        return u
            

  
    
    @staticmethod
    def rho(M, N, m):
        """Returns sqrt((sqr(M)-sqr(N+m))*(sqr(M)-sqr(N-m)))/(8.*sqr(M))"""
        
        Msqr = M**2
        
        return ((Msqr-(N+m)**2) * (Msqr-(N-m)**2) )**0.5 / (8.*Msqr)
    
    
   
    
    def setInitialStateMomenta_t(self, output_momenta, E_cm):
        """Generate the initial state momenta."""

        if self.n_initial not in [1,2]:
            raise InvalidCmd(
               "This PS generator only supports 1 or 2 initial states")

        if self.n_initial == 1:
            if self.initial_masses[0]==0.:
                raise PhaseSpaceGeneratorError(
                    "Cannot generate the decay phase-space of a massless particle.")
            if self.E_cm != self.initial_masses[0]:
                raise PhaseSpaceGeneratorError(
                    "Can only generate the decay phase-space of a particle at rest.")

        if self.n_initial == 1:
            output_momenta[0] = torch.tensor([self.initial_masses[0] , 0., 0., 0.],dtype=torch.double, device=output_momenta[0].device)
            return

        elif self.n_initial == 2:
            if self.initial_masses[0] == 0. or self.initial_masses[1] == 0.:
                output_momenta[0] = torch.tensor([E_cm/2.0 , 0., 0., +E_cm/2.0],dtype=torch.double, device=output_momenta[0].device)
                output_momenta[1] = torch.tensor([E_cm/2.0 , 0., 0., -E_cm/2.0],dtype=torch.double, device=output_momenta[0].device)
            else:
                M1sq = self.initial_masses[0]**2
                M2sq = self.initial_masses[1]**2
                E1 = (E_cm**2+M1sq-M2sq)/ E_cm
                E2 = (E_cm**2-M1sq+M2sq)/ E_cm
                Z = math.sqrt(E_cm**4 - 2*E_cm**2*M1sq - 2*E_cm**2*M2sq + M1sq**2 - 2*M1sq*M2sq + M2sq**2) / E_cm
                output_momenta[0] = torch.tensor([E1/2.0 , 0., 0., +Z/2.0],dtype=torch.double, device=output_momenta[0].device)
                output_momenta[1] = torch.tensor([E2/2.0 , 0., 0., -Z/2.0],dtype=torch.double, device=output_momenta[0].device)
        return
    
    

    def generateKinematics(self, E_cm, random_variables):
        """Generate a self.n_initial -> self.n_final phase-space point
        using the random variables passed in argument.
        """
        
        
        assert (random_variables.shape[0]==self.nDimPhaseSpace())

        # Make sure that none of the random_variables is NaN.
        if any(torch.isnan(random_variables)):
            raise PhaseSpaceGeneratorError("Some of the random variables passed "+
              "to the phase-space generator are NaN: %s"%str(random_variables.data.tolist()))
        
       
        # The distribution weight of the generate PS point
        weight = 1.
        
        
        output_momenta_t=[]
        mass = self.masses_t[0]
        if self.n_final == 1:
            if self.n_initial == 1:
                raise InvalidCmd("1 > 1 phase-space generation not supported.")
            if mass/E_cm < 1.e-7 or ((E_cm-mass)/mass) > 1.e-7:
                raise PhaseSpaceGeneratorError("1 > 2 phase-space generation needs a final state mass equal to E_c.o.m.")
            output_momenta_t.append(torch.tensor([mass/2., 0., 0., mass/2.], dtype=torch.double, device=random_variables.device))
            output_momenta_t.append(torch.tensor([mass/2., 0., 0., -mass/2.], dtype=torch.double, device=random_variables.device))
            output_momenta_t.append(torch.tensor([mass   , 0., 0.,       0.], dtype=torch.double, device=random_variables.device))
            weight = self.get_flatWeights(E_cm, 1)
            return output_momenta_t, weight
        
        M    = [ 0. ]*(self.n_final-1)
        M[0] = E_cm
        M=torch.tensor(M,requires_grad=False, dtype=torch.double, device=random_variables.device)
        weight *= self.generateIntermediatesMassive(M, E_cm, random_variables)
        
        
        
        Q_t=torch.tensor([M[0], 0., 0., 0.],requires_grad=False, dtype=torch.double, device=random_variables.device)
        
        M=torch.cat((M,self.masses_t[-1:]),-1)
        
        q_t=(4.*M[:-1]*self.rho(M[:-1],M[1:],self.masses_t[:-1]))
        rnd=random_variables[self.n_final-2:3*self.n_final-4]
        cos_theta_t=(2.*rnd[0::2]-1.)
        sin_theta_t=(torch.sqrt(1.-cos_theta_t**2))
        phi_t=2*math.pi*rnd[1::2]
        cos_phi_t=torch.cos(phi_t)
        sqrt=torch.sqrt(1.-cos_phi_t**2)
        sin_phi_t=(torch.where(phi_t>math.pi,-sqrt,sqrt))
        a=torch.unsqueeze((q_t*sin_theta_t*cos_phi_t),0)
        b=torch.unsqueeze((q_t*sin_theta_t*sin_phi_t),0)
        c=torch.unsqueeze((q_t*cos_theta_t),0)
       
        lv=torch.cat((torch.zeros_like(a),a,b,c),0)
        
        
       
        for i in range(self.n_initial+self.n_final-1):
            
            if i < self.n_initial:
                
                output_momenta_t.append(torch.zeros_like(lv[:,0]))
                continue

            
            p2 =(lv[:,i-self.n_initial])
            p2=set_square_t(p2,self.masses_t[i-self.n_initial]**2)
            
            
            p2=boost_t(p2,boostVector_t(Q_t))
           
            p2=set_square_t(p2,self.masses_t[i-self.n_initial]**2)
            
            output_momenta_t.append(p2)
            
            nextQ_t=Q_t-p2
            
            nextQ_t=set_square_t(nextQ_t,M[i-self.n_initial+1]**2)
           
            Q_t = nextQ_t
        
        output_momenta_t.append(Q_t)
        
        
        
        self.setInitialStateMomenta_t(output_momenta_t,E_cm)
        
        return output_momenta_t, weight

    
    def generateIntermediatesMasslessVec(self, M_t, E_cm, random_variables): 
        """Generate intermediate masses for a massless final state."""
        
        
        
        u = self.bisect_vec(random_variables[:self.n_final-2])
       
        for i in range(2, self.n_final):
            M_t[i-1] = torch.sqrt(u[i-2]*(M_t[i-2]**2))
        
        return self.get_flatWeights(E_cm,self.n_final)
   



    def generateIntermediatesMassive(self, M, E_cm, random_variables):
        """Generate intermediate masses for a massive final state."""
        
       
        
        M[0] -= torch.sum(self.masses_t)
        
        weight = self.generateIntermediatesMasslessVec(M, E_cm, random_variables)
       
        K_t=M.clone()
        
        masses_sum=torch.flip(torch.cumsum(torch.flip(self.masses_t,(-1,)),-1),(-1,))
        M+=masses_sum[:-1]
        
        weight *= 8.*self.rho(
            M[self.n_final-2],
            self.masses_t[self.n_final-1],
            self.masses_t[self.n_final-2] )
        
       
        
       
        weight*=torch.prod((self.rho(M[:self.n_final-2],M[1:],self.masses_t[:self.n_final-2])/
                            self.rho(K_t[:self.n_final-2],K_t[1:],0.)) * (M[1:self.n_final-1]/K_t[1:self.n_final-1]),-1)
        
        weight *= torch.pow(K_t[0]/M[0],2*self.n_final-4)
        
        
        return weight

    def invertKinematics_t(self, E_cm, momenta):
        """ Returns the random variables that yields the specified momenta configuration."""

       
        assert (len(momenta) == (self.n_initial + self.n_final) )
        moms = list(momenta)
        
        weight = 1.

        if self.n_final == 1:
            if self.n_initial == 1:
                raise PhaseSpaceGeneratorError("1 > 1 phase-space generation not supported.")
            return [], self.get_flatWeights(E_cm,1) 

        
        random_variables = torch.tensor([-1.0]*self.nDimPhaseSpace(),dtype=torch.double,device=momenta[0].device)
        
        M    = [0., ]*(self.n_final-1)
        M[0] = E_cm
        M=torch.tensor(M,dtype=torch.double,device=momenta[0].device)

        
        Q_0=torch.tensor([M[0],0.,0.,0.],dtype=torch.double,device=momenta[0].device)
        Q=[torch.zeros_like(Q_0)]*(self.n_final-1)
        Q[0]=Q_0
        
        for i in range(2,self.n_final):
            for k in range(i, self.n_final+1):
                Q[i-1] = Q[i-1] + moms[k+self.n_initial-1]
            M[i-1] = abs(square_t(Q[i-1]))**0.5
        
        weight = self.invertIntermediatesMassive_t(M, E_cm, random_variables)
        
        for i in range(self.n_initial,self.n_final+1):
            
            boost_vec = -boostVector_t(Q[i-self.n_initial])
            p=boost_t(moms[i],boost_vec)
            random_variables[self.n_final-2+2*(i-self.n_initial)] = (cosTheta_t(p)+1.)/2.
            phi = phi_t(p)
            if (phi < 0.):
                phi += 2.*math.pi
            random_variables[self.n_final-1+2*(i-self.n_initial)] = phi / (2.*math.pi)
        
        return random_variables, weight

    def invertIntermediatesMassive_t(self, M, E_cm, random_variables):
        """ Invert intermediate masses for a massive final state."""

        
        K=M.clone()
        
        masses_sum=torch.flip(torch.cumsum(torch.flip(self.masses_t,(-1,)),-1),(-1,))
        K-=masses_sum[:-1]
        
        weight = self.invertIntermediatesMassless_t(K, E_cm, random_variables)
        
        
        weight *= 8.*self.rho(M[self.n_final-2],
                              self.masses_t[self.n_final-1],
                              self.masses_t[self.n_final-2])
        
        weight*=torch.prod((self.rho(M[:self.n_final-2],M[1:],self.masses_t[:self.n_final-2])/
                            self.rho(K[:self.n_final-2],K[1:],0.)) * (M[1:self.n_final-1]/K[1:self.n_final-1]),-1)
       
        weight *= torch.pow(K[0]/M[0],2*self.n_final-4)

        return weight

    def invertIntermediatesMassless_t(self, K, E_cm, random_variables):
        """ Invert intermediate masses for a massless final state."""

        u=(K[1:]/K[:-1])**2
        exp=torch.arange(self.n_final-2,0,step=-1).to(K.device)
        random_variables[:self.n_final-2]=self.massless_map(u,exp)
        
        
        return self.get_flatWeights(E_cm, self.n_final)
    
    
    



#=========================================================================================
# Standalone main for debugging / standalone trials
#=========================================================================================
if __name__ == '__main__':

    import random

    E_cm  = 5000.0
    dev = torch.device("cuda:"+str(4)) if torch.cuda.is_available() else torch.device("cpu")
    # Try to run the above for a 2->8.
    my_PS_generator = FlatInvertiblePhasespace([0.]*2, [100. + 10.*i for i in range(8)],
                                            beam_Es =(E_cm/2.,E_cm/2.))
     #Try to run the above for a 2->1.    
    #my_PS_generator = FlatInvertiblePhasespace([0.]*2, [5000.0],beam_Es =(E_cm/2.,E_cm/2.))
    random_variables = [random.random() for _ in range(my_PS_generator.nDimPhaseSpace())]
    
    random_variables_t = torch.tensor(random_variables,dtype=torch.double,device=dev,requires_grad=True)
    start_time=datetime.datetime.utcnow()
        
    momenta, wgt = my_PS_generator.generateKinematics(E_cm, random_variables_t)
    
    
    print ("\n =========================")
    print (" ||    PS generation    ||")
    print (" =========================")

    print ("\nRandom variables :\n",random_variables_t)
    print (momenta)
    
    print ("Phase-space weight : %.16e\n"%wgt)
    
    variables_reconstructed, wgt_reconstructed = \
                                         my_PS_generator.invertKinematics_t(E_cm, momenta)
    
    """
    print ("\n =========================")
    print (" || Kinematic inversion ||")
    print (" =========================")
    print ("\nReconstructed random variables :\n",variables_reconstructed)
    """
    differences = [
        abs(variables_reconstructed[i]-random_variables[i])
        for i in range(len(variables_reconstructed))
    ]
    print ("Reconstructed weight = %.16e"%wgt_reconstructed)
    if differences:
        print ("\nMax. relative diff. in reconstructed variables = %.3e"%\
            max(differences[i]/random_variables[i] for i in range(len(differences))))
        if(max(differences[i]/random_variables[i] for i in range(len(differences)))>1e-12):
            print("ALARM!!!!!!!")
    print ("Rel. diff. in PS weight = %.3e\n"%((wgt_reconstructed-wgt)/wgt))
    

    print('-'*100)
   
 

