#!/usr/bin/env python2

import random
import os
import sys
import vegas
import math
import torch
import datetime


from processes.all_processes import *
from model.parameters import ModelParameters
from phase_space_generator.vectors import Vector, LorentzVector,LorentzVectorDict, LorentzVectorList

from nisrep.normalizing_flows.manager import *
from nisrep.PhaseSpace.flat_phase_space_generator import *


torch.set_default_dtype(torch.double)

class Colour:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

    
    
module_name = os.path.basename(os.path.dirname(os.path.realpath( __file__ )))

all_process_classes = [{all_process_classes}]

# For now, the feature of specifying an SLHA param card to initialise
# the value of independent parameters is not supported yet.
active_model = ModelParameters(None)

# Center of mass of the collision in GeV
E_cm = 500


print("")
print("The module '%s' contains %d processes"%(module_name, len(all_process_classes)))
print("")
#print(str(active_model))
print("")


#print(all_process_classes)

for process_class in all_process_classes:
    
    print(">>> Running process %s%s%s"%(Colour.BLUE,process_class.__name__,Colour.END))

    # Generate a random PS point for this process
    process = process_class()
    external_masses = process.get_external_masses(active_model)

    # Ensure that E_cm offers enough twice as much energy as necessary 
    # to produce the final states
    this_process_E_cm = max( E_cm, sum(external_masses[1])*2. )
    

    element=0

    my_ps_generator=FlatInvertiblePhasespace(external_masses[0], external_masses[1],
        beam_Es = (this_process_E_cm/2.,this_process_E_cm/2.),)
    
    s=this_process_E_cm**2
    
    def f(x):
        momenta, jac = my_ps_generator.generateKinematics_batch(E_cm, x)
        q=0
        #element=torch.empty(x.shape[0],1).to(x.device)
        element=[0]*momenta.shape[0]
        #e=0
        element=[process.smatrix(LorentzVectorList(momenta[ind,:,:]), active_model)*jac[ind]/(2*s)
                 for ind, q in enumerate(element)]
        
        return torch.tensor(element,device=x.device)
    
    def fv(x):
        momenta, jac = my_ps_generator.generateKinematics_batch(E_cm, torch.tensor(x).unsqueeze(0))
        element=process.smatrix(LorentzVectorList(momenta[0,:,:]), active_model)*jac[0]
        
        return element/(2*s)

       
        
 
    
    n_flow = my_ps_generator.nDimPhaseSpace() # number of dimensions

    NF =  PWQuadManager(n_flow=n_flow)
    NF.create_model(n_cells=2, n_bins=10, NN=[10,10,10,10,10],dev=torch.device("cpu"))
    optim = torch.optim.Adamax(NF._model.parameters(),lr=5e-3, weight_decay=1e-04) 
    
    
    print("-----")
    start_time=datetime.datetime.utcnow()
    integv=vegas.Integrator([[0, 1]]*my_ps_generator.nDimPhaseSpace())
    integv(fv, nitn=5, neval=1000)
    result = integv(fv, nitn=15, neval=1000)
    
    print(result.summary())
    print(result.mean)
    print(str(result.mean) + " +/- " +str(result.sdev))
    print(str(result.mean/(2.56819*10**(-9))) + " +/- " +str(result.sdev/(2.56819*10**(-9))))
    end_time=datetime.datetime.utcnow()
    print("Duration:")
    print((end_time-start_time).total_seconds())
    print("-----")
    
    start_time=datetime.datetime.utcnow()
    sig,sig_err=NF._train_variance_forward_seq(f,optim,"./logs/tmp/",100,200,0,True, False,True,mini_batch_size=50)
   
 
    print("-----------")
    print(str(sig) + " +/- " +str(sig_err))
    sig=sig/(2.56819*10**(-9)) #GeV**2 -> pb
    sig_err=sig_err/(2.56819*10**(-9))
    print(str(sig) + " +/- " +str(sig_err))
    end_time=datetime.datetime.utcnow()
    print("Duration:")
    print((end_time-start_time).total_seconds())
    print("-----------")
    print('Initial loss')
    print(NF.int_loss)
    print('Epoch of best result')
    print(NF.best_epoch)
    print('Best loss')
    print(NF.best_loss)
    print('Best loss relative')
    print(NF.best_loss_rel)


