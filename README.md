This project is deprecated. Please refer to [Zünis](https://github.com/ndeutschmann/zunis/) for the follow-up implementation.
# NF
An implementation of importance sampling with normalizing flows based on (Müller et. al, 2019), implemented in Pytorch. A phase space generator based on "RAMBO on diet" (Plätzer, 2018) is included in order to be used with matrix elements generated by, for example, MadGraph 5.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Status](#status)


## General info
Conventional adaptive importance sampling, like with the VEGAS algorithm, often fails when the integration variables are correlated, for example for a Gaussian double peak on the diagonal. Training normalizing flows via neural networks offers the possibility to generate a variable transformation which reduces the variance even in such cases.



## Technologies
* Python - version >2.5
* PyTorch - version 1.6
* LHAPDF - version 6.3.0

## Setup
Install the package using 
`pip install . `
from inside the folder.
Afterwards, the code can be run by importing the package directly!

## Code Examples
Training a double peak:
```
from nisrep.normalizing_flows.manager import *

def camel(x):
    return torch.exp( -((x[:,0]-0.75)**2+(x[:,1]-0.75)**2)/(0.2**2))+torch.exp( -((x[:,0]-0.25)**2+(x[:,1]-0.25)**2)/(0.2**2))

n_flow=2

NF =  PWQuadManager(n_flow=n_flow)
NF.create_model(2,4, [3]*3)
optim = torch.optim.Adamax(NF._model.parameters(),lr=2e-3, weight_decay=1e-04) 
NF._train_variance_forward_seq(camel,optim,True, "./logs/tmp/",10000,300,0,True, True,preburn_time=50)
```
Afterwards, `NF.best_model` can be used for Monte-Carlo integration!
```
res, res_err=NF.integrate(f, nitn, neval)
```
If the function is a matrix element of collision event, the phase space generator module can be used to map 
the output of the model into the phase space:
```
from nisrep.PhaseSpace.flat_phase_space_generator import *

mass=100
my_PS_generator=FlatInvertiblePhasespace([mass]*2, [mass + 0.*i for i in range(4)],
                                            pdf=None,pdf_active=False)
momenta, wgt = my_PS_generator.generateKinematics_batch(E_cm, w,pT_mincut=0,
                                                            delR_mincut=0, rap_maxcut=-1,pdgs=[2,-1])
```

Take a look at the [Demo Notebook](https://www.niklasgoetz.com/gallery/Demo.zip)
## Features

* Neural importance sampling based on variance loss with linear and quadratic coupling cells
* Flat phase space generator for massive and massless 2 -> X processes, supporting 
  PDFs as well as pT, del_r and rapidity cuts


To-do list:
* Inversion of the coupling cells and the phase space generator


## Status
Project is: This project is deprecated. Please refer to [Zünis](https://github.com/ndeutschmann/zunis/) for the follow-up implementation.


