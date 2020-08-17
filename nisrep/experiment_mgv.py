import torch
from nisrep.normalizing_flows.manager import *
import datetime
import pprint
import vegas


neval=5000
nitn=30

def prov(para):
   
    n_flow=para["n_flow"]
    n_bins=para["n_bins"]
    NN_width=para["NN_width"]
    NN_length=para["NN_length"]
    dev=para["dev"]
    lr=para["lr"]
    wd=para["weight_decay"]
    var_n=para["var_n"]
    batch_size=para["batch_size"]
    pt=para["pt"]
    q=para["q"]
    internal_id=para["id"]
    fv=para["f"]
    dev=para["dev"]
    logdir=para["logdir"]+"/"+str(internal_id)
    log=para["log"]
    if not os.path.exists(logdir):
                os.mkdir(logdir)
    
    filename=logdir+'/log.txt'
    f=open(filename,"w+") 
  
    f.write(str(nitn)+" iterations with "+str(neval)+" evaluations \n")
    start_time=datetime.datetime.utcnow()
    integv=vegas.Integrator([[0, 1]]*n_flow,max_nhcube=1)

    
    result = integv(fv, nitn=nitn, neval=neval,)
    sig=result.mean/(2.5681894616*10**(-9))
    sig_err=result.sdev/(2.5681894616*10**(-9))
    m = vegas.AdaptiveMap(integv.map)


    y = np.random.uniform(0., 1., (var_n, n_flow)) 
    x = np.empty(y.shape, float)           
    jac = np.empty(y.shape[0], float)
    f2 = np.empty(y.shape[0], float)

    m.map(y, x, jac)  
    for i in range(var_n):
        f2[i] = (jac[i] * fv(x[i]))
    v_var=np.var(f2)
    w_max=np.max(f2)
    w_mean=np.mean(f2)
    
    

    f.write("RESULT \n")
    f.write("{0:5E}  +/- {1:3E}  \n".format(result.mean,result.sdev))
    
    f.write("{0:5E}  +/- {1:3E} pb \n".format(result.mean/(2.5681894616*10**(-9)),result.sdev/(2.56819*10**(-9))))
    
    f.write("Final Variance: "+str(v_var)+"\n")
    
    f.write("Unweighting efficiency: "+str(w_mean/w_max)+"\n")
    end_time=datetime.datetime.utcnow()

    f.write("Duration: \n")
    f.write(str((end_time-start_time).total_seconds())+"\n")
    f.write("-----\n")
    
    q.put((0, None,0,(nitn+5)*neval, 0,
           0,0, 0,"VEGAS",(end_time-start_time).total_seconds(),internal_id,sig, 
           sig_err,v_var))
    
    
    f.write(pprint.pformat(para))

    f.close()
    pass