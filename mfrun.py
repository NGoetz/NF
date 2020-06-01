import torch
import numpy as np
from experiment_seq import *
from experiment_vegas import *
from multiprocessing import Process,Queue,Manager
from collections import OrderedDict
import datetime



def exp_range(start, end, mul):
    while start <= end:
        yield start
        start *= mul
        
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def create_fun(gn, gw):
    
    if(gn==1):
        def f(x):
                return torch.exp(-torch.sum((x-0.5)**2/(gw**2),-1))
        return f
    
    if(gn==2):
        def f(x):
                return torch.exp(-torch.sum((x-0.25)**2/(gw**2),-1))+torch.exp(-torch.sum((x-0.75)**2/(gw**2),-1))
        return f
    
    if(gn==4):
        def f(x):
            shift=torch.ones_like(x)*0.25
            shift1=shift.clone()*3
            lim=int((shift.shape[1]/2))
            shift2=torch.cat((shift[:,:lim],shift1[:,lim:]),-1)
            shift3=torch.cat((shift1[:,:lim],shift[:,lim:]),-1)
            return torch.exp(-torch.sum((x-shift)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift1)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift2)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift3)**2/(gw**2),-1))
        return f
    
    if(gn==8):
        def f(x):
            shift=torch.ones_like(x)*0.25#000
            shift1=shift.clone()*3#111
            lim=int((shift.shape[1]/3))
            shift2=torch.cat((shift[:,:lim],shift1[:,lim:2*lim],shift[:,2*lim:]),-1) #010
            shift3=torch.cat((shift1[:,:lim],shift[:,lim:]),-1)#100
            shift4=torch.cat((shift1[:,:lim],shift1[:,lim:2*lim],shift[:,2*lim:]),-1) #110
            shift5=torch.cat((shift[:,:2*lim],shift1[:,2*lim:]),-1) #001
            shift6=torch.cat((shift1[:,:lim],shift[:,lim:2*lim],shift1[:,2*lim:]),-1) #101
            shift7=torch.cat((shift[:,:lim],shift1[:,lim:]),-1) #011
            return torch.exp(-torch.sum((x-shift)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift1)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift2)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift3)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift4)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift5)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift6)**2/(gw**2),-1))+torch.exp(-torch.sum((x-shift7)**2/(gw**2),-1))
        return f
    
def create_funv(gn, gw):
    
    if(gn==1):
        def f(x):
                return np.exp(-np.sum((x-0.5)**2/(gw**2),-1))
        return f # peak at [0.5,...,0.5]
    
    if(gn==2):
        def f(x):
                return np.exp(-np.sum((x-0.25)**2/(gw**2),-1))+np.exp(-np.sum((x-0.75)**2/(gw**2),-1))
        return f  #peak at [0.25,...,0.25] and [0.75,...,0.75]
    
    if(gn==4):
        
        def f(x):
            shift=np.ones_like(x)*0.25
            shift1=shift.copy()*3
            lim=int((shift.shape[0]/2))
            shift2=np.concatenate((shift[:lim],shift1[lim:]),-1)
            shift3=np.concatenate((shift1[:lim],shift[lim:]),-1)
            return np.exp(-np.sum((x-shift)**2/(gw**2),-1))+np.exp(-np.sum((x-shift1)**2/(gw**2),-1))+np.exp(-np.sum((x-shift2)**2/(gw**2),-1))+np.exp(-np.sum((x-shift3)**2/(gw**2),-1))
        return f #peak at [0.25,...,0.25], [0.75,...,0.75],[0.25,...,0.75], [0.25,...,0.75]
                #in dim4, there would be in the [0,1] plane only two peaks, as two are identical
                # other than gn=2, the [1,2] plane has 4 peaks
    
    if(gn==8):
        
        def f(x):
            shift=np.ones_like(x)*0.25#000
            shift1=shift.copy()*3#111
            lim=int(shift.shape[0]/3)
            shift2=np.concatenate((shift[:lim],shift1[lim:2*lim],shift[2*lim:]),0) #010
            shift3=np.concatenate((shift1[:lim],shift[lim:]),0)#100
            shift4=np.concatenate((shift1[:lim],shift1[lim:2*lim],shift[2*lim:]),0) #110
            shift5=np.concatenate((shift[:2*lim],shift1[2*lim:]),0) #001
            shift6=np.concatenate((shift1[:lim],shift[lim:2*lim],shift1[2*lim:]),0) #101
            shift7=np.concatenate((shift[:lim],shift1[lim:]),0) #011
            return np.exp(-np.sum((x-shift)**2/(gw**2),-1))+np.exp(-np.sum((x-shift1)**2/(gw**2),-1))+np.exp(-np.sum((x-shift2)**2/(gw**2),-1))+np.exp(-np.sum((x-shift3)**2/(gw**2),-1))+ np.exp(-np.sum((x-shift4)**2/(gw**2),-1))+np.exp(-np.sum((x-shift5)**2/(gw**2),-1))+np.exp(-np.sum((x-shift6)**2/(gw**2),-1))+np.exp(-np.sum((x-shift7)**2/(gw**2),-1))
        return f #here, in [0,1] plane there are actually 4 peaks
              
if __name__ == '__main__':
  
    gpus=4
    hypopt_n=75
    hypopt=False
    n_bins_r=(4,13)
    lr_r=(1e-5,2e-2)
    weight_decay_r=(1e-7,1e-4)
    batch_size_r=(60000,90000)#20000,90000
    repeat=1
    
    best_loss_rel=1000
    best_loss=1000
    best_id=0
    exdict={}
    exdictv={}
    i=0
    logdir='logs/sacred/mfrun10'
    for nnl in my_range(5,5,5): #4,10,1
        for nnw in my_range(15,15,4): #7,20,2  #15
            for dim in exp_range(2,2,2): #exp(2,32,2)
                for cc in my_range(int(2*np.ceil(np.log2(dim))), int(2*np.ceil(np.log2(dim))),1): 
                    #int(np.ceil(2*np.log2(dim))), dim,1, 
                        for gn in exp_range(2,2,2): #exp 1,8,2 limit by dim!
                            for gw in my_range(0.25,0.25,0.04): #0.1,0.3,0.05 #>0.15
                                f=create_fun(gn,gw)
                                
                                fv=create_funv(gn,gw)
                                exdict[str(i)]={"n_flow":dim,"NN_length":nnl, "NN_width": nnw, "n_cells":cc,
                                                "f": f, "logdir":logdir, "gn": gn, "gw": gw}
                                exdictv[str(i)]={"n_flow":dim,"f": fv, "logdir":logdir,"gn": gn, "gw": gw}
                                i=i+1
                                

    ex_n=len(exdict)
   
    if hypopt:
        print(str(len(exdict)*(hypopt_n+gpus)))
        cfgdict={}
        for i in range(ex_n):
            lr=torch.ones(1)
            wd=torch.ones(1)
            for j in range(int(hypopt_n-hypopt_n%gpus)):
                bs=torch.randint(batch_size_r[0],batch_size_r[1], (1,))
                lr.uniform_(lr_r[0],lr_r[1])
                wd.uniform_(weight_decay_r[0],weight_decay_r[1])
                b=torch.randint(n_bins_r[0], n_bins_r[1], (1,))
                cfgdict[str(i)+";"+str(j)]={"batch_size":bs.item(), "lr":lr.item(),"weight_decay": wd.item(),"n_bins": b.item(),
                                            "logdir":logdir+"/hypopt", "internal_id": j}

        ex_init(logdir+"/hypopt") 
        time_total={}
        hyp_res={}
        for i in range(ex_n):
            start_time=datetime.datetime.utcnow()
            processes = []
            ex_cfg=exdict[str(i)]
            j=0
            hypdict={}
            best_loss_rel=1000
            best_id=0

            while j<int(hypopt_n-hypopt_n%gpus):
                m=Manager()
                q=m.Queue()
                for n in range(gpus):
                    hyp_cfg=cfgdict[str(i)+";"+str(j)]
                    j=j+1

                    hyp_cfg['dev']=n
                    hyp_cfg['q']=q
                    z = ex_cfg.copy()
                    z.update(hyp_cfg)
                    config_updates={}
                    config_updates['config_updates']=z
                    p = Process(target=ex.run, kwargs=config_updates)
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                    while not q.empty():
                        res=q.get()
                        hyp_res[str(res[1])]={'best_loss':res[0],'best_loss_rel':res[2],'best_count':res[3], 'loss_var': res[6], 
                                          'best_epoch': res[7], 'dim': z['n_flow'], "gn": z['gn'],
                                          "gw": z['gw'],"nnl": z['NN_length'], "nnw":z['NN_width'], "time":res[9] }
                        if(res[2]<best_loss_rel):  
                            best_loss_rel=res[2]
                            best_id=res[10]

            exdict[str(i)].update(cfgdict[str(i)+";"+str(best_id)])
            time_total[str(i)]=(datetime.datetime.utcnow()-start_time).total_seconds()
        hyp_res_sort = sorted((value['gn'], value['gw'], value['dim'], value['nnl'],
                               value['nnw'], value['best_loss'],  value['best_loss_rel'], value['loss_var'], 
                               value['best_epoch'], 
                               value['best_count'],value['time'],key) for (key,value) in hyp_res.items())
        filename_hyp=logdir+'/hypopt/list'
        file_hyp= open(filename_hyp,"w+")
        file_hyp.write('ID, number of peaks, width of peaks, NN_length, NN_width, dimensions, best_loss, best_loss_rel,'+
                   'var_loss, best_epoch, best_count, time \n')
        file_hyp.close()

        for k in hyp_res_sort:
            file_hyp= open(filename_hyp,"a")
            file_hyp.write(str(k[11])+', '+str(k[0])+', '+str(k[1])+', '+str(k[3])+', '+str(k[4])+', '+str(k[2])
                       +', '+str(k[5])+', '+str(k[6])+', '+str(k[7])+
                       ', '+str(k[8])+', '+str(k[9])+', '+str(k[10]))
            file_hyp.write("\n")
            file_hyp.close() 
                    
    #print(time_total)            
    ex_init(logdir+"/NIS")
    exv_init(logdir+"/VEGAS")
    
    if not os.path.exists(logdir):
                os.mkdir(logdir)
    filename=logdir+'/list'
    file= open(filename,"w+")
    file.write('ID, number of peaks, width of peaks, NN_length, NN_width,dimensions, modus, best_loss, best_loss_rel,'+
           'var_loss, best_epoch, best_count, time, time_total\n')
    file.close()
    for z in range(repeat):
        for i in range(ex_n):
            resdict={}
            processes = []
            m=Manager()
            q=m.Queue()
            if(len(exdict)==0 or len(exdictv)==0):
                    break
            run_cfg=exdict[str(i)]
            run_cfgv=exdictv[str(i)]
            for n in range(gpus+1):
                n=n-1
                run_cfg['q']=q
                run_cfg['dev']=n
                run_cfg['logdir']=logdir+"/NIS"
                run_cfgv['q']=q
                config_updates={}
                config_updatesv={}
                config_updates['config_updates']=run_cfg
                config_updatesv['config_updates']=run_cfgv
                if n>=0:
                    p = Process(target=ex.run, kwargs=config_updates)
                else:
                    p=Process(target=exv.run, kwargs=config_updatesv) 
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                while not q.empty():
                    res=q.get()
                    if(res[8]=='NIS'):
                        resdict[str(res[1])]={'best_loss':res[0],'best_loss_rel':res[2],'best_count':res[3], 'loss_var': res[6], 
                                          'best_epoch': res[7], 'dim': run_cfg['n_flow'], "gn": run_cfg['gn'],
                                          "gw": run_cfg['gw'], "modus": res[8], "time":res[9], "nnl": run_cfg['NN_length'],
                                          "nnw": run_cfg['NN_width'] }
                    else:
                        resdict[str(res[1])+'V']={'best_loss':res[0],'best_loss_rel':res[2],
                                                  'best_count':res[3], 'loss_var': res[6], 
                                          'best_epoch': res[7], 'dim': run_cfg['n_flow'], "gn": run_cfg['gn'],
                                          "gw": run_cfg['gw'], "modus": res[8], "time":res[9], "nnl": run_cfg['NN_length'],
                                          "nnw": run_cfg['NN_width'] }


            for (key,value) in resdict.items():
                if hypopt and value['modus']=='NIS':
                    total_time=time_total[str(i)]+value['time']
                else:
                    total_time=value['time']
                file=open(filename,"a")
                file.write(str(key) +', '+str(value['gn'])+', '+str(value['gw'])+', '+str(value['nnl'])+', '+str(value['nnw'])
                           +', '+str(value['dim'])+', '+
                           str(value['modus'])+', '+str(value['best_loss'])
                           +', '+str(value['best_loss_rel'])+', '+str(value['loss_var'])+
                           ', '+str(value['best_epoch'])+', '+str(value['best_count'])
                           +', '+str(value['time'])+','+str(total_time))
                file.write("\n")
                file.close() 