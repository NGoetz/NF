import torch
from experiment_seq import ex
from multiprocessing import Process,Queue,Manager
from collections import OrderedDict
from operator import itemgetter

def exp_range(start, end, mul):
    while start <= end:
        yield start
        start *= mul
        
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

        
if __name__ == '__main__':
  
    random=True
    n=1000
    best_loss_rel=1000
    best_loss=1000
    best_per=1e30
    best_id=0
    best_id2=0
    best_id3=0
    exdict={}
    i=0
    logdir='logs/sacred/runs4'
    if not random:
        for bs in exp_range(5000,20000, 2): 
            for nnl in my_range(5,9,2): 
                for nnw in my_range(6,10,2): 
                    for lr in exp_range(1e-5,1e-3,2):
                        for wd in exp_range(1e-7,1e-6,3):
                            for b in my_range(12,18,3): 
                                i=i+1
                                exdict[str(i)]={"batch_size":bs,"NN_length":nnl, "NN_width": nnw, "lr":lr,"weight_decay": wd,
                                                "n_bins": b, "logdir":logdir}
    else:
        lr=torch.ones(1)
        wd=torch.ones(1)
        for j in range(n):
            bs=torch.randint(11000, 220000, (1,))
            nnl=torch.randint(4, 7, (1,))
            nnw=torch.randint(9, 12, (1,))
            lr.uniform_(1e-3,1e-2)
            wd.uniform_(1e-7,1e-6)
            b=torch.randint(5, 15, (1,))
            exdict[str(j)]={"batch_size":bs.item(),"NN_length":nnl.item(), "NN_width": nnw.item(), "lr":lr.item(),
                            "weight_decay": wd.item(),"n_bins": b.item(), "logdir":logdir}

    
    print(str(len(exdict)))
    resdict={}
    while len(exdict)>0:
        processes = []
        m=Manager()
        q=m.Queue()
        for n in range(6):
            if(len(exdict)==0 ):
                break
            run_cfg=exdict.popitem()[1]
            #run_cfg={} #!!!!
            run_cfg['q']=q
            run_cfg['dev']=n
            config_updates={}
            config_updates['config_updates']=run_cfg
            p = Process(target=ex.run, kwargs=config_updates)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            while not q.empty():
                res=q.get()
                resdict[str(res[1])]={'best_loss':res[0],'best_loss_rel':res[2],'best_per':res[2]*res[3], 
                                      'DKL':res[4], 'varJ': res[5]}
                if(res[0]<best_loss):  
                    best_loss=res[0]
                    best_id=res[1]
                if(res[2]<best_loss_rel):
                    best_loss_rel=res[2]
                    best_id2=res[1]
                if(res[2]*res[3]<best_per):
                    best_per=res[2]*res[3]
                    best_id3=res[1]
            
       
   
    resdict_sort = sorted((value['best_loss_rel'],value['best_loss'],value['best_per'],value['DKL'],value['varJ'], key) for (key,value) in resdict.items())
    filename=logdir+'/result'
    file= open(filename,"w+")
    file.write("Id of best run rel: "+str(best_id2)+" with relative loss: "+str(best_loss_rel)+
               " \n Id of best run: "+str(best_id)+" with loss: "+str(best_loss)+
               " \n Id of best gain: "+str(best_id3)+" with gain factor: "+str(best_per))
    file.close() 
    filename=logdir+'/list'
    file= open(filename,"w+")
    for k in resdict_sort:
        file.write('Id: '+str(k[5])+' best_loss_rel: '+str(k[0])+
                  ' best_loss: '+str(k[1])+' best_per: '+str(k[2])+' varJ: '+str(k[3])+' DKL: '+str(k[4]))
        file.write("\n")
    file.close() 
    
                    
                
    
