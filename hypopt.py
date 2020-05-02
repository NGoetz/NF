import torch
from experiment_seq import ex
from multiprocessing import Process,Queue,Manager

def exp_range(start, end, mul):
    while start <= end:
        yield start
        start *= mul
        
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

        
if __name__ == '__main__':
  
    best_var=1000
    best_id=0
    exdict={}
    i=0
   
    for bs in my_range(8000,20000, 2000): 
        for nnl in my_range(3,10,1): 
            for nnw in my_range(4,16,2): 
                for lr in exp_range(1e-7,1e-1,2):
                    for wd in exp_range(1e-8,1e-6,5):
                        for b in my_range(5,25,5): 
                            i=i+1
                            exdict[str(i)]={"batch_size":bs,"NN_length":nnl, "NN_width": nnw, "lr":lr,"weight_decay": wd,
                                            "n_bins": b}
                            
    
    print(str(len(exdict)))
    
    while len(exdict)>0:
        processes = []
        m=Manager()
        q=m.Queue()
        for n in range(6):
            if(len(exdict)==0):
                break
            run_cfg=exdict.popitem()[1]
            
            run_cfg['q']=q
            run_cfg['dev']=n
            config_updates={}
            config_updates['config_updates']=run_cfg
            p = Process(target=ex.run, kwargs=config_updates)
            p.start()
            processes.append(p)
        for p in processes:
            while not q.empty():
                res=q.get()
                if(res[0]<best_var):
                    best_var=res[0]
                    best_id=res[1]
            p.join()
            
                
        
     
       
   
    filename='logs/sacred/runs1/result'
    file= open(filename,"w+")
    file.write("Id of best run: "+str(best_id)+" with variance: "+str(best_var))
    file.close()                            
    
                    
                
    
