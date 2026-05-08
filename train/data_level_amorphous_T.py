import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
from utils.dataset import Amorphous_T_Dataset
from model.models import Dual_modal_trainer, Predictor
from model.embedding import embed_x, embed_T
from tqdm import tqdm

import sys
import os


def collate_fn(batch):
    return batch[0]

def data_level_amorphous_T(seed, device, EPOCH3=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

    name="../results/log_data_level_amorphous_T"
    log_name = f"{name}_{seed}.txt"
    if os.path.exists(log_name):
        os.remove(log_name)
    log_file = open(log_name, "a")
    sys.stdout = log_file
    
    
    str_dataset_train1=Amorphous_T_Dataset("train", device)
    str_dataset_test1=Amorphous_T_Dataset( "test", device)
   
    
    
    str_train_loader1=torch.utils.data.DataLoader(str_dataset_train1, shuffle=True, collate_fn=collate_fn)
    str_test_loader1=torch.utils.data.DataLoader(str_dataset_test1, shuffle=False, collate_fn=collate_fn)
    
    


    g=Dual_modal_trainer(x_in_dim=2935, p_in_dim=512)
    g.load_state_dict(torch.load("../model/g"+str(seed)+".pt"))
    g=g.to(device)
    
        
    
    pr=Predictor(x_in_dim=2935)
    pr.load_state_dict(torch.load("../model/pr"+str(seed)+".pt"))
    pr=pr.to(device)

    
    
    

  
    pr1=Predictor(x_in_dim=2935).to(device)  
    pr1.encoder_x.load_state_dict(g.encoder_x.state_dict())
    pr1.decoder.load_state_dict(pr.decoder.state_dict())

    
    
    optimizer=torch.optim.Adam([{"params":pr1.encoder_x.parameters(), "lr":1e-2}, {"params":list(pr1.decoder.linear1.parameters())+list(pr1.decoder.linear2.parameters()), "lr":1e-4}, {"params":list(pr1.decoder.linear3.parameters())+list(pr1.decoder.linear4.parameters()), "lr":1e-6}])
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
   
    for epoch in tqdm(range(EPOCH3)):
        loss_list=[]
        for node_attr, edge_index, edge_attr, temperature, s_idx, y_list in str_train_loader1:
            
            e_x=embed_x(node_attr, edge_attr, edge_index, s_idx).to(device)
           
            loss=0
            for T in range(1):
                y=y_list[T]
                e_T=embed_T(temperature[T], s_idx, device)
                out=pr1(torch.cat([e_x,e_T],-1))
                loss=loss+torch.nn.functional.l1_loss(out, y)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        
        test_loss_list=[[],[],[],[]]
        with torch.no_grad():
            for node_attr, edge_index, edge_attr, temperature, s_idx, y_list in str_test_loader1:
                
                e_x=embed_x(node_attr, edge_attr, edge_index, s_idx).to(device)
            
                for T in range(1):
                    y=y_list[T]
                    e_T=embed_T(temperature[T], s_idx, device)
                    out=pr1(torch.cat([e_x,e_T],-1))
                    loss=torch.nn.functional.l1_loss(out, y)
                    test_loss_list[[1000,1500,2000,2500].index(temperature[T])].append(loss.item())
        
        print(epoch, f"{np.mean(loss_list)*10:.3f}", f"{np.mean(test_loss_list[0])*10:.3f}", f"{np.mean(test_loss_list[1])*10:.3f}", f"{np.mean(test_loss_list[2])*10:.3f}", f"{np.mean(test_loss_list[3])*10:.3f}", flush=True)
       

   

        scheduler.step()
    
    sys.stdout = sys.__stdout__  
    log_file.close()
    
    




