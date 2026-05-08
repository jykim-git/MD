import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
from utils.dataset import Liflow_Dataset
from model.models import Dual_modal_trainer, Predictor
from model.embedding import embed_x, embed_T
from tqdm import tqdm
from utils.closed_form_initialization import intial_value
import sys
import os


def collate_fn(batch):
    return batch[0]

def model_level(seed, device, EPOCH1=50, EPOCH2=50):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

    name="../results/log_model_level"
    log_name = f"{name}_{seed}.txt"
    if os.path.exists(log_name):
        os.remove(log_name)
    log_file = open(log_name, "a")
    sys.stdout = log_file
    
    trj_dataset_train=Liflow_Dataset("train", device)
    trj_dataset_test=Liflow_Dataset("test", device)
    
    
    trj_train_loader=torch.utils.data.DataLoader(trj_dataset_train, shuffle=True, collate_fn=collate_fn)
    trj_test_loader=torch.utils.data.DataLoader(trj_dataset_test, shuffle=False, collate_fn=collate_fn)
    

    g=Dual_modal_trainer(x_in_dim=2935, p_in_dim=512).to(device)
    
    optimizer=torch.optim.Adam(g.parameters(), lr=1e-3)
    print("epoch", "train loss", "test 600K (log10MAE)", "test 800K (log10MAE)", "test 1000K (log10MAE)", "test 1200K (log10MAE)")
    for epoch in tqdm(range(EPOCH1)):
        loss_list=[]
        for p_list, node_attr, edge_index, edge_attr, temperature, s_idx, y_list in trj_train_loader:          
            e_x=embed_x(node_attr, edge_attr, edge_index, s_idx)
           
            loss=0
            for T in range(4):
                e_p=p_list[T]
                y=y_list[T]
                e_T=embed_T(temperature[T], s_idx, device)
                out1, out2=g(e_p, torch.cat([e_x,e_T],-1))
                loss=loss+torch.nn.functional.l1_loss(out1, y)+torch.nn.functional.l1_loss(out2, y)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        test_loss_list=[[],[],[],[]]
        with torch.no_grad():
            for p_list, node_attr, edge_index, edge_attr, temperature, s_idx, y_list in trj_test_loader:
                
                e_x=embed_x(node_attr, edge_attr, edge_index, s_idx)
            
                for T in range(4):
                    e_p=p_list[T].to(device)
                    y=y_list[T].to(device)
                    e_T=embed_T(temperature[T], s_idx, device)
                    out1, out2=g(e_p, torch.cat([e_x,e_T],-1))
                    loss=torch.nn.functional.l1_loss(out1, y)+torch.nn.functional.l1_loss(out2, y)
                    test_loss_list[T].append(loss.item())
        
        print(epoch, f"{np.mean(loss_list)*10:.3f}", f"{np.mean(test_loss_list[0])*10:.3f}", f"{np.mean(test_loss_list[1])*10:.3f}", f"{np.mean(test_loss_list[2])*10:.3f}", f"{np.mean(test_loss_list[3])*10:.3f}", flush=True)

    
    torch.save(g.cpu().state_dict(),"../model/g"+str(seed)+".pt")
    
    X_list, P_list=trj_dataset_train.get_lists()
    W=intial_value(X_list,P_list, g.encoder_x.W.to("cpu"), g.encoder_p.W.to("cpu"), list(range(len(X_list))), lam=1e-5)  
    
    pr=Predictor(x_in_dim=2935).to(device)
    pr.encoder_x.W=torch.nn.Parameter(W.detach().clone().to(device))
    pr.decoder.load_state_dict(g.decoder.state_dict())

    optimizer=torch.optim.Adam(pr.parameters(), lr=1e-5)

    for epoch in tqdm(range(EPOCH2)):
        loss_list=[]
        for _, node_attr, edge_index, edge_attr, temperature, s_idx, y_list in trj_train_loader:
            
            e_x=embed_x(node_attr, edge_attr, edge_index, s_idx)
           
            loss=0
            for T in range(4):
                y=y_list[T]
                e_T=embed_T(temperature[T], s_idx, device)
                out=pr(torch.cat([e_x,e_T],-1))
                loss=loss+torch.nn.functional.l1_loss(out, y)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        test_loss_list=[[],[],[],[]]
        with torch.no_grad():
            for _, node_attr, edge_index, edge_attr, temperature, s_idx, y_list in trj_test_loader:
                
                e_x=embed_x(node_attr, edge_attr, edge_index, s_idx)
            
                for T in range(4):
                    y=y_list[T].to(device)
                    e_T=embed_T(temperature[T], s_idx, device).to(device)
                    out=pr(torch.cat([e_x,e_T],-1))
                    loss=torch.nn.functional.l1_loss(out, y)
                    test_loss_list[T].append(loss.item())
        
        print(epoch, f"{np.mean(loss_list)*10:.3f}", f"{np.mean(test_loss_list[0])*10:.3f}", f"{np.mean(test_loss_list[1])*10:.3f}", f"{np.mean(test_loss_list[2])*10:.3f}", f"{np.mean(test_loss_list[3])*10:.3f}", flush=True)
    
    torch.save(pr.cpu().state_dict(),"../model/pr"+str(seed)+".pt")

    
    sys.stdout = sys.__stdout__  
    log_file.close()
    
    




