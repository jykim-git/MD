import torch
import pickle
import pandas as pd
import numpy as np
from model.embedding import embed_x, embed_T
class Liflow_Dataset(torch.utils.data.Dataset):
    def __init__(self, split, device):
           
            self.device=device
            with open("../data/liflow_node_attr_"+split+".pkl", "rb") as f:
                self.node_attr=pickle.load(f)
            with open("../data/liflow_edge_index_"+split+".pkl", "rb") as f:
                self.edge_index=pickle.load(f)
            with open("../data/liflow_edge_attr_"+split+".pkl", "rb") as f:
                self.edge_attr=pickle.load(f)
            self.temperature=[600, 800, 1000, 1200]
            with open("../data/liflow_s_idx_"+split+".pkl", "rb") as f:
                self.s_idx=pickle.load(f)
            with open("../data/liflow_y_600K_"+split+".pkl", "rb") as f:
                self.y1=torch.log10(torch.FloatTensor(pickle.load(f))).to(device)
            with open("../data/liflow_y_800K_"+split+".pkl", "rb") as f:
                self.y2=torch.log10(torch.FloatTensor(pickle.load(f))).to(device)
            with open("../data/liflow_y_1000K_"+split+".pkl", "rb") as f:
                self.y3=torch.log10(torch.FloatTensor(pickle.load(f))).to(device)
            with open("../data/liflow_y_1200K_"+split+".pkl", "rb") as f:
                self.y4=torch.log10(torch.FloatTensor(pickle.load(f))).to(device)

           
            with open("../data/liflow_p_"+split+"_600K.pkl", "rb") as f:
                     self.p1=(pickle.load(f))
            with open("../data/liflow_p_"+split+"_800K.pkl", "rb") as f:
                     self.p2=(pickle.load(f))
            with open("../data/liflow_p_"+split+"_1000K.pkl", "rb") as f:
                     self.p3=(pickle.load(f))
            with open("../data/liflow_p_"+split+"_1200K.pkl", "rb") as f:
                     self.p4=(pickle.load(f))
        
        
    def __len__(self):
        return len(self.y1)
    
    def __getitem__(self,idx):
        return [self.p1[idx].to(self.device), self.p2[idx].to(self.device),self.p3[idx].to(self.device),self.p4[idx].to(self.device)], self.node_attr[idx].to(self.device), self.edge_index[idx].to(self.device), self.edge_attr[idx].to(self.device), self.temperature, self.s_idx[idx], [self.y1[idx].reshape(1,1)/10,self.y2[idx].reshape(1,1)/10, self.y3[idx].reshape(1,1)/10, self.y4[idx].reshape(1,1)/10]
    def get_lists(self):
        X_list1=[torch.cat([embed_x(self.node_attr[idx], self.edge_attr[idx], self.edge_index[idx], self.s_idx[idx]), embed_T(self.temperature[0],self.s_idx[idx], "cpu")],-1) for idx in range(len(self.node_attr))]
        X_list2=[torch.cat([embed_x(self.node_attr[idx], self.edge_attr[idx], self.edge_index[idx], self.s_idx[idx]), embed_T(self.temperature[1],self.s_idx[idx], "cpu")],-1) for idx in range(len(self.node_attr))]
        X_list3=[torch.cat([embed_x(self.node_attr[idx], self.edge_attr[idx], self.edge_index[idx], self.s_idx[idx]), embed_T(self.temperature[2],self.s_idx[idx], "cpu")],-1) for idx in range(len(self.node_attr))]
        X_list4=[torch.cat([embed_x(self.node_attr[idx], self.edge_attr[idx], self.edge_index[idx], self.s_idx[idx]), embed_T(self.temperature[3],self.s_idx[idx], "cpu")],-1) for idx in range(len(self.node_attr))]
        X_list=X_list1+X_list2+X_list3+X_list4
        P_list=self.p1+self.p2+self.p3+self.p4
        return X_list, P_list


class Obelix_Dataset(torch.utils.data.Dataset):
    def __init__(self, split, device):

            with open("../data/obelix_node_attr_"+split+".pkl", "rb") as f:
                self.node_attr=pickle.load(f)
            with open("../data/obelix_edge_index_"+split+".pkl", "rb") as f:
                self.edge_index=pickle.load(f)
            with open("../data/obelix_edge_attr_"+split+".pkl", "rb") as f:
                self.edge_attr=pickle.load(f)
            self.temperature=[300]

            with open("../data/obelix_y_"+split+".pkl", "rb") as f:
                self.y=torch.log10(torch.FloatTensor(pickle.load(f))).to(device)
            with open("../data/obelix_s_idx_"+split+".pkl", "rb") as f:
                self.s_idx=pickle.load(f)

            
            self.device=device
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.node_attr[idx].to(self.device), self.edge_index[idx].to(self.device), self.edge_attr[idx].to(self.device), self.temperature, self.s_idx[idx], [self.y[idx].reshape(1,1)/10]
    


class Amorphous_T_Dataset(torch.utils.data.Dataset):
    def __init__(self, split, device):
         

            with open("../data/amorphous_T_node_attr_"+split+".pkl", "rb") as f:
                self.node_attr=pickle.load(f)
            with open("../data/amorphous_T_edge_index_"+split+".pkl", "rb") as f:
                self.edge_index=pickle.load(f)
            with open("../data/amorphous_T_edge_attr_"+split+".pkl", "rb") as f:
                self.edge_attr=pickle.load(f)
            with open("../data/amorphous_T_temp_"+split+".pkl", "rb") as f:
                self.temperature=pickle.load(f)
            with open("../data/amorphous_T_y_"+split+".pkl", "rb") as f:
                self.y=torch.log10(torch.FloatTensor(pickle.load(f))).to(device)

            with open("../data/amorphous_T_s_idx_"+split+".pkl", "rb") as f:
                self.s_idx=pickle.load(f)
            self.device=device
            


            
            
            
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.node_attr[idx].to(self.device), self.edge_index[idx].to(self.device), self.edge_attr[idx].to(self.device), [self.temperature[idx]], self.s_idx[idx], [self.y[idx].reshape(1,1)/10]