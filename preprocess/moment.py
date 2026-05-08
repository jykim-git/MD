import torch
from momentfm import MOMENTPipeline
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

def trajectory_embedding_liflow(device="cuda:0", data_dir="../../liflow_data/data/universal/"):
    positions_600K=np.load(data_dir+"positions_600K.npz",allow_pickle=True)
    positions_800K=np.load(data_dir+"positions_800K.npz",allow_pickle=True)
    positions_1000K=np.load(data_dir+"positions_1000K.npz",allow_pickle=True)
    positions_1200K=np.load(data_dir+"positions_1200K.npz",allow_pickle=True)

    #to load material id(name)
    df_train_600K=pd.read_csv(data_dir+"train_600K.csv") 
    df_test_600K=pd.read_csv(data_dir+"test_600K.csv")
    train_name_600K=df_train_600K["name"].tolist()
    test_name_600K=df_test_600K["name"].tolist()

    
    

    pipe = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small",
        model_kwargs={"task_name": "embedding"},
    ).to(device)

    pipe.init()
    
    with open("../data/liflow_s_idx_train.pkl", "rb") as f: #Li idx
        s_idx_train=pickle.load(f)

    #600K train set

    
    p_train=[]
    for i, mat_id in tqdm(enumerate(train_name_600K)):
        s=s_idx_train[i]
        
        x_enc= torch.FloatTensor(np.abs(np.fft.rfftn(positions_600K[mat_id].transpose(1,2,0)[s], [2501])))

        # to obtain trajectory embedding, you need: 
        # torch tensor of trajectory (N, 3, T)
               
        results = []
        total_size = x_enc.shape[0]
        for i in range(0, total_size, 100): #due to memory shortage, chunk data with size of 100 atoms 
                    end = i + 100
                    chunk_data = x_enc[i:end].to(device)
                    chunk_embedding= pipe(x_enc=chunk_data).embeddings.cpu()
                    results.append(chunk_embedding)
        p_train.append(torch.cat(results, dim=0))
    
    with open("../data/liflow_p_train_600K.pkl", "wb") as f:
        pickle.dump(p_train, f) 

    #800K train set

    p_train=[]
    for i, mat_id in tqdm(enumerate(train_name_600K)): # name is the same across T
        s=s_idx_train[i]
        x_enc= torch.FloatTensor(np.abs(np.fft.rfftn(positions_800K[mat_id].transpose(1,2,0)[s], [2501])))
        # to obtain trajectory embedding, you need: 
        # torch tensor of trajectory (N, 3, T)        
        results = []
        total_size = x_enc.shape[0]
        for i in range(0, total_size, 100): #due to memory shortage, chunk data with size of 100 atoms
                    end = i + 100
                    chunk_data = x_enc[i:end].to(device)
                    chunk_embedding= pipe(x_enc=chunk_data).embeddings.cpu()
                    results.append(chunk_embedding)
        p_train.append(torch.cat(results, dim=0))
    
    with open("../data/liflow_p_train_800K.pkl", "wb") as f:
        pickle.dump(p_train, f) 

    p_train=[]
    for i, mat_id in tqdm(enumerate(train_name_600K)): # name is the same across T
        s=s_idx_train[i]
        # to obtain trajectory embedding, you need: 
        # torch tensor of trajectory (N, 3, T)
        x_enc= torch.FloatTensor(np.abs(np.fft.rfftn(positions_1000K[mat_id].transpose(1,2,0)[s], [2501])))
                
        results = []
        total_size = x_enc.shape[0]
        for i in range(0, total_size, 100): #due to memory shortage, chunk data with size of 100 atoms
                    end = i + 100
                    chunk_data = x_enc[i:end].to(device)
                    chunk_embedding= pipe(x_enc=chunk_data).embeddings.cpu()
                    results.append(chunk_embedding)
        p_train.append(torch.cat(results, dim=0))
    
    with open("../data/liflow_p_train_1000K.pkl", "wb") as f:
        pickle.dump(p_train, f) 


    p_train=[]
    for i, mat_id in tqdm(enumerate(train_name_600K)): # name is the same across T
        s=s_idx_train[i]
        x_enc= torch.FloatTensor(np.abs(np.fft.rfftn(positions_1200K[mat_id].transpose(1,2,0)[s], [2501])))
        # to obtain trajectory embedding, you need: 
        # torch tensor of trajectory (N, 3, T)        
        results = []
        total_size = x_enc.shape[0]
        for i in range(0, total_size, 100): #due to memory shortage, chunk data with size of 100 atoms
                    end = i + 100
                    chunk_data = x_enc[i:end].to(device)
                    chunk_embedding= pipe(x_enc=chunk_data).embeddings.cpu()
                    results.append(chunk_embedding)
        p_train.append(torch.cat(results, dim=0))
    
    with open("../data/liflow_p_train_1200K.pkl", "wb") as f:
        pickle.dump(p_train, f) 

    
    with open("../data/liflow_s_idx_test.pkl", "rb") as f: #Li idx
        s_idx_test=pickle.load(f)

    #600K test set

    p_test=[]
    for i, mat_id in tqdm(enumerate(test_name_600K)):
        s=s_idx_test[i]
        x_enc= torch.FloatTensor(np.abs(np.fft.rfftn(positions_600K[mat_id].transpose(1,2,0)[s], [2501])))
        # to obtain trajectory embedding, you need: 
        # torch tensor of trajectory (N, 3, T)        
        results = []
        total_size = x_enc.shape[0]
        for i in range(0, total_size, 100): #due to memory shortage, chunk data with size of 100 atoms
                    end = i + 100
                    chunk_data = x_enc[i:end].to(device)
                    chunk_embedding= pipe(x_enc=chunk_data).embeddings.cpu()
                    results.append(chunk_embedding)
        p_test.append(torch.cat(results, dim=0))
    
    with open("../data/liflow_p_test_600K.pkl", "wb") as f:
        pickle.dump(p_test, f) 

    #800K test set

    p_test=[]
    for i, mat_id in tqdm(enumerate(test_name_600K)): # name is the same across T
        s=s_idx_test[i]
        x_enc= torch.FloatTensor(np.abs(np.fft.rfftn(positions_800K[mat_id].transpose(1,2,0)[s], [2501])))
        # to obtain trajectory embedding, you need: 
        # torch tensor of trajectory (N, 3, T)        
        results = []
        total_size = x_enc.shape[0]
        for i in range(0, total_size, 100): #due to memory shortage, chunk data with size of 100 atoms
                    end = i + 100
                    chunk_data = x_enc[i:end].to(device)
                    chunk_embedding= pipe(x_enc=chunk_data).embeddings.cpu()
                    results.append(chunk_embedding)
        p_test.append(torch.cat(results, dim=0))
    
    with open("../data/liflow_p_test_800K.pkl", "wb") as f:
        pickle.dump(p_test, f) 

    p_test=[]
    for i, mat_id in tqdm(enumerate(test_name_600K)): # name is the same across T
        s=s_idx_test[i]
        x_enc= torch.FloatTensor(np.abs(np.fft.rfftn(positions_1000K[mat_id].transpose(1,2,0)[s], [2501])))
        # to obtain trajectory embedding, you need: 
        # torch tensor of trajectory (N, 3, T)        
        results = []
        total_size = x_enc.shape[0]
        for i in range(0, total_size, 100): #due to memory shortage, chunk data with size of 100 atoms
                    end = i + 100
                    chunk_data = x_enc[i:end].to(device)
                    chunk_embedding= pipe(x_enc=chunk_data).embeddings.cpu()
                    results.append(chunk_embedding)
        p_test.append(torch.cat(results, dim=0))
    
    with open("../data/liflow_p_test_1000K.pkl", "wb") as f:
        pickle.dump(p_test, f) 


    p_test=[]
    for i, mat_id in tqdm(enumerate(test_name_600K)): # name is the same across T
        s=s_idx_test[i]
        x_enc= torch.FloatTensor(np.abs(np.fft.rfftn(positions_1200K[mat_id].transpose(1,2,0)[s], [2501])))
        # to obtain trajectory embedding, you need: 
        # torch tensor of trajectory (N, 3, T)        
        results = []
        total_size = x_enc.shape[0]
        for i in range(0, total_size, 100): #due to memory shortage, chunk data with size of 100 atoms
                    end = i + 100
                    chunk_data = x_enc[i:end].to(device)
                    chunk_embedding= pipe(x_enc=chunk_data).embeddings.cpu()
                    results.append(chunk_embedding)
        p_test.append(torch.cat(results, dim=0))
    
    with open("../data/liflow_p_test_1200K.pkl", "wb") as f:
        pickle.dump(p_test, f) 


    






