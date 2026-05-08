import numpy as np
import pandas as pd
from tqdm import tqdm
from ase import Atoms
from ase.io import read
from sevenn.calculator import SevenNetCalculator
import torch
import pickle

def structure_liflow(device="cuda:0", data_dir="../../liflow_data/data/universal/"):
    
    calculator=SevenNetCalculator("7net-0", device=device)
    
    atomic_numbers=np.load(data_dir+"atomic_numbers.npy",allow_pickle=True).tolist()
    
    positions_600K=np.load(data_dir+"positions_600K.npz",allow_pickle=True)
    
    lattice=np.load(data_dir+"lattice.npy",allow_pickle=True).tolist()
    
    #to load MSD & material id(name)
    df_train_600K=pd.read_csv(data_dir+"train_600K.csv") 
    df_test_600K=pd.read_csv(data_dir+"test_600K.csv")
    y_train_600K=df_train_600K["msd_t_Li"].tolist() #list of Li MSD at final step
    y_test_600K=df_test_600K["msd_t_Li"].tolist() #list of Li MSD at final step
    train_name_600K=df_train_600K["name"].tolist()
    test_name_600K=df_test_600K["name"].tolist()

    df_train_800K=pd.read_csv(data_dir+"train_800K.csv") 
    df_test_800K=pd.read_csv(data_dir+"test_800K.csv")
    y_train_800K=df_train_800K["msd_t_Li"].tolist() #list of Li MSD at final step
    y_test_800K=df_test_800K["msd_t_Li"].tolist() #list of Li MSD at final step

    df_train_1000K=pd.read_csv(data_dir+"train_1000K.csv") 
    df_test_1000K=pd.read_csv(data_dir+"test_1000K.csv")
    y_train_1000K=df_train_1000K["msd_t_Li"].tolist() #list of Li MSD at final step
    y_test_1000K=df_test_1000K["msd_t_Li"].tolist() #list of Li MSD at final step

    df_train_1200K=pd.read_csv(data_dir+"train_1200K.csv") 
    df_test_1200K=pd.read_csv(data_dir+"test_1200K.csv")
    y_train_1200K=df_train_1200K["msd_t_Li"].tolist() #list of Li MSD at final step
    y_test_1200K=df_test_1200K["msd_t_Li"].tolist() #list of Li MSD at final step


    #obtain structure embedding from sevennet (for train set)
    
    node_attr_train=[] #list of torch FloatTensors
    edge_index_train=[] # list of torch LongTensors 
    edge_attr_train=[] #list of torch FloatTensors
    s_idx_train=[] #list of list(Li index)
    

    for mat_id in tqdm(train_name_600K):
        # to obtain structure embedding, you need: 
        # 1) numpy array of atomic number (N),
        # 2) numpy array of atomic structure (N,3)
        # 3) numpy array of unit cell (3,3)
      
        atoms=Atoms(
                numbers=atomic_numbers[mat_id],
                positions=positions_600K[mat_id][0],
                cell=lattice[mat_id],
                pbc=True           
        )
        calculator.calculate2(atoms)

        result=calculator.results
        node_attr_train.append(result["x"].cpu())       
        edge_index_train.append(result["edge_index"].cpu())
        edge_attr_train.append(torch.concatenate([result["edge_embedding"],result["edge_attr"]],-1).cpu())
        s_idx_train.append((np.where(atomic_numbers[mat_id]==3)[0]).tolist())
                  
    
    #save all
    with open("../data/liflow_node_attr_train.pkl", "wb") as f:
        pickle.dump(node_attr_train, f)
    with open("../data/liflow_edge_index_train.pkl", "wb") as f:
        pickle.dump(edge_index_train, f)
    with open("../data/liflow_edge_attr_train.pkl", "wb") as f:
        pickle.dump(edge_attr_train, f)
    with open("../data/liflow_s_idx_train.pkl", "wb") as f:
        pickle.dump(s_idx_train, f)
    
    with open("../data/liflow_y_600K_train.pkl", "wb") as f:
        pickle.dump(y_train_600K, f)
    with open("../data/liflow_y_800K_train.pkl", "wb") as f:
        pickle.dump(y_train_800K, f)
    with open("../data/liflow_y_1000K_train.pkl", "wb") as f:
        pickle.dump(y_train_1000K, f)
    with open("../data/liflow_y_1200K_train.pkl", "wb") as f:
        pickle.dump(y_train_1200K, f)
    


    #obtain structure embedding from sevennet (for test set)
    
    node_attr_test=[] #list of torch FloatTensors
    edge_index_test=[] # list of torch LongTensors 
    edge_attr_test=[] #list of torch FloatTensors
    s_idx_test=[] #list of list(Li index)
    

    for mat_id in tqdm(test_name_600K):

        # to obtain structure embedding, you need: 
        # 1) numpy array of atomic number (N),
        # 2) numpy array of atomic structure (N,3)
        # 3) numpy array of unit cell (3,3)
         
        atoms=Atoms(
                numbers=atomic_numbers[mat_id],
                positions=positions_600K[mat_id][0],
                cell=lattice[mat_id],
                pbc=True           
        )
        calculator.calculate2(atoms)

        result=calculator.results
        node_attr_test.append(result["x"].cpu())       
        edge_index_test.append(result["edge_index"].cpu())
        edge_attr_test.append(torch.concatenate([result["edge_embedding"],result["edge_attr"]],-1).cpu())
        s_idx_test.append((np.where(atomic_numbers[mat_id]==3)[0]).tolist())
                  
    #save all
    with open("../data/liflow_node_attr_test.pkl", "wb") as f:
        pickle.dump(node_attr_test, f)
    with open("../data/liflow_edge_index_test.pkl", "wb") as f:
        pickle.dump(edge_index_test, f)
    with open("../data/liflow_edge_attr_test.pkl", "wb") as f:
        pickle.dump(edge_attr_test, f)
    with open("../data/liflow_s_idx_test.pkl", "wb") as f:
        pickle.dump(s_idx_test, f)
    
    with open("../data/liflow_y_600K_test.pkl", "wb") as f:
        pickle.dump(y_test_600K, f)
    with open("../data/liflow_y_800K_test.pkl", "wb") as f:
        pickle.dump(y_test_800K, f)
    with open("../data/liflow_y_1000K_test.pkl", "wb") as f:
        pickle.dump(y_test_1000K, f)
    with open("../data/liflow_y_1200K_test.pkl", "wb") as f:
        pickle.dump(y_test_1200K, f)


def structure_obelix(device="cuda:0", data_dir="../../obelix_data/"):
    
    calculator=SevenNetCalculator("7net-0", device=device)
    

    #obtain structure embedding from sevennet (for train set)
    df = pd.read_csv(data_dir+"train.txt")
    
    node_attr_train=[] #list of torch FloatTensors
    edge_index_train=[] # list of torch LongTensors 
    edge_attr_train=[] #list of torch FloatTensors
    s_idx_train=[] #list of list(Li index)
    y_train=[] #list of float(ionic conductivity)
    
    
    for _, row in df.iterrows():
        Cif_id = row["Cif ID"]

        y_string=str(row["Ionic conductivity (S cm-1)"])
        
        if (not y_string.startswith("<")) and ("available" not in str(Cif_id)) and (not pd.isna(Cif_id)):
            y=float(y_string)
            ID=row["ID"]
            atom=read(data_dir+"/train_cifs/train_randomized_cifs/"+ID+".cif")
            pos=atom.get_positions()
            atomic_number=atom.get_atomic_numbers()
            lattice=atom.get_cell()
            s_idx=np.where(atomic_number==3)[0]
            if len(s_idx)>0:         
                atoms=Atoms(
                        numbers=atomic_number,
                        positions=pos,
                        cell=lattice,
                        pbc=True           
                )
                calculator.calculate2(atoms)

                result=calculator.results
                node_attr_train.append(result["x"].cpu())       
                edge_index_train.append(result["edge_index"].cpu())
                edge_attr_train.append(torch.concatenate([result["edge_embedding"],result["edge_attr"]],-1).cpu())
                s_idx_train.append(s_idx.tolist())
                y_train.append(y)
                  
    
    #save all
    with open("../data/obelix_node_attr_train.pkl", "wb") as f:
        pickle.dump(node_attr_train, f)
    with open("../data/obelix_edge_index_train.pkl", "wb") as f:
        pickle.dump(edge_index_train, f)
    with open("../data/obelix_edge_attr_train.pkl", "wb") as f:
        pickle.dump(edge_attr_train, f)
    with open("../data/obelix_s_idx_train.pkl", "wb") as f:
        pickle.dump(s_idx_train, f)
    
    with open("../data/obelix_y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
 
      

    #obtain structure embedding from sevennet (for test set)
    df = pd.read_csv(data_dir+"test.txt")
    
    node_attr_test=[] #list of torch FloatTensors
    edge_index_test=[] # list of torch LongTensors 
    edge_attr_test=[] #list of torch FloatTensors
    s_idx_test=[] #list of list(Li index)
    y_test=[] #list of float(ionic conductivity)
    
    
    for _, row in df.iterrows():
        Cif_id = row["Cif ID"]
        y_string=str(row["Ionic conductivity (S cm-1)"])
        
        if (not y_string.startswith("<")) and ("available" not in str(Cif_id)) and (not pd.isna(Cif_id)):

            y=float(y_string)
            ID=row["ID"]
            atom=read(data_dir+"/test_cifs/test_randomized_cifs/"+ID+".cif")
            pos=atom.get_positions()
            atomic_number=atom.get_atomic_numbers()
            lattice=atom.get_cell()
            s_idx=np.where(atomic_number==3)[0]
            if len(s_idx)>0:         
                atoms=Atoms(
                        numbers=atomic_number,
                        positions=pos,
                        cell=lattice,
                        pbc=True           
                )
                calculator.calculate2(atoms)

                result=calculator.results
                node_attr_test.append(result["x"].cpu())       
                edge_index_test.append(result["edge_index"].cpu())
                edge_attr_test.append(torch.concatenate([result["edge_embedding"],result["edge_attr"]],-1).cpu())
                s_idx_test.append(s_idx.tolist())
                y_test.append(y)
                  
    
    #save all
    with open("../data/obelix_node_attr_test.pkl", "wb") as f:
        pickle.dump(node_attr_test, f)
    with open("../data/obelix_edge_index_test.pkl", "wb") as f:
        pickle.dump(edge_index_test, f)
    with open("../data/obelix_edge_attr_test.pkl", "wb") as f:
        pickle.dump(edge_attr_test, f)
    with open("../data/obelix_s_idx_test.pkl", "wb") as f:
        pickle.dump(s_idx_test, f)
    
    with open("../data/obelix_y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)





def structure_amorphous_T(device="cuda:0", data_dir="../data/"):
    
    calculator=SevenNetCalculator("7net-0", device=device)
    

    #obtain structure embedding from sevennet (for train set)
    
    with open(data_dir+"amorphous_T_pos_train.pkl", "rb") as f:
        pos_train=pickle.load(f)
    with open(data_dir+"amorphous_T_atomic_number_train.pkl", "rb") as f:
        atomic_number_train=pickle.load(f)
    with open(data_dir+"amorphous_T_lattice_train.pkl", "rb") as f:
        lattice_train=pickle.load(f)
    
    
    
    
    node_attr_train=[] #list of torch FloatTensors
    edge_index_train=[] # list of torch LongTensors 
    edge_attr_train=[] #list of torch FloatTensors
    
    
    
    for i in tqdm(range(len(pos_train))):
                pos=np.array(pos_train[i])
                atomic_number=np.array(atomic_number_train[i])
                lattice=np.array(lattice_train[i])

                atoms=Atoms(
                        numbers=atomic_number,
                        positions=pos,
                        cell=lattice,
                        pbc=True           
                )
                calculator.calculate2(atoms)

                result=calculator.results
                node_attr_train.append(result["x"].cpu())       
                edge_index_train.append(result["edge_index"].cpu())
                edge_attr_train.append(torch.concatenate([result["edge_embedding"],result["edge_attr"]],-1).cpu())
                
                  
    
    #save all
    with open("../data/amorphous_T_node_attr_train.pkl", "wb") as f:
        pickle.dump(node_attr_train, f)
    with open("../data/amorphous_T_edge_index_train.pkl", "wb") as f:
        pickle.dump(edge_index_train, f)
    with open("../data/amorphous_T_edge_attr_train.pkl", "wb") as f:
        pickle.dump(edge_attr_train, f)
    
 
      

    #obtain structure embedding from sevennet (for test set)
    
    with open(data_dir+"amorphous_T_pos_test.pkl", "rb") as f:
        pos_test=pickle.load(f)
    with open(data_dir+"amorphous_T_atomic_number_test.pkl", "rb") as f:
        atomic_number_test=pickle.load(f)
    with open(data_dir+"amorphous_T_lattice_test.pkl", "rb") as f:
        lattice_test=pickle.load(f)
    
    
    
    
    node_attr_test=[] #list of torch FloatTensors
    edge_index_test=[] # list of torch LongTensors 
    edge_attr_test=[] #list of torch FloatTensors
    
    
    
    for i in tqdm(range(len(pos_test))):
                pos=np.array(pos_test[i])
                atomic_number=np.array(atomic_number_test[i])
                lattice=np.array(lattice_test[i])

                atoms=Atoms(
                        numbers=atomic_number,
                        positions=pos,
                        cell=lattice,
                        pbc=True           
                )
                calculator.calculate2(atoms)

                result=calculator.results
                node_attr_test.append(result["x"].cpu())       
                edge_index_test.append(result["edge_index"].cpu())
                edge_attr_test.append(torch.concatenate([result["edge_embedding"],result["edge_attr"]],-1).cpu())
                
                  
    
    #save all
    with open("../data/amorphous_T_node_attr_test.pkl", "wb") as f:
        pickle.dump(node_attr_test, f)
    with open("../data/amorphous_T_edge_index_test.pkl", "wb") as f:
        pickle.dump(edge_index_test, f)
    with open("../data/amorphous_T_edge_attr_test.pkl", "wb") as f:
        pickle.dump(edge_attr_test, f)
