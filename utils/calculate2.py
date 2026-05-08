def calculate2(self, atoms=None, properties=None, system_changes=all_changes):
        is_ts_type = isinstance(self.model, torch_script_type)

        # call parent class to set necessary atom attributes
        Calculator.calculate(self, atoms, properties, system_changes)
        
        data = AtomGraphData.from_numpy_dict(
            unlabeled_atoms_to_graph(atoms, self.cutoff, with_shift=is_ts_type)
        )
        



        data.to(self.device)  # type: ignore

        

        x=[]
        edge_index=[]
        edge_embedding=[]
        edge_attr=[]
        
        def hook_fn(model, input, output):
            
            x.append(output.clone().x.detach())
            edge_index.append(output.clone().edge_index.detach())
            edge_embedding.append(output.clone().edge_embedding.detach())
            edge_attr.append(output.clone().edge_attr.detach())
           
        handles=[]
        for i, module in enumerate(self.model):
            if i==10:
                handles.append(module.register_forward_hook(hook_fn))

  
        self.model(data)

        self.results={}
        
        self.results["x"]=x[0]
        self.results["edge_index"]=edge_index[0]
        self.results["edge_embedding"]=edge_embedding[0]
        self.results["edge_attr"]=edge_attr[0]
        for handle in handles:
            handle.remove()
  