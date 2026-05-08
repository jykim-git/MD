import torch

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W=torch.nn.Parameter(torch.randn(in_dim, out_dim))

    def forward(self, input):
        return input @ self.W    

class Decoder(torch.nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        
        self.linear1=torch.nn. Linear(in_dim,hidden_dim)
        self.gate1=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(hidden_dim,hidden_dim)
        self.gate2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(hidden_dim,hidden_dim)
        self.gate3=torch.nn.ReLU()
        self.linear4=torch.nn.Linear(hidden_dim,out_dim)
        
                      
    def forward(self, input) :
        out=self.gate1(self.linear1(input))
        out=self.gate2(self.linear2(out))
        out=self.gate3(self.linear3(out))
        out=self.linear4(out)
            
        return out 
    
class Dual_modal_trainer(torch.nn.Module):
    def __init__(self,  x_in_dim, p_in_dim, decoder_hidden_dim=4000, out_dim=1, decoder_in_dim=8):
        super().__init__()
        self.encoder_p=Encoder(p_in_dim, decoder_in_dim)
        self.encoder_x=Encoder(x_in_dim, decoder_in_dim)
        self.norm=torch.nn.LayerNorm(decoder_in_dim, elementwise_affine=False, bias=False)
        self.decoder=Decoder(decoder_in_dim, decoder_hidden_dim, out_dim)
    def forward(self,p,x):
        h_p=self.encoder_p(p)
        h_x=self.encoder_x(x)
        h=h_p+h_x
        h=self.norm(h)
        h=torch.mean(h, 0, keepdim=True)
        h_x=self.norm(h_x)
        h_x=torch.mean(h_x, 0, keepdim=True)
        out1=self.decoder(h)
        out2=self.decoder(h_x)
        return out1, out2

class Predictor(torch.nn.Module):
    def __init__(self, x_in_dim, decoder_hidden_dim=4000, out_dim=1, decoder_in_dim=8):
        super().__init__()
        self.encoder_x=Encoder(x_in_dim, decoder_in_dim)
        self.norm=torch.nn.LayerNorm(decoder_in_dim, elementwise_affine=False, bias=False)
        self.decoder=Decoder(decoder_in_dim, decoder_hidden_dim, out_dim)
    def forward(self,x):
        h=self.encoder_x(x)
        h=self.norm(h)
        h=torch.mean(h, 0, keepdim=True)
        out=self.decoder(h)
        return out

