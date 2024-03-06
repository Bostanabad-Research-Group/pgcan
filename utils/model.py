import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, network, encoder = None ,hbc = False , data_params = None) -> None:
        super().__init__()
        self.pinns = True
        self.network = network
        self.encoder = encoder
        self.hbc = hbc
        self.data_params = data_params

        if encoder is not None:
            self.encoder = encoder
            self.pinns = False

        
    def forward(self, x, y , z = None):
        if z == None:
            out = torch.cat([x, y], dim=-1)
        else:
            out = torch.cat([x, y,z], dim=-1)

        if self.pinns:

            out = self.network(out)
 
        elif self.encoder != None:
            if z == None:

                out = self.encoder(x, y)


            else:
                out = self.encoder(x, y,z)

            if self.network is not None:
                out = self.network(out)



        return out 
    

class Model_M4fused(nn.Module):
    
    def __init__(self, network, encoder = None  , data_params = None) -> None:
        super().__init__()
        self.pinns = True
        self.network = network
        self.encoder = encoder

        self.data_params = data_params

        if encoder is not None:
            self.encoder = encoder
            self.pinns = False

        
    def forward(self, x, y , z = None) :
        if z == None:
            X = torch.cat([x, y], dim=-1)
        else:
            X = torch.cat([x, y , z], dim=-1)
        if self.pinns:
            out = self.network(X)
            u_ = out 
        else:
            if z == None:
                out = self.encoder(x, y)
            else:
                out = self.encoder(x, y , z)

            if self.network is not None:
                out = self.network(input = X , features = out)

                u_ = out

        return out #, u_
    
