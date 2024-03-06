import torch
import torch.nn as nn
from utils.grid_sample import grid_sample_2d

dtype = torch.float32
device = torch.device("cuda:0")



class Encoder (nn.Module):

    def __init__(self, n_features = 4, res = [8, 12, 20, 18, 32], n_cells = 2, domain_size = [0, 1, 0, 1],  mode = 'bilinear') -> None:
        super().__init__()
        self.n_features = n_features 
        self.mode = mode
        self.res = res
        self.n_cells = n_cells

        
        self.F_active = nn.ParameterList([torch.rand(size=(self.n_cells, self.n_features, self.res[i], self.res[i]),
             dtype=dtype , device = device).data.uniform_(-1e-5,1e-5) for i in range(len(self.res))])

        # self.F_active = nn.ParameterList([torch.rand(size=(self.n_cells, self.n_features, self.res[i], self.res[i]),
        #      dtype=dtype , device = device) for i in range(len(self.res))])

        # for f in self.F_active:
        #     nn.init.xavier_normal_(f)

        self.domain_size = domain_size
        #self.active = nn.SiLU()


    def forward(self, x, y):
        features  = []
        x_min, x_max, y_min, y_max = self.domain_size
        x = (x - x_min)/(x_max - x_min)
        y = (y - y_min)/(y_max - y_min)
        x = x*2-1
        y = y*2-1

        x = torch.cat([x, y], dim=-1).unsqueeze(0).unsqueeze(0)


        for idx , alpha in enumerate(self.F_active):
            x3 = x.repeat([self.n_cells,1,1,1])
            #alpha = Gaussian_average(alpha)
            F = grid_sample_2d(alpha, x3, step= self.mode, offset=True)
            #F = CosineSampler2d.apply(alpha, x, 'zeros', True, 'cosine', True)
            features.append(F.sum(0).view(self.n_features,-1).t())
        
        # sum_level = sum(self.res[i]**2 for i in range(len(self.res)))
        # F = sum(self.res[len(self.res)-1-i]**2.0/sum_level * features[i] for i in range(len(self.res))) #*  sum(tuple(features))
        F = torch.cat(tuple(features) , 1)
        return F
    

class Encoder_conv2(nn.Module):

    def __init__(self, n_features = 4, res = [8, 12, 20, 18, 32], n_cells = 2, domain_size = [0, 1, 0, 1] , mode = 'bilinear') -> None:
        super().__init__()
        self.mode = mode
        self.n_features = n_features 
        self.res = res
        self.n_cells = n_cells



        a = [torch.rand(size=(self.n_cells, self.n_features, self.res[i], self.res[i]),
             dtype=dtype).to(device).data.uniform_(-1e-5,1e-5) for i in range(len(self.res))]
        

        #b = [torch.empty(size = (self.n_cells[0], self.n_features, self.res[0], self.res[0]), dtype=dtype, device=device) ]
             
        self.F_active = nn.ParameterList(a)
        #self.beta = {}
        self.domain_size = domain_size


        self.conv_layer = nn.Conv2d(in_channels=self.n_features , out_channels=self.n_features,
                                               groups = self.n_features ,kernel_size=3, padding=1, bias=False).to(device)
        
        #self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2).to(device)

    def forward(self, x, y):
        features  = []
        x_min, x_max, y_min, y_max = self.domain_size
        x = (x - x_min)/(x_max - x_min)
        y = (y - y_min)/(y_max - y_min)
        x = x*2-1
        y = y*2-1

        x = torch.cat([x, y], dim=-1).unsqueeze(0).unsqueeze(0)
        x = x.repeat([self.n_cells,1,1,1])




        for idx , alpha in enumerate(self.F_active):

            #print(f"alpha1 shape {alpha.shape}")
            

            #print(f"alpha0 shape {alpha.shape}")
            alpha = self.conv_layer(alpha)
            self.beta = alpha
            #print(f"alpha1 shape {alpha.shape}")
            #alpha = self.max_pool(alpha)
            #print(f"alpha shape after max pool{alpha.shape}")
            alpha = nn.Tanh()(alpha)
           

            F = grid_sample_2d(alpha, x, step=self.mode, offset=True)
            #print(f"pixel shape {F.shape}")
            dim = alpha.shape[1]
            features.append(F.sum(0).view(dim,-1).t())

        
        F = torch.cat(tuple(features) , 1)
        #F_ = torch.cat([F  , boundary_f] , 0)
        #print(F_.shape)
        return F#_

if __name__ == '__main__':
    encoder = Encoder()
    features = encoder(torch.rand(1,1).to(device), torch.rand(1,1).to(device))
    print(features)
    print(features.shape)