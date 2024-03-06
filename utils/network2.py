import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 2, layers = [16, 16], activation = 'tanh', encoder = 'cosine') -> None:
        super(Network, self).__init__()
        activation_list = {'tanh':nn.Tanh(), 'Silu':nn.SiLU(), 'Sigmoid':nn.Sigmoid()}
        activation = activation_list[activation]
        l = nn.ModuleList()
        l.append(nn.Linear(input_dim, layers[0]))
        l.append(activation)
        for i in range(0,len(layers)-1):
            l.append(nn.Linear(layers[i], layers[i+1]))
            l.append(activation )
        l.append(nn.Linear(layers[-1], output_dim, bias=True))
        self.layers = nn.Sequential(*l).to('cuda')

    def forward(self, input):
        out = self.layers[0](input)
        for layer in self.layers[1:]:
            out = layer(out)

        #idx = torch.arange(0,input.shape[0] , 21).to(torch.int64)

        #O= out[idx , :]


        return out
    
class NetworkM4(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 2, layers = [40 , 40 , 40 , 40 ], activation = 'tanh', encoder = 'cosine') -> None:
        super(NetworkM4, self).__init__()
        activation_list = {'tanh':nn.Tanh(), 'Silu':nn.SiLU(), 'Sigmoid':nn.Sigmoid()}
        activation = activation_list[activation]
        neuron = layers[0]
        self.U = nn.Linear(input_dim, neuron).to('cuda')
        self.V = nn.Linear(input_dim, neuron).to('cuda')
        self.H1 = nn.Linear(input_dim, neuron).to('cuda')
        self.last= nn.Linear(neuron, output_dim).to('cuda')

        l = nn.ModuleList()
        for i in range(len(layers)):
            l.append(nn.Linear(neuron, neuron))
            #l.append(activation )
        #l.append(nn.Linear(layers[-1], output_dim, bias=True))
        self.layers = nn.Sequential(*l).to('cuda')


        print(self.layers)

    def forward(self, input):
        
        U = nn.Tanh()(self.U(input))
        V = nn.Tanh()(self.V(input))
        H = nn.Tanh()(self.H1(input))
        #out = 



        #out = self.layers[0](input)
        for layer in self.layers:

            Z = nn.Tanh()(layer(H))
            H = (1-Z)*U + Z*V
            #out = layer(out)
        
        out = self.last(H)

        #idx = torch.arange(0,input.shape[0] , 21).to(torch.int64)

        #O= out[idx , :]


        return out
    

class NetworkM4_fused(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 2, layers = [40 , 40 , 40 , 40 ], activation = 'tanh', encoder = 'cosine') -> None:
        super(NetworkM4_fused, self).__init__()
        activation_list = {'tanh':nn.Tanh(), 'Silu':nn.SiLU(), 'Sigmoid':nn.Sigmoid()}
        activation = activation_list[activation]
        #self.U = nn.Linear(input_dim, 40).to('cuda')
        #self.V = nn.Linear(input_dim, 40).to('cuda')
        self.H1 = nn.Linear(input_dim, layers[0]).to('cuda')
        self.last= nn.Linear(layers[0], output_dim).to('cuda')

        l = nn.ModuleList()
        for i in range(len(layers)):
            l.append(nn.Linear(layers[i], layers[i]))
            #l.append(activation )
        #l.append(nn.Linear(layers[-1], output_dim, bias=True))
        self.layers = nn.Sequential(*l).to('cuda')

    def forward(self, input , features):

        #features (N,256)
        F = int(features.shape[1]/2)

        U = features[:,:F]

        #print(f" U shape: {U.shape}")
        V = features[:,F:]
        
        #U = nn.Tanh()(self.U(input))
        #V = nn.Tanh()(self.V(input))
        H = nn.Tanh()(self.H1(input))
        #out = 



        #out = self.layers[0](input)
        for layer in self.layers:

            Z = nn.Tanh()(layer(H))
            H = (1-Z)*U + Z*V
            #out = layer(out)
        
        out = self.last(H)

        #idx = torch.arange(0,input.shape[0] , 21).to(torch.int64)

        #O= out[idx , :]


        return out
 
 


if __name__ == '__main__':
    net = Network()
    print(net.layers)