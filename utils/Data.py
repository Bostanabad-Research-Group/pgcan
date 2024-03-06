import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy.stats.qmc import Sobol
from utils.initialize import initialize


def get_data_spatial_lid_cavity(
        params = {'N_f': 50000, 'N_bc':50000, 'Re':100, 'domain_size':[0,1,0,1], 'ntest': 100, 'seed': 1234 , 'A':1}, train = True, use_data = False):

    dtype, device = initialize(seed = params['seed']) 
    domain_size = params['domain_size']
    N_f = params['N_f']
    N_bc = params['N_bc']
    seed = params['seed']
    ntest = params['ntest']
    Re = params['Re']


    # The exact solution
    def exact_solution(return_outs = False):
        DATA_np = np.loadtxt("./Data/A{}_mu100_sin.txt".format(params['A']))
        DATA_ = DATA_np[np.random.permutation(DATA_np.shape[0]), :]    #It has solutions for 90K points
        DATA = torch.from_numpy(DATA_[:params['N_f'],:6]).to(device).type(dtype) #x , y , u , v , p , U (the rest are derivatives computed by comsol)
        xc , yc = DATA[:,0:1] , DATA[:,1:2]
        xc_ , yc_ = DATA_[:,0:1] , DATA_[:,1:2]

        sol = DATA[:,2:6] #u,v
        sol_ = DATA_[:,2:6]


        yc.requires_grad = True
        xc.requires_grad = True


        b_data = []
        for i in range(sol_.shape[0]):
            if xc_[i,:] == 0. or xc_[i,:] == 1. or yc_[i,:] == 0. or yc_[i,:] == 1. :
                b_data.append([xc_[i,0] , yc_[i,0] , sol_[i,0] , sol_[i,1] , sol_[i,2] , sol_[i,3]])

        b_data = np.array(b_data)
        print(b_data.shape)
        #print(DATA_np.shape[0])
        DATA_b = torch.from_numpy(b_data).to(dtype).to(device)
        
                              
        #print(f"how many b points? {len(b_data)}")


        """

        DATA_bnp = np.loadtxt(r"./Data/cavity_boundary_data.txt")
        DATA_b = torch.from_numpy(DATA_bnp).to(device).type(dtype)

        """

        xb , yb = DATA_b[:,0:1] , DATA_b[:,1:2]
        yb.requires_grad = True
        xb.requires_grad = True
        sol_bc = DATA_b[:,2:5] #u,v
        

        if return_outs:
            u = DATA[:,2]
            v = DATA[:,3]
            p = DATA[:,4]
            U = DATA[:,5]
            return xc, yc, sol, xb, yb, sol_bc, u, v, p, U


        return xc, yc, sol, xb, yb, sol_bc

    # training
    
    xmin, xmax, ymin, ymax = domain_size
    Sobol1d = Sobol(d = 1, scramble=True, seed = seed)
    Sobol2d = Sobol(d = 2, scramble=True, seed = seed)
    if train == True:


        if use_data == False:
            x, y, sol, x_bc, y_bc, sol_bc = exact_solution(return_outs = False)

            sol_bc = sol_bc[:,[0,1]]
            
            return x, y, sol, x_bc, y_bc, sol_bc

        else:
            x_c, y_c, u_c, x_bc, y_bc, u_bc = exact_solution(return_outs = False)

            u_bc = u_bc[:,[0,1]]

            return x_c, y_c, u_c, x_bc, y_bc, u_bc, x_c[:10000,:], y_c[:10000 , :], u_c[:10000 , :3]
        






if __name__ == '__main__':
    from plot import plot_u
    domain_size = [0,1,0,1]
    data_params = {'N_f': 50000, 'N_bc':10000, 't': 1, 'nu':0.002, 
                'domain_size':domain_size, 'ntest': 100, 'seed': 12345, \
                    'sol':'(x*y  + x**3 + 2/(1+ x**2) + torch.sin(y*3)  - 3.0) * (x * y * (x-1)) * 15'}
