import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss


#from torchmetrics import TotalVariation
#tv = TotalVariation()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_f(x = 'x', y = 'y', expr = 'x * y', nu = 1):
    from sympy import lambdify, Derivative, Symbol, simplify
    
    x = Symbol('x')
    y = Symbol('y')
    u = expr.replace('torch.','')
    u = simplify(u)
    ux = Derivative(u, x, evaluate = True)
    uxx = Derivative(ux, x, evaluate = True)
    uy = Derivative(u, y, evaluate = True)
    uyy = Derivative(uy, y, evaluate = True)
    f4 = u * ux + nu * (uxx + uyy)
    f=lambdify([x,y],f4, "numpy")
    return f



def helmholtz_3d_source_term(x, y, z, a1, a2, a3, coefficient=1):

    u_gt = torch.sin(a1*torch.pi*y) * torch.sin(a2*torch.pi*x) * torch.sin(a3*torch.pi*z)

    u_yy = -(a1*torch.pi)**2 * u_gt

    u_xx = -(a2*torch.pi)**2 * u_gt

    u_zz = -(a3*torch.pi)**2 * u_gt

    return  u_yy + u_xx + u_zz + coefficient*u_gt



class Loss_Functions():
    def __init__(self, model, name = 'NS', params = {}, loss_type = 'mse') -> None:
        self.name = name
        self.model = model
        self.params = params
        
        if loss_type == 'mse':
            self.loss_type = mse_loss
        elif loss_type == 'logcosh':
            self.loss_type = smooth_l1_loss

        if 'Re' not in params.keys():
            raise ValueError('Re must be specified for NS')
        self.Re = self.params['Re']
        self.original_Re = self.params['Re']


    def pde_loss(self, x, y , z = None, return_pde_terms = False):

        S = self.model(x,y)

        u = S[:,0].reshape(-1,1)
        v = S[:,1].reshape(-1,1)
        p = S[:,2].reshape(-1,1)
        

        u_x = torch.autograd.grad(u, x, torch.ones_like(u), True, True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), True, True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), True, True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), True, True)[0]

        v_x = torch.autograd.grad(v, x, torch.ones_like(v), True, True)[0]
        v_y = torch.autograd.grad(v, y, torch.ones_like(v), True, True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), True, True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), True, True)[0]

        p_x = torch.autograd.grad(p, x, torch.ones_like(p), True, True)[0]
        p_y = torch.autograd.grad(p, y, torch.ones_like(p), True, True)[0]
        

        diffusion_u = 1/100  * (u_xx + u_yy)
        diffusion_v = 1/100  * (v_xx + v_yy)

        NSu = (u * u_x + v * u_y + p_x - diffusion_u)
        NSv = (u * v_x + v * v_y + p_y - diffusion_v)

        cont = u_x + v_y

        pde_loss = self.loss_type(NSu, torch.zeros_like(NSu)) +  self.loss_type(NSv, torch.zeros_like(NSv)) + self.loss_type(cont, torch.zeros_like(cont))#+ w_fpde * f_pde

        if return_pde_terms:
            abs_res = torch.abs(NSu) + torch.abs(NSv)
            return abs_res
            

        return pde_loss
    

    def bc_loss_nop(self, x, y, u_bc, normalize = False):

        S = self.model(x,y)
        psi = S[...,0].reshape(-1,1)
        p = S[...,1].reshape(-1,1)
        u = torch.autograd.grad(psi, y, torch.ones_like(psi), True, True)[0]
        v = -1*torch.autograd.grad(psi, x, torch.ones_like(psi), True, True)[0]
        uv = torch.cat([u,v], axis = -1)

        if normalize:
            diff = torch.square(uv - u_bc)
            diff2 = (diff - diff.min(axis = 0)[0])/(diff.max(axis = 0)[0] - diff.min(axis = 0)[0]) 
            mse_bc = torch.mean(diff2) 
            return mse_bc

        #mse_bc = torch.mean(torch.square(uv - u_bc)) 
        mse_bc = self.loss_type(uv, u_bc)
        return mse_bc

    def bc_loss_one_p(self, x, y, u_bc, normalize = False):

        S = self.model(x,y)
        p = S[:,2].reshape(-1,1)
        u =  S[:,0].reshape(-1,1)
        v =  S[:,1].reshape(-1,1)
        uv = torch.cat([u,v], axis = -1)

        x, y = torch.tensor(0.0).to(device).reshape(-1,1), torch.tensor(0.0).to(device).reshape(-1,1)
        Sp = self.model(x, y)

        mse_p = torch.mean(torch.square(Sp[:,2].reshape(-1,1)))

        if normalize:
            diff = torch.square(uv - u_bc[:,:2])
            diff2 = (diff - diff.min(axis = 0)[0])/(diff.max(axis = 0)[0] - diff.min(axis = 0)[0]) 
            mse_bc = torch.mean(diff2) 
            return mse_bc

        mse_bc = self.loss_type(uv, u_bc[:,:2]) #torch.mean(torch.square(uv - u_bc[:,:2]))

        return mse_bc + mse_p
    
    def data_loss(self, x, y, u_data):

        S  = self.model(x,y)
        psi = S[:,0].reshape(-1,1)
        p = S[:,1].reshape(-1,1)
        u = torch.autograd.grad(psi, y, torch.ones_like(psi), True, True)[0]
        v = -1*torch.autograd.grad(psi, x, torch.ones_like(psi), True, True)[0]
        #uv = torch.cat([u,v], axis = -1)
        uvp = torch.cat([u,v,p], axis = -1)
        mse_data = torch.mean(torch.square(uvp - u_data)) 

        return mse_data

    def MSE_loss(self, x, y, u_bc):

        S = self.model(x,y).reshape(-1,1)
        mse_data = torch.mean(torch.square(S - u_bc)) 

        return mse_data

