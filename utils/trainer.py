import torch
import torch.nn as nn
import time
from torch.utils.data import TensorDataset, DataLoader
from utils.plot import  compare_plot
import numpy as np
import pandas as pd
import os
import time

class Trainer(nn.Module):
    def __init__(
        self, 
        model, 
        pde_loss_function = None,
        bc_loss_function = None, 
        train_size = 0.8,
        epochs = 50, 
        lr = 1e-3,
        optimizer = 'adam',
        data_params = {'N_f': 50000, 'N_bc':50000, 'N_d':1000, 't':1.0, 'nu':0.01, 'domain_size':[0,1,0,1], 'ntest': 100, 'seed': 1234, 'Neuman_bc': True}, 
        data_func = None, 
        use_data = False,
        loss_functions = None,
        dynamic_weights = False, 
        device = 'cuda',
        title = 'default',
   
        ) -> None:
        super().__init__()
        
        #print(optimizer, type(optimizer))
        self.optim_type = optimizer
        self.model = model
        self.pde_loss_function = pde_loss_function
        self.bc_loss_function = bc_loss_function
        self.loss_functions = loss_functions
        self.dynamic_weights = dynamic_weights
        self.alpha = 1.0
        self.beta = 1.0
        self.lambdaa = 0.9
        self.device = device
        self.coeff_bc = 1.0
        self.coeff_pde = 1.0
        self.title = title
        self.data_params = data_params
        self.data_func = data_func
        self.lr = lr



        self.optim = None


        self.use_data = use_data
        self.train_size = train_size
        self.neuman_bc = False


        x_c, y_c, u_c, x_bc, y_bc, u_bc = data_func(data_params, train = True)
        size_bc = int(len(x_bc) * self.train_size)

        if use_data:
            x_c, y_c, u_c, x_bc, y_bc, u_bc, x_d, y_d, u_d = data_func(data_params, train = True, use_data = True)
            self.x_d, self.y_d, self.u_d = x_d, y_d, u_d

        
        size_c = int(len(x_c) * self.train_size)

        
        self.epochs = epochs
        self.xtrain_c,  self.ytrain_c,  self.utrain_c =  x_c[:size_c],  y_c[:size_c],  u_c[:size_c]
        self.xval_c,  self.yval_c,  self.uval_c =  x_c[size_c:],  y_c[size_c:],  u_c[size_c:]


        self.xtrain_bc,  self.ytrain_bc, self.utrain_bc =  x_bc[:size_bc],  y_bc[:size_bc] , u_bc[:size_bc]
        self.xval_bc,  self.yval_bc ,   self.uval_bc =  x_bc[size_bc:], y_bc[size_bc:],  u_bc[size_bc:]
    
        self.xtrain_c,  self.ytrain_c, self.utrain_c =  x_c[:size_c],  y_c[:size_c],  u_c[:size_c]
        self.xval_c,  self.yval_c ,self.uval_c =  x_c[size_c:],  y_c[size_c:],  u_c[size_c:]


        self.train_losses = {"bc": [], "pde": [], "data": [], "total":[] , "reconstruction" : [] , "ic":[]}
        self.val_losses = {"bc": [], "pde": [], "data": [], "total":[] , "reconstruction" : [] , 'ic':[]}
        self.gradient_loss_total = {'pde':[], 'bc':[]}
        self.gradient_loss_network = {'pde':[], 'bc':[]}
        self.gradient_loss_encoder = {'pde':[], 'bc':[]}
        self.title = title
        self.time_loss = []
        self.weights ={'alpha':[], 'beta':[]}

        self.count = 0


        if self.pde_loss_function is None:
            self.col_exist = False
        else:
            self.col_exist = True
        


        self.bc_loss = self.bc_loss_function(self.xtrain_bc, self.ytrain_bc, self.utrain_bc)
        self.ic_loss = torch.tensor([0.0], device = self.device, requires_grad=True)


        self.new_dic_col = self.data_params.copy()


        self.adam = torch.optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.adam, milestones= np.linspace(0,self.epochs,4).tolist(), gamma=0.1)
        self.iter = 0


    def update_data(self):
        self.data_params['seed'] = int(torch.randint(1000, 100000, (1,)))

        x_c, y_c, u_c, x_bc, y_bc, u_bc = self.data_func(self.data_params, train = True)
        size_bc = int(len(x_bc) * self.train_size)

        if self.use_data:
            x_c, y_c, u_c, x_bc, y_bc, u_bc, x_d, y_d, u_d = self.data_func(self.data_params, train = True, use_data = True)
            self.x_d, self.y_d, self.u_d = x_d, y_d, u_d


        size_c = int(len(x_c) * self.train_size)

        self.xtrain_c,  self.ytrain_c,  self.utrain_c =  x_c[:size_c],  y_c[:size_c],  u_c[:size_c]
        self.xval_c,  self.yval_c,  self.uval_c =  x_c[size_c:],  y_c[size_c:],  u_c[size_c:]
       
        self.xtrain_bc,  self.ytrain_bc, self.utrain_bc =  x_bc[:size_bc],  y_bc[:size_bc] , u_bc[:size_bc]
        self.xval_bc,  self.yval_bc ,   self.uval_bc =  x_bc[size_bc:], y_bc[size_bc:],  u_bc[size_bc:]
    
        self.xtrain_c,  self.ytrain_c, self.utrain_c =  x_c[:size_c],  y_c[:size_c],  u_c[:size_c]
        self.xval_c,  self.yval_c ,self.uval_c =  x_c[size_c:],  y_c[size_c:],  u_c[size_c:]


    def compute_dynamic_weights(self):

        delta_pde_teta = torch.autograd.grad(self.pde_loss, self.model.parameters(),  retain_graph=True, allow_unused=True)
        values = [p.reshape(-1,).cpu().tolist() for p in delta_pde_teta if p is not None]
        delta_pde_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

        delta_bc_teta = torch.autograd.grad(self.bc_loss, self.model.parameters(),  retain_graph=True, allow_unused=True)
        values = [p.reshape(-1,).cpu().tolist() for p in delta_bc_teta if p is not None]
        delta_bc_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))


        if self.use_data:


            #print("yes")
            delta_data_teta = torch.autograd.grad(self.data_loss, self.model.parameters(),  retain_graph=True)
            values2 = [p.reshape(-1,).cpu().tolist() for p in delta_data_teta]
            delta_data_teta_abs = torch.abs(torch.tensor([v for val in values2 for v in val]))
            temp2 = torch.mean(delta_pde_teta_abs) / torch.mean(delta_data_teta_abs)
        else:
            temp2 = self.beta

        temp = torch.mean(delta_pde_teta_abs) / torch.mean(delta_bc_teta_abs)

        
        return (1.0 - self.lambdaa) * self.alpha + self.lambdaa * temp, (1.0 - self.lambdaa) * self.beta + self.lambdaa * temp2
    

    def closure(self):

        self.adam.zero_grad()
        self.dp_loss = torch.tensor([0.0], device = self.device, requires_grad=True)


        self.neuman_bc_loss = torch.tensor([0.0], device = self.device, requires_grad=True)


        self.data_loss = torch.tensor([0.0], device = self.device, requires_grad=True)

        self.bc_loss = self.bc_loss_function(self.xtrain_bc, self.ytrain_bc, self.utrain_bc)

        if self.col_exist and len(self.xtrain_c) > 0:

            self.pde_loss = self.pde_loss_function(self.xtrain_c, self.ytrain_c, return_pde_terms =False)

            #print(f'uux {uux.detach().cpu().numpy()} and diffusion {diff.detach().cpu().numpy()}')
        else:
            self.pde_loss = torch.tensor([0.0], device = self.device, requires_grad=True)


        self.data_p_loss = torch.tensor([0.0], device = self.device, requires_grad=True)

        self.gradient_loss_total['pde'].append(0.)#(torch.autograd.grad(self.pde_loss, self.model.parameters(), retain_graph = True))
        self.gradient_loss_total['bc'].append(0.)#(torch.autograd.grad(self.alpha * self.bc_loss, self.model.parameters(), retain_graph = True))


        if self.dynamic_weights:

            if self.iter%100 == 0:
                self.alpha, self.beta = self.compute_dynamic_weights()
                self.weights['alpha'].append(self.alpha)
                self.weights['beta'].append(self.beta)

            loss =  self.alpha * self.bc_loss + self.beta * 0.* self.data_loss + self.pde_loss + self.data_p_loss[0] + self.dp_loss[0] + self.neuman_bc_loss[0]
        else:
            m = nn.ReLU()
            loss = self.data_p_loss  + 1.* self.bc_loss + 1.  * self.pde_loss + 0. *self.data_loss + self.dp_loss + self.neuman_bc_loss


        loss.backward(retain_graph=True)
        ############################ Evaluate and save ############################################
        if self.col_exist:
            self.pde_val_loss = self.pde_loss
        else:
            self.pde_val_loss = torch.tensor([0.0], device = self.device, requires_grad=True)
        

        self.bc_val_loss = self.bc_loss#self.bc_loss_function(self.xval_bc, self.yval_bc, self.uval_bc)
        self.ic_val_loss = torch.tensor([0.0], device = self.device, requires_grad=True)

        self.data_p_loss = torch.tensor([0.0], device = self.device, requires_grad=True)
        self.neuman_bc_val_loss = torch.tensor([0.0], device = self.device, requires_grad=True)


        if self.dynamic_weights:
                self.loss_val = self.data_p_loss  + 1.* self.bc_val_loss + 1.  * self.pde_val_loss + self.neuman_bc_val_loss
        else:
                m = nn.ReLU()
                self.loss_val = self.data_p_loss  + 1.* self.bc_val_loss + 1.  * self.pde_val_loss + self.neuman_bc_val_loss
        return loss
    
##############################################################################
############################### FORWARD ######################################
##############################################################################

    def forward(self):

        Eigs = []
        Lines = []
        metric = []

        self.kernel_dict = {}
        self.mother_res = {}


        start = time.time()
        self.time = start

        test_params = self.data_params.copy()
        test_params['seed'] = 3
        x_test, y_test, u_test , _,_,_= self.data_func(params=test_params, train=True)

        self.model.train()
        

        for iter in range(self.epochs):
            self.iter = iter
            loss = self.closure()
            self.adam.step()
            self.scheduler.step()

            self.train_losses['pde'].append(self.pde_loss.detach().cpu().numpy().item())
            self.train_losses['bc'].append(self.bc_loss.detach().cpu().numpy().item())
            self.train_losses['ic'].append(self.ic_loss.detach().cpu().numpy().item())
            self.train_losses['total'].append(loss.data.cpu().numpy().item())
            self.val_losses['pde'].append(self.pde_val_loss.detach().cpu().numpy().item())
            self.val_losses['bc'].append(self.bc_val_loss.detach().cpu().numpy().item())
            self.val_losses['ic'].append(self.ic_val_loss.detach().cpu().numpy().item())
            self.val_losses['total'].append(self.loss_val.detach().cpu().numpy().item())

            temp = "{:.2e}".format((time.time() - self.time)/60)
            self.time_loss.append(temp)


            if self.iter % 10 == 0:
                print(f"\n\rEp: {self.iter} T Loss: {self.train_losses['total'][self.iter]:.5e} \
                    V Loss: {self.val_losses['total'][self.iter]:.5e} Etime: {temp} min ", end ="")
            
            if self.iter % 10 == 0 and self.iter > 0:
                text3 = f"\rEp: {self.iter} T Loss: {self.train_losses['total'][self.iter]:.5e} \
                    V Loss: {self.val_losses['total'][self.iter]:.5e} coeff_bc: {self.alpha} \
                        coeff_data: {self.beta}  Etime: {temp} min"
                Lines.append(text3)


            if iter % 100 == 0:
                self.update_data()


            if  (self.iter+1)%(int(self.epochs/5)) == 0:
                torch.save(self.model, f'./Saved_Models/Mtrained_epoch{self.iter}_'+self.title+'.pt')


            if  (self.iter+1)%(int(self.epochs/10))== 0 and self.iter>0:

                end = time.time()
                ET = end - start

                u_predict = self.model(x_test,y_test)
                rrmse = compare_plot(x_test,y_test,u_test, u_predict, title=f"{self.title}_{self.iter}")
                metric.append(rrmse)


        end = time.time()
        ET = end - start


        metric = np.array(metric)
        df = pd.DataFrame({"rL2_u":metric[:,0] , "rL2_v":metric[:,1] , "rL2_p":metric[:,2] ,"rL2_U":metric[:,3]})
        df.to_csv(f"./Text/history_{self.title}.csv")


        torch.save(self.model.state_dict(), f'./Saved_Models/Wtrained_{self.iter}_'+self.title+'.pt')
        torch.save(self.model, f'./Saved_Models/Mtrained_epoch{self.iter}_'+self.title+'.pt')


            

