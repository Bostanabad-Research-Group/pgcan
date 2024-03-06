import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import pandas as pd
from sklearn.metrics import r2_score

def contour_helper(grid, x, y, z, colormap_min = 0, colormap_max = 0 , title = 'title', levels=50):
    if colormap_max == colormap_min:
    # getting the value range
        vmin = np.min(z)
        vmax = np.max(z)
    else:
        vmin = colormap_min
        vmax = colormap_max

    # plotting a contour
    plt.subplot(grid)
    plt.tricontour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    plt.tricontourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.title(title)
    cbar = plt.colorbar(pad=0.02, aspect=22, format='%.2f')   # '%.0e''%.3f'
    cbar.mappable.set_clim(vmin, vmax)
    

def compare_plot(xp_mesh, yp_mesh, sol, sol_p, ntest = 100, title = 'default'):
    N = sol.shape[0]
    print(N)
    u,v,p = sol[:,0].cpu().numpy(), sol[:,1].cpu().numpy(), sol[:,2].cpu().numpy()
    #psi = sol_p[:,0].reshape(-1,1)
    pp = sol_p[:,2].reshape(-1,1).detach().cpu().numpy()
    up = sol_p[:,0].reshape(-1,1).detach().cpu().numpy()#torch.autograd.grad(psi, yp_mesh, torch.ones_like(psi), True, True)[0].detach().cpu().numpy()
    vp = sol_p[:,1].reshape(-1,1).detach().cpu().numpy()#-1*torch.autograd.grad(psi, xp_mesh, torch.ones_like(psi), True, True)[0].detach().cpu().numpy()
    subplot_size = 3.
    (width, height) = (3.6*subplot_size, 3*subplot_size) #5 col and 3 rows, each of size subplot_size
    fig = plt.figure(figsize = (width, height))
    gs = GridSpec(3, 3)
    #self.contour_helper(gs[0, 0], self.xp_mesh, self.yp_mesh, self.psip_mesh, '$\psi$')
    xp_mesh = xp_mesh.detach().cpu().numpy()
    yp_mesh = yp_mesh.detach().cpu().numpy()
    contour_helper(gs[0, 0], xp_mesh.reshape(-1,), yp_mesh.reshape(-1,), u.reshape(-1,),title = '$u$')
    contour_helper(gs[0, 1], xp_mesh.reshape(-1,), yp_mesh.reshape(-1,), v.reshape(-1,), title ='$v$')
    contour_helper(gs[0, 2], xp_mesh.reshape(-1,), yp_mesh.reshape(-1,), p.reshape(-1,), title ='$p$')

    contour_helper(gs[1, 0], xp_mesh.reshape(-1,), yp_mesh.reshape(-1,), up.reshape(-1,), title ='$u_{p}$')
    contour_helper(gs[1, 1], xp_mesh.reshape(-1,), yp_mesh.reshape(-1,), vp.reshape(-1,), title ='$v_{p}$')
    contour_helper(gs[1, 2], xp_mesh.reshape(-1,), yp_mesh.reshape(-1,), pp.reshape(-1,), title ='$p_{p}$')

    contour_helper(gs[2, 0], xp_mesh.reshape(-1,), yp_mesh.reshape(-1,), up.reshape(-1,)-u.reshape(-1,), title ='$u$ error')
    contour_helper(gs[2, 1], xp_mesh.reshape(-1,), yp_mesh.reshape(-1,), vp.reshape(-1,)-v.reshape(-1,), title ='$v$ error')
    contour_helper(gs[2, 2], xp_mesh.reshape(-1,), yp_mesh.reshape(-1,), pp.reshape(-1,)-p.reshape(-1,), title ='$p$ error')

    plt.tight_layout()
    plt.savefig('./Figures/'+title+ 'solution_.jpg', dpi = 300)
    #plt.show()

    plt.figure(figsize = (8,6))
    _ = plt.plot(u.reshape(-1,), up.reshape(-1,), label = f'R = {np.round(r2_score(u.reshape(-1,), up.reshape(-1,)),3)}')
    _ = plt.plot([u.reshape(-1,).min(), u.reshape(-1,).max()], [u.reshape(-1,).min(), u.reshape(-1,).max()], c= 'r')
    plt.xlabel('True Value', fontsize = 14)
    plt.ylabel('Predicted Value', fontsize = 14)
    plt.legend()
    plt.savefig('./Figures/'+title+'_u_vs_up.jpg', dpi = 300)

    plt.figure(figsize = (8,6))
    _ = plt.plot(v.reshape(-1,), vp.reshape(-1,), label = f'R = {np.round(r2_score(v.reshape(-1,), vp.reshape(-1,)),3)}')
    _ = plt.plot([v.reshape(-1,).min(), v.reshape(-1,).max()], [v.reshape(-1,).min(), v.reshape(-1,).max()], c= 'r')
    plt.xlabel('True Value', fontsize = 14)
    plt.ylabel('Predicted Value', fontsize = 14)
    plt.legend()
    plt.savefig('./Figures/'+title+'_v_vs_vp.jpg', dpi = 300)


    U = np.sqrt(u.reshape(-1,)**2 + v.reshape(-1,)**2)
    Up = np.sqrt(up.reshape(-1,)**2 + vp.reshape(-1,)**2)
    # RRMSE = [np.sqrt(1/N*(np.sum((u.reshape(-1,) - up.reshape(-1,))**2.0))/(np.sum((u.reshape(-1,))**2.0))) ,\
    #          np.sqrt(1/N*(np.sum((v.reshape(-1,) - vp.reshape(-1,))**2.0))/(np.sum((v.reshape(-1,))**2.0))) , \
    #          np.sqrt(1/N*(np.sum((p.reshape(-1,) - pp.reshape(-1,))**2.0))/(np.sum((up.reshape(-1,))**2.0))) ]

    rl2_u = np.linalg.norm(u.flatten()-up.flatten())/ np.linalg.norm(u.flatten())#np.sqrt((1/N)*(np.sum((u.reshape(-1,) - up.reshape(-1,))**2.0))/(np.sum((u.reshape(-1,))**2.0)))
    rl2_v = np.linalg.norm(v.flatten()-vp.flatten())/ np.linalg.norm(v.flatten())
    rl2_p = np.linalg.norm(p.flatten()-pp.flatten())/ np.linalg.norm(p.flatten())
    rl2_U = np.linalg.norm(Up-U)/ np.linalg.norm(U)

    data = {"x":xp_mesh.flatten() , "y":yp_mesh.flatten() , "u":u.flatten()  , "up":up.flatten() , "v":v.flatten()  , "vp":vp.flatten()  ,  "p":p.flatten()  , "pp":pp.flatten() , "U":U.flatten()  , "Up":Up.flatten() , "rL2": rl2_U}
    
    # df = pd.DataFrame(data)
    # df.to_csv(f"./Text/csv/{title}.csv")

    print("____________________________________________________________________________________________________________")
    print(f"RRMSE is {rl2_U}")
    print("____________________________________________________________________________________________________________")      
    """

    solution = pd.DataFrame({'x':xp_mesh.reshape(-1,), 
                  'y':yp_mesh.reshape(-1,), 'u':u.reshape(-1,), 'up':up.reshape(-1,), 
                  'v':v.reshape(-1,), 'vp':vp.reshape(-1,), 'p':p.reshape(-1,), 'pp':pp.reshape(-1,)})

                  
                  """
    return [rl2_u , rl2_v , rl2_p , rl2_U]#[0] , RRMSE[1] , RRMSE[2]

    #solution.to_csv(f'./plots/predictions/csvs/solution{title}.csv')

def plot_helper(grid, x, y, color = 'k', label = None, title = None):
    # getting the value range
    vmin = np.min(y)
    vmax = np.max(y)
    # plotting a contour
    plt.subplot(grid)
    plt.semilogy(x[20:], y[20:], color= color, label = label)
    plt.title(title)
    plt.legend()
    #plt.ylim((vmin, vmax))

def loss_plot(train_loss, val_loss, title = 'default'):
    fig = plt.figure(figsize=(25,5))
    if 'convection' or 'Burgers' in title:
        gs = GridSpec(1,4)
    else:
        gs = GridSpec(1,3)

    
    epochs = len(train_loss['pde'])

    plot_helper(gs[0,0], x = range(epochs), y = np.array(train_loss['pde']), color = 'b', label='$train loss$')
    plot_helper(gs[0,0], x = range(epochs), y = np.array(val_loss['pde']), color = 'r', label='$val loss$', title = 'PDE loss')

    plot_helper(gs[0,1], x = range(epochs), y = train_loss['bc'], color = 'b', label='$train loss$')
    plot_helper(gs[0,1], x = range(epochs), y = val_loss['bc'], color = 'r', label='$val loss$', title = 'BC loss')

    if 'convection' or 'Burgers' in title:
        plot_helper(gs[0,2], x = range(epochs), y = train_loss['ic'], color = 'b', label='$train loss$')
        plot_helper(gs[0,2], x = range(epochs), y = val_loss['ic'], color = 'r', label='$val loss$', title = 'IC loss')

    #plot_helper(gs[0,2], x = range(epochs), y = train_loss['data'], color = 'b', label='$train loss$')
    #plot_helper(gs[0,2], x = range(epochs), y = train_loss['data'], color = 'r', label='$val loss$', title = 'Data loss')

    plot_helper(gs[0,-1], x = range(epochs), y = train_loss['total'], color = 'b', label='$train loss$')
    plot_helper(gs[0,-1], x = range(epochs), y = val_loss['total'], color = 'r', label='$val loss$', title = 'Total loss')

    fig.supxlabel('Epochs')
    fig.supylabel('MSE Error')

    plt.savefig('./Figures/loss_'+title+'.jpg', dpi = 300)




if __name__ == '__main__':
    pass
