import time
from datetime import datetime

# Importing utility functions and classes from custom modules
from utils.initialize import initialize
from utils.Data import get_data_spatial_lid_cavity
from utils.network2 import Network, NetworkM4, NetworkM4_fused
from utils.encoder import Encoder_conv2, Encoder
from utils.model import Model, Model_M4fused
from utils.losses import Loss_Functions
from utils.trainer import Trainer
from utils.plot import loss_plot

# -----------------------------------------------------------------------------------------
# Parameter Setup Section
# -----------------------------------------------------------------------------------------

# Seeds for creating randomness in the experiment
seeds = [108]  # Example seeds: [108, 213, 387, 522, 790]

# Parametric encoding cell size for the model
RES = [9]

# Names of the models being compared
model_names = ['fused', 'PIXEL', 'PINN', 'M4']

# Number of training epochs
epochs = 50000

# -----------------------------------------------------------------------------------------
# NS Cavity Example Parameters
# -----------------------------------------------------------------------------------------

# Name of the PDE being solved (Navier-Stokes in this example)
pde_name = 'NS'

# Parameter A as described in Table 1 of the paper, with multiple values for experimentation
As = [5, 1]

# Choice of optimizer
optimizer = 'adam'

# Domain size for the cavity problem
domain_size = [0., 1., 0., 1.]

# Setup for experiment parameters including collocation points, boundary points, etc.
params = {'N_f': 5000, 'N_bc': 1000, 'Re': 1000, 'domain_size': domain_size, 'ntest': 100, 'seed': 0, 'use1p': True, 'A': 0}

# Function to generate boundary and collocation points
data_func = get_data_spatial_lid_cavity

# -----------------------------------------------------------------------------------------
# Experiment Loop
# -----------------------------------------------------------------------------------------

# Looping through seeds, As values, and model names to conduct experiments
for seed in seeds:
    for A in As:
        for network_name in model_names:
            
            # Timestamp for unique file naming
            today = datetime.now()
            d2 = today.strftime("%B%d-%H-%M")
            start = time.time()

            # Updating parameters for the current iteration
            params['A'] = A
            params['seed'] = seed
            title = f"_A{params['A']}"

            # Setting dynamic weight flags based on the model type
            dynamic_weight_flag = network_name in ['M4', 'fused']

            # Initializing other flags (unused in the provided code)
            hbc_flag = False
            evo = False
            data_flag = False
            ae = False

            # Formatting the title for saving results based on network type and dynamic weights
            if network_name in ['PINN', 'M4', 'ours', 'fused', 'mfused']:
                title = f"{d2}_{pde_name}_{network_name}_{epochs}_dw_{dynamic_weight_flag}_seed{seed}" + title
            else:
                title = f"{d2}_{pde_name}_{network_name}_{epochs}_{optimizer}_dw_{dynamic_weight_flag}_seed{seed}" + title

            # -----------------------------------------------------------------------------------------
            # Initialization and Model Setup
            # -----------------------------------------------------------------------------------------

            # Initialize computation device and data type based on seed
            dtype, device = initialize(seed=params['seed'])

            # Selecting the network and encoder based on the model name
            if network_name == 'M4':
                network = NetworkM4(input_dim=2, layers=[40, 40, 40, 40], output_dim=3)
                encoder = None
            elif network_name == 'PINN':
                network = Network(input_dim=2, layers=[40, 40, 40, 40, 40, 40, 40, 40], output_dim=3)
                encoder = None
            elif network_name == 'PIXEL':
                network = Network(input_dim=4, layers=[16], output_dim=3)
                encoder = Encoder(n_features=4, res=[16], n_cells=96, domain_size=domain_size, mode='cosine')
            elif network_name == 'fused':
                network = NetworkM4_fused(input_dim=2, layers=[64, 64, 64], output_dim=3)
                encoder = Encoder_conv2(n_features=128, res=[9], n_cells=2, domain_size=domain_size, mode='cosine')

            # Defining the model by combining encoder and decoder (network)
            model2 = Model_M4fused(encoder=encoder, network=network, data_params=params) \
                if network_name == 'fused' else Model(encoder=encoder, network=network, data_params=params)

            # -----------------------------------------------------------------------------------------
            # Loss Function Setup and Training
            # -----------------------------------------------------------------------------------------

            # Defining loss functions for the model
            loss_functions = Loss_Functions(model=model2, name=pde_name, params=params, loss_type='mse')

            # BC loss function for cavity with a single pressure point
            bc_loss_function = loss_functions.bc_loss_one_p

            # Trainer setup for model training
            trainer3 = Trainer(model=model2, pde_loss_function=loss_functions.pde_loss,
                               bc_loss_function=bc_loss_function, lr=1e-2,
                               epochs=epochs, title=title, data_params=params,
                               data_func=data_func, use_data=False,
                               optimizer=optimizer, loss_functions=loss_functions, dynamic_weights=dynamic_weight_flag)

            # Start training
            trainer3()

            # -----------------------------------------------------------------------------------------
            # Experiment Result Processing
            # -----------------------------------------------------------------------------------------

            # Calculating elapsed time
            end = time.time()
            ET = end - start

            # Saving elapsed time to file
            with open(f'./Text/ET_{title}.txt', 'w') as file_object:
                file_object.write(str(ET))

            print(f"Elapsed Time: {ET}")

            # Plotting and saving loss curves
            loss_plot(trainer3.train_losses, trainer3.val_losses, title=title)

            # Function to count the number of trainable parameters in the model
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Counting and saving the number of parameters
            num_param = count_parameters(model2)
            print(num_param)
            with open(f'./Text/ET_{title}.txt', 'a') as file_object:
                file_object.write("\n" + str(num_param))
