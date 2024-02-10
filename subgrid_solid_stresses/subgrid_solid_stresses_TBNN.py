# This Python script is dedicating to the training and validation of Artifical Neural Network models for the subgrid solid stresses in the filtered Two-Fluid Model 
# This file is part of the fTFM_ANN_modeling project (https://github.com/bahardy/fTFM_ANN_modeling.git) distributed under BSD-3-Clause license 
# Copyright (c) Baptiste HARDY. All rights reserved. 

import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
import math 
import os 
import pickle 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import sys 

from turbulence_preprocessor import TurbulenceDataProcessor

sys.path.append('../')
from filtered_drag.terminalVelocity import getTerminalVelocity

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.markersize'] = 1


#%% --------------- List of constant physical quantities ------------------ %%#
g        = 9.81   # m/s2
alp_max  = 0.64
rho_g    = 1.2    # kg/m3
mu_g     = 1.8e-5 # Pa.s

def loadData():

    # -------------------------- List of filtered fine-grid simulation cases -------------------------- # 
    data_list = []

    # Comment lines to exclude some cases from the training 
    # Careful ! the cases must be commented starting from the last one, otherwise arrays containing physical and numerical parameters will be erroneous 
    data_list.append('../data/case_1/')
    # data_list.append('../data/case_2/')
    # data_list.append('../data/case_3/')
    # data_list.append('../data/case_4/')
    # data_list.append('../data/case_5/')
    # data_list.append('../data/case_6/')
    # data_list.append('../data/case_7/')
    # data_list.append('../data/case_8/')
    # data_list.append('../data/case_9/')
    # data_list.append('../data/case_10/')

    n_cases = len(data_list)

    # -------------------------- Physical and numerical parameters --------------------------  # 
    dp       = np.array([75e-6, 90e-6, 100e-6, 75e-6, 150e-6, 150e-6, 180e-6, 130e-6, 180e-6, 120e-6])
    rho_p    = np.array([1500,  1500,  1500,    3000,  2500,   1800,   1600,   1800,   2500,   2000])
    alp_mean = np.array([0.05,  0.05,  0.05,    0.05,  0.05,   0.05,   0.05,   0.05,   0.05,   0.05])
    Lx       = np.array([0.0384, 0.0713, 0.1006, 0.1343, 0.2093, 0.1762, 0.2204, 0.1399, 0.2763, 0.1299])

    dp       = dp[:n_cases]
    rho_p    = rho_p[:n_cases]
    alp_mean = alp_mean[:n_cases]
    Lx       = Lx[:n_cases]

    Lz = 4*Lx 
    nx = 160 #number of grid points in the transverse direction
    nz = nx*4 
    Delta = Lx/nx

    Ut = np.zeros((n_cases,))
    for i in range(Ut.shape[0]):
        Ut[i] = getTerminalVelocity(rho_p[i], rho_g, dp[i], g, mu_g)

    nb_filters = 9
    filt_size  = np.array([0,2,4,6,8,10,12,16,20])
    filt_size  = filt_size[:nb_filters]
    filt_size  = filt_size.reshape(-1,1)
    Delta      = Delta.reshape(-1,1)
    Delta_f    = np.matmul(Delta, filt_size.T) #filt_size*Delta

    time_values = [[200], [61], [61], [60], [100], [19], [19], [19], [20], [16]]
    time_values = time_values[:n_cases]
    nproc = 1

    # Dimensionless numbers and chracteristic length scale 
    Fr = Ut**2/(g*dp)
    Lc_def = 0
    if (Lc_def == 0):
        Lc = dp*Fr**(1./3.)
    elif(Lc_def == 1):
        Lc = Ut**2/g
    elif(Lc_def ==  2): 
        Lc = dp

    Ret = rho_g*Ut*dp/mu_g

    df_PhyParam = pd.DataFrame(data=np.concatenate((dp.reshape(-1,1),rho_p.reshape(-1,1),alp_mean.reshape(-1,1),Lx.reshape(-1,1)),axis=1),
                               columns=['dp', 'rhop', 'alp_mean', 'Lx'])

    #  -------------------------- Data Loading --------------------------  # 
    # ndata_max    = nx*nx*nz*len(time_values)*(nb_filters-1) # maximum size of dataset for each quantity
    L_all        = []

    IPART        =  2 

    for i_case, folder in enumerate(data_list):
        L_base_stats = []
        L_vrz_vdz    = []
        L_dpdz       = []
        L_grad_alp   = []
        L_grad_u     = []
        L_drag_z     = []
        L_inv_tau    = []
        L_sigma      = []
        L_cst        = []
        data_folder = folder # + '/analysis/' 
        for nf in range(1, nb_filters):
            for time in time_values[i_case]:
                for iproc in range(0,nproc):            
                    filename  = data_folder + "base_stats/base_stats_{:03d}_iph{:02d}_filt{:03d}_p{:03d}.dat".format(time, IPART, nf+1, iproc)
                    df_       = pd.read_csv(filename, skiprows=1, delim_whitespace=True, names=['alp', 'var_alp', 'u', 'v', 'w', 'qp'])
                    L_base_stats.append(df_)

                    ndata     = df_.shape[0]

                    filename  = data_folder + "vrz_vdz/vrz_vdz_{:03d}_filt{:03d}_p{:03d}.dat".format(time, nf+1, iproc)
                    df_       = pd.read_csv(filename, skiprows=1, usecols=[0,1], delim_whitespace=True, names=['vrz', 'alp_vdz']) #load only first two columns (3rd column is not useful here)
                    L_vrz_vdz.append(df_)

                    filename  = data_folder + "alp_dpdz/alp_dpdz_{:03d}_iph{:02d}_filt{:03d}_p{:03d}.dat".format(time, IPART, nf+1, iproc) 
                    df_       = pd.read_csv(filename, skiprows=1, usecols=[0], delim_whitespace=True, names=['dpdz']) #load only first column (2nd column is not useful here)
                    L_dpdz.append(df_)

                    filename  = data_folder + "grad_alp/grad_alp_{:03d}_iph{:02d}_filt{:03d}_p{:03d}.dat".format(time, IPART, nf+1, iproc)
                    df_       = pd.read_csv(filename, skiprows=1, delim_whitespace=True, names=['dalpdx', 'dalpdy', 'dalpdz'])
                    L_grad_alp.append(df_)

                    filename  = data_folder + "grad_u/grad_u_{:03d}_iph{:02d}_filt{:03d}_p{:03d}.dat".format(time, IPART, nf+1, iproc)
                    df_       = pd.read_csv(filename, skiprows=1, delim_whitespace=True, names=['dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz', 'dwdx', 'dwdy', 'dwdz', 'div_u', 'S_norm'])
                    L_grad_u.append(df_)

                    filename  = data_folder + "drag_z/drag_z_{:03d}_filt{:03d}_p{:03d}.dat".format(time, nf+1, iproc)
                    df_       = pd.read_csv(filename, skiprows=1, delim_whitespace=True, names=['Igpz_filt', 'Igpz_filt_approx'])
                    L_drag_z.append(df_)

                    filename  = data_folder + "invtau_pf_res/invtau_pf_res_{:03d}_filt{:03d}_p{:03d}.dat".format(time, nf+1, iproc)
                    df_       = pd.read_csv(filename, skiprows=1, delim_whitespace=True, names=['inv_tau_pf'])
                    L_inv_tau.append(df_)

                    filename  = data_folder + "sigma_sgs/sigma_sgs_{:03d}_iph{:02d}_filt{:03d}_p{:03d}.dat".format(time, IPART, nf+1, iproc)
                    df_       = pd.read_csv(filename, skiprows=1, delim_whitespace=True, names=['sig_xx', 'sig_xy', 'sig_xz', 'sig_yy', 'sig_yz', 'sig_zz'])
                    L_sigma.append(df_)

                    df_ = pd.DataFrame([Delta_f[i_case, nf], alp_mean[i_case], Ut[i_case], Ret[i_case], Fr[i_case]]*np.ones((ndata,1)), index=range(ndata), columns=['Delta_f', 'alp_mean', 'Ut', 'Ret', 'Fr'])
                    L_cst.append(df_)

        df_base_stats = pd.concat(L_base_stats, ignore_index=True)
        df_vrz_vdz    = pd.concat(L_vrz_vdz, ignore_index=True)
        df_dpdz       = pd.concat(L_dpdz, ignore_index=True)
        df_grad_alp   = pd.concat(L_grad_alp, ignore_index=True)
        df_grad_u     = pd.concat(L_grad_u, ignore_index=True)
        df_drag_z     = pd.concat(L_drag_z, ignore_index=True)
        df_inv_tau    = pd.concat(L_inv_tau, ignore_index=True)
        df_sigma      = pd.concat(L_sigma, ignore_index=True)
        df_cst        = pd.concat(L_cst, ignore_index=True)

        #Scaling (1st part) 
        df_grad_u     = df_grad_u*Ut[i_case]/g      #Time scale Ut/g
        df_drag_z     = df_drag_z/(rho_p[i_case]*g) #
        df_sigma      = df_sigma/(rho_p[i_case]*Ut[i_case]**2)

        # Merge dataframes 
        df            = pd.concat([df_base_stats, df_vrz_vdz, df_dpdz, df_grad_alp, df_grad_u, df_drag_z, df_inv_tau, df_sigma, df_cst], axis=1)

        # Modify work variables
        df['vrz']  = df['vrz']/df['alp'] #we work with vrz instead of alp*vrz
        F          = (alp_mean[i_case]*rho_p[i_case] + (1-alp_mean[i_case])*rho_g)*g # external forcing 
        df['dpdz'] = df['dpdz']/df['alp'] + F #add external forcing to the perdiodic pressure gradient to get the "total" pressure gradient 

        # Scaling (2nd part)
        df['alp_vdz']   = df['alp_vdz']/(alp_max*Ut[i_case])
        df['alp']       = df['alp']/(alp_max)
        df['vrz']       = df['vrz']/Ut[i_case]
        df['dpdz']      = df['dpdz']/(rho_p[i_case]*g)
        df['Delta_f']   = df['Delta_f']/Lc[i_case]

        L_all.append(df)

    df_filteredData = pd.concat(L_all, ignore_index=True) 
    return df_PhyParam, df_filteredData 

#%% --------------- Main Script ------------------ %% # 

#---------------------------- Data loading and pre-processing -------------------------------- #
df_param, df_data = loadData()
print('Data is loaded') 

# Reshape grad_u and stresses to num_points X 3 X 3 arrays
num_points = df_data.shape[0]
grad_u     = np.zeros((num_points, 3, 3))
sigma_sgs  = np.zeros((num_points, 3, 3))

# grad_u has 9 independent components 
grad_u[:,0,0] = df_data['dudx'].to_numpy()
grad_u[:,0,1] = df_data['dudy'].to_numpy()
grad_u[:,0,2] = df_data['dudz'].to_numpy()
grad_u[:,1,0] = df_data['dvdx'].to_numpy()
grad_u[:,1,1] = df_data['dvdy'].to_numpy()
grad_u[:,1,2] = df_data['dvdz'].to_numpy()
grad_u[:,2,0] = df_data['dwdx'].to_numpy()
grad_u[:,2,1] = df_data['dwdy'].to_numpy()
grad_u[:,2,2] = df_data['dwdz'].to_numpy()

# sigma_sgs has 6 independent components 
sigma_sgs[:,0,0] = df_data['sig_xx']
sigma_sgs[:,0,1] = df_data['sig_xy']
sigma_sgs[:,0,2] = df_data['sig_xz']
sigma_sgs[:,1,0] = df_data['sig_xy'] #sig_yx = sig_xy
sigma_sgs[:,1,1] = df_data['sig_yy']
sigma_sgs[:,1,2] = df_data['sig_yz']
sigma_sgs[:,2,0] = df_data['sig_xz'] #sig_zx = sig_xz
sigma_sgs[:,2,1] = df_data['sig_yz'] #sig_zy = sig_yz
sigma_sgs[:,2,2] = df_data['sig_zz'] 


alp     = df_data['alp'].to_numpy().reshape(-1,1)
vrz     = df_data['vrz'].to_numpy().reshape(-1,1)
Delta_f = df_data['Delta_f'].to_numpy().reshape(-1,1)

# Neural Network Parameters
enforce_realizability = False                 # Set to True to enforce realizability constraint on Reynolds stresses
num_realizability_its = 5                    # Number of iterations to enforce realizability
seed                  = 1234                 # Random seed
num_layers            = 8                    # Number of hidden layers
num_nodes             = 30                   # Number of nodes per layer
train_fraction        = 0.8                  # Fraction of data used for training 
val_fraction          = 1 - train_fraction   # Fraction of data used for validation/testing
initial_lr            = 0.0005               # Initial learning rate 
BATCH_SIZE            = 500                  # Size of a data batch
N_EPOCHS_MAX          = 1000                 # Maximum number of training epochs
activation            = 'relu'               # Activation function of the nodes 
# TBNN parametrs 
num_scalar_invariants = 3
num_tensors           = 4


# Model training options
train_model     = 1 # set to 1 to start new training 
resume_training = 0 # set to 1 to resume previous training 
# if neither train_model nor resume_training is set to 1 ('true'), the previously trained model from the given folder is loaded 

# Create folder for saving model 
training_folder = './models/subgrid_stress_TBNN_model_subset/'
postpro_folder  = training_folder 
# postpro_folder  = './models/subgrid_stress_TBNN_model_withRealizability/' 
os.makedirs(training_folder,exist_ok=True)
os.makedirs(postpro_folder,exist_ok=True)
os.makedirs(postpro_folder + '/figures/',exist_ok=True)


#Compute strain rate and rotation rate tensors 
data_processor = TurbulenceDataProcessor()
Sij, Rij       = data_processor.calc_Sij_Rij(grad_u)
sb    = data_processor.calc_scalar_basis(Sij, Rij, num_invariants=num_scalar_invariants)  # Scalar basis
tb    = data_processor.calc_tensor_basis(Sij, Rij, num_tensor_basis=num_tensors)                         # Tensor basis
label = data_processor.calc_output(sigma_sgs)

x = np.concatenate((sb, alp, vrz, Delta_f), axis=1)
print("Data processing: done")

if seed:
    np.random.seed(seed) # sets the random seed 

# Split into training and test data sets
x_train, tb_train, label_train, x_test, tb_test, label_test, train_idx, test_idx = \
    TurbulenceDataProcessor.train_test_split_TBNN(x, tb, label, fraction=train_fraction, seed=seed)

if (train_model):
    print("TBNN Model building starts...")

    #---------------------------- Build Tensor Basis Neural Network ----------------------------#
    num_inputs       = x.shape[-1]
    num_tensor_basis = tb.shape[1]
    x_input          = tf.keras.Input(shape=num_inputs)
    tb_input         = tf.keras.Input(shape=(num_tensor_basis,9))

    normalization_scalar_layer = tf.keras.layers.Normalization(axis=1, name='NormalizationLayerSB', input_dim=num_inputs)
    normalization_scalar_layer.adapt(x)
    scalar_layer               = normalization_scalar_layer(x_input)
    hidden_layer               = tf.keras.layers.Dense(units=num_nodes, activation='relu')(scalar_layer)
    for i in range(num_layers-1):
        hidden_layer           = tf.keras.layers.Dense(units=num_nodes, activation='relu')(hidden_layer)
    linear_layer               = tf.keras.layers.Dense(units=num_tensor_basis, activation='linear',name='LinearLayer')(hidden_layer)

    normalization_tensor_layer = tf.keras.layers.Normalization(axis=1, name='NormalizationLayerTB', input_dim=num_tensor_basis)
    normalization_tensor_layer.adapt(tb)
    tensor_layer               = normalization_tensor_layer(tb_input)
    output                     = tf.keras.layers.Dot((1,1))([linear_layer, tensor_layer])
    model                      = tf.keras.Model(inputs=[x_input, tb_input], outputs=output)

    tf.keras.utils.plot_model(model, training_folder + 'model.png') 

    #---------------------------- Compile model ----------------------------# 
    n_train_tot     = x_train.shape[0]
    n_val           = int(x_train.shape[0]*val_fraction)    # number of data points used for validation during training procedure
    n_train         = n_train_tot - n_val                   # number of data points used for training 
    steps_per_epoch = math.ceil(n_train/BATCH_SIZE)         # number of batch passes per epoch 
    lr_schedule     = keras.optimizers.schedules.InverseTimeDecay(
                        initial_lr, 
                        decay_steps=steps_per_epoch*10, 
                        decay_rate=1, 
                        staircase=False) #decay applied every 10 epochs
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(lr_schedule))
    print("Model is compiled")
    model.summary()

    # Creates callbacks for early stopping, log and checkpointing
    stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    #Train the model 
    print("Model training starts...")
    training_history = model.fit([x_train, tb_train], 
                        label_train, 
                        batch_size=BATCH_SIZE, 
                        epochs=N_EPOCHS_MAX, 
                        validation_split=val_fraction, 
                        verbose=1, 
                        callbacks=[stop_callback])

    history = training_history.history
    with open(training_folder + 'history', 'wb') as file:
        pickle.dump(history, file)

    print("Model saving ...")
    model.save(training_folder + 'model.tf', save_format='tf')


else: #Load TBNN model 
    
    with open(training_folder + 'history', 'rb') as file:
            history = pickle.load(file)
    
    model = keras.models.load_model(training_folder + 'model.tf')
    model.summary() 

    if (postpro_folder != training_folder): 
        model.save(postpro_folder + 'model.tf', save_format='tf')



#Plot loss history
fig = plt.figure()
fig.set_size_inches(3.5,3.5)
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.yscale("log") 
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig(postpro_folder + '/figures/' + 'loss_history.pdf', format='pdf',bbox_inches='tight',pad_inches=0.1)

# Make predictions on train and test data to get train error and test error
prediction_train = model([x_train, tb_train]).numpy()
prediction_test  = model([x_test, tb_test]).numpy()

y_train = prediction_train
y_test = prediction_test 

# Enforce realizability
if enforce_realizability:
    for i in range(num_realizability_its):
        y_train = TurbulenceDataProcessor.make_realizable(y_train)
        y_test  = TurbulenceDataProcessor.make_realizable(y_test)
    print("Realizability enforced.")

#Statistics 
y_train = y_train.reshape(-1, 3, 3)
y_test  = y_test.reshape(-1, 3, 3)
label_train     = label_train.reshape(-1,3,3)
label_test      = label_test.reshape(-1,3,3)

y_norm_train = np.zeros((y_train.shape[0],1))
y_norm_test  = np.zeros((y_test.shape[0],1))
label_norm_train = np.zeros((label_train.shape[0],1))
label_norm_test  = np.zeros((label_test.shape[0],1))

eig_val_train = np.zeros((label_train.shape[0],3))
eig_val_test  = np.zeros((label_test.shape[0],3))
eig_val_pred_train  = np.zeros((y_train.shape[0],3))
eig_val_pred_test   = np.zeros((y_test.shape[0],3))

for i in range(label_train.shape[0]): 
    label_norm_train[i] = np.trace(np.matmul(label_train[i,:,:], label_train[i,:,:]))
    y_norm_train[i]     = np.trace(np.matmul(y_train[i,:,:], y_train[i,:,:]))
    eig_val_train[i,:], eig_vect             = np.linalg.eig(label_train[i,:,:])
    eig_val_pred_train[i,:], eig_vect_pred   = np.linalg.eig(y_train[i,:,:])


for i in range(label_test.shape[0]):    
    label_norm_test[i]  = np.trace(np.matmul(label_test[i,:,:], label_test[i,:,:]))
    y_norm_test[i]      = np.trace(np.matmul(y_test[i,:,:], y_test[i,:,:]))
    eig_val_test[i,:], eig_vect             = np.linalg.eig(label_test[i,:,:])
    eig_val_pred_test[i,:], eig_vect_pred   = np.linalg.eig(y_test[i,:,:])

# ---------------------------- Model evaluation --------------------------- #
# Model assessment based on stress tensor individual components 
    
tau_test_flat = data_processor.calc_flatten_tensor(label_test)
tau_pred_flat = data_processor.calc_flatten_tensor(y_test)

X = tau_test_flat.flatten()
Y = tau_pred_flat.flatten()
rmse = mean_squared_error(X, Y, squared=False)
with open(training_folder + "rmse_tau_ij.csv", 'w') as file:
    print('{:1.6f}'.format(rmse), file=file)
R2_test  = r2_score(X, Y)

print ("R2 score on individual components:", R2_test)

freq = 5
bounds = np.array([-0.5, 0.75])
x0 = bounds[0] + 0.2*(bounds[1] - bounds[0])
y0 = bounds[0] + 0.8*(bounds[1] - bounds[0])
fig = plt.figure()
fig.set_size_inches(3.5,3.5)
plt.plot(X[::freq], Y[::freq], marker='.', linestyle = 'none', alpha=0.8)
plt.xlabel(r'$\tau_{s,ij}^*$, fine-grid data')
plt.ylabel(r'$\tau_{s,ij}^*$, TBNN model') 
plt.xlim(bounds)
plt.ylim(bounds)
plt.plot(bounds, bounds, '-', color='tab:red', linewidth=1)
plt.text(x0 , y0, '$R^2 = {:1.2f}$'.format(R2_test), fontsize = 16)
plt.savefig(postpro_folder  + '/figures/' + 'subgrid_stresses_TBNN_scatter_plot.png', format='png',bbox_inches='tight',pad_inches=0.1)

# Model assessment based on stress tensor norm 
R2_train = r2_score(label_norm_train, y_norm_train)
R2_test  = r2_score(label_norm_test, y_norm_test)
print ("R2 score on norm:", R2_test)

bounds = np.array([0, 0.8])
x0 = bounds[0] + 0.2*(bounds[1] - bounds[0])
y0 = bounds[0] + 0.8*(bounds[1] - bounds[0])
fig = plt.figure()
fig.set_size_inches(3.5,3.5)
plt.plot(label_norm_test[::freq], y_norm_test[::freq], marker='.', linestyle = 'none', alpha=0.8)
plt.xlabel(r'$\boldsymbol{\tau}_s^* : \boldsymbol{\tau}_s^*$, fine-grid data')
plt.ylabel(r'$\boldsymbol{\tau}_s^* : \boldsymbol{\tau}_s^*$, TBNN model')
plt.xlim(bounds)
plt.ylim(bounds)
plt.plot(bounds, bounds, '-', color='tab:red', linewidth=1)
plt.text(x0 , y0, '$R^2 = {:1.2f}$'.format(R2_test), fontsize = 16)
plt.savefig(postpro_folder  + '/figures/' + 'subgrid_stresses_norm_TBNN_scatter_plot.png', format='png',bbox_inches='tight',pad_inches=0.1)

# Model assessment based on stress tensor eigenvalues 
R2_train = r2_score(eig_val_train.flatten(), eig_val_pred_train.flatten())
R2_test  = r2_score(eig_val_test.flatten(),  eig_val_pred_test.flatten())
print ("R2 score on eigenvalues:", R2_test)

bounds = np.array([-0.33, 0.66])
x0 = bounds[0] + 0.2*(bounds[1] - bounds[0])
y0 = bounds[0] + 0.8*(bounds[1] - bounds[0])

fig = plt.figure()
fig.set_size_inches(3.5,3.5)
plt.plot(np.min(eig_val_test[::freq,:], axis=1), np.min(eig_val_pred_test[::freq,:],axis=1), marker='.', linestyle = 'none', alpha=0.8, label='$\lambda_{\mathrm{min}}$')
plt.plot(np.max(eig_val_test[::freq,:], axis=1), np.max(eig_val_pred_test[::freq,:],axis=1), marker='.', linestyle = 'none', alpha=0.8, label='$\lambda_{\mathrm{max}}$')
plt.xlabel(r'Eigenvalues $\boldsymbol{\tau}_s^*$, fine-grid data')
plt.ylabel(r'Eigenvalues $\boldsymbol{\tau}_s^*$, TBNN model')
plt.xlim(bounds)
plt.ylim(bounds)
plt.plot(bounds, bounds, '-', color='tab:red', linewidth=1)
plt.text(x0 , y0, '$R^2 = {:1.2f}$'.format(R2_test), fontsize = 16)
plt.savefig(postpro_folder + '/figures/' + 'subgrid_stresses_eigenval_TBNN_scatter_plot.png', format='png',bbox_inches='tight',pad_inches=0.1)

# Acces output of selected layers
gn_model  = tf.keras.Model(model.input, model.get_layer(name='LinearLayer').output)
tb_model  = tf.keras.Model(model.input, model.get_layer(name='NormalizationLayerTB').output)

gn_model_output  = gn_model.predict([x_test, tb_test])
tb_model_output  = tb_model.predict([x_test, tb_test])

mean_contribution = np.mean(np.fabs(gn_model_output), axis=0)
rel_contribution  = mean_contribution/np.sum(mean_contribution)*100

plt.show()

# if __name__ == "__main__":
#     main()
#     print("Job finished")

