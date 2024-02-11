"""
This Python script is dedicating to the training and validation of Artifical Neural Network models for the subgrid solid stresses in the filtered Two-Fluid Model
A Multi-Layer Perceptron is trained to predict the so-called mesoscale (or eddy) viscosity, defined as : mu_s,meso = sqrt(tau_s:tau_s)/(2*sqrt(S_s:S_s))  
This file is part of the fTFM_ANN_modeling project (https://github.com/bahardy/fTFM_ANN_modeling.git) distributed under BSD-3-Clause license 
Copyright (c) Baptiste HARDY. All rights reserved. 
"""

import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
import os 
import pickle 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import sys
sys.path.append('../')
from filtered_drag.terminalVelocity import getTerminalVelocity
from turbulence_preprocessor import TurbulenceDataProcessor
from subgrid_solid_stresses_TBNN import PhyParam, loadData

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.markersize'] = 3


#%% -------------- Load data --------------------- %%#
cst_param, df_param, df_data = loadData()
g     = cst_param.g
rho_g = cst_param.rho_g
mu_g  = cst_param.mu_g

dp    = df_param['dp'][0]     #only one case 
rho_p = df_param['rhop'][0] 
Ut    = getTerminalVelocity(rho_p,rho_g,dp,g,mu_g)
print(Ut)

alp     = df_data['alp'].to_numpy().reshape(-1,1)
vrz     = df_data['vrz'].to_numpy().reshape(-1,1)
Delta_f = df_data['Delta_f'].to_numpy().reshape(-1,1)
u       = df_data['u'].to_numpy().reshape(-1,1)
v       = df_data['v'].to_numpy().reshape(-1,1)
w       = df_data['w'].to_numpy().reshape(-1,1)
dudx    = df_data['dudx'].to_numpy().reshape(-1,1)
dudy    = df_data['dudy'].to_numpy().reshape(-1,1)
dudz    = df_data['dudz'].to_numpy().reshape(-1,1)
dvdx    = df_data['dvdx'].to_numpy().reshape(-1,1)
dvdy    = df_data['dvdy'].to_numpy().reshape(-1,1)
dvdz    = df_data['dvdz'].to_numpy().reshape(-1,1)
dwdx    = df_data['dwdx'].to_numpy().reshape(-1,1)
dwdy    = df_data['dwdy'].to_numpy().reshape(-1,1)
dwdz    = df_data['dwdz'].to_numpy().reshape(-1,1)

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

#%% -------------- Build Neural Netwoek --------------------- %%#

#Flags for NN 
seed = 1234
num_layers = 3
num_nodes = [128, 32, 8]
train_fraction = 0.8
val_fraction = 0.2
initial_lr = 0.0005
batch_size = 500
n_epochs_max = 1000
activation = 'relu'
train_model = 0

#Folder for the results of the training 
training_folder = './models/eddy_viscosity_ANN_model_subset/'
postpro_folder  = './models/eddy_viscosity_ANN_model_subset/'

os.makedirs(training_folder,exist_ok=True)
os.makedirs(postpro_folder,exist_ok=True)
os.makedirs(postpro_folder + '/figures/' ,exist_ok=True)

data_processor = TurbulenceDataProcessor()
Sij, Rij       = data_processor.calc_Sij_Rij(grad_u)

num_points = Sij.shape[0]
S_norm = np.zeros((num_points,))
for i in range(num_points):
    S_norm[i] = np.sqrt(np.trace(np.matmul(Sij[i, :, :], Sij[i, :, :])))
S_norm = S_norm.reshape(-1,1)    

labels = data_processor.calc_eddy_viscosity(sigma_sgs, Sij)
x      = np.concatenate((alp, S_norm, Delta_f), axis=1) # 3-marker model 
# x      = np.concatenate((alp, u, v, w, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, Delta_f), axis=1) # full 14-marker model 

if seed:
    np.random.seed(seed) # sets the random seed 

# Split into training and test data sets
x_train, labels_train, x_test, labels_test, train_idx, test_idx = TurbulenceDataProcessor.train_test_split(x, labels, fraction=train_fraction, seed=seed)

#%% -------------- Train NN --------------------- %%#

if (train_model):
    print("Mesoscale Viscosity Model building starts...")

    # Build network structure
    num_inputs = x.shape[-1]
    x_input  = tf.keras.Input(shape=num_inputs)

    normalization_scalar_layer = tf.keras.layers.Normalization(axis=1, name='NormalizationLayer')
    normalization_scalar_layer.adapt(x)
    scalar_layer = normalization_scalar_layer(x_input)
    hidden_layer = tf.keras.layers.Dense(units=num_nodes[0], activation='relu')(scalar_layer)
    for i in range(num_layers-1):
        hidden_layer = tf.keras.layers.Dense(units=num_nodes[i+1], activation='relu')(hidden_layer)
    output = tf.keras.layers.Dense(units=1, activation='linear',name='OutputLayer')(hidden_layer)

    model = tf.keras.Model(inputs=x_input, outputs=output)

    tf.keras.utils.plot_model(model, training_folder + 'model.png') 

    # Compile model 
    n_train_tot  = x_train.shape[0]
    n_val        = int(x_train.shape[0]*val_fraction) # number of data points used for validation while training 
    n_train      = n_train_tot - n_val                # number of data points used for training 
    steps_per_epoch = int(n_train/batch_size)         # number of batch passes per epoch 
    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(initial_lr, decay_steps=steps_per_epoch*10, decay_rate=1, staircase=False) #decay applied every 10 epochs
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(lr_schedule))
    print("Model is compiled")
    model.summary()

    # Creates callbacks for early stopping, log and checkpointing
    stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    #Train the model 
    print("Model training starts...")
    training_history = model.fit(x_train, 
                        labels_train, 
                        batch_size=batch_size, 
                        epochs=n_epochs_max, 
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

#Plot loss history
fig = plt.figure()
fig.set_size_inches(4,4)
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.yscale("log") 
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig(postpro_folder + 'loss_history.pdf', format='pdf',bbox_inches='tight',pad_inches=0.1)

# Make predictions on train and test data to get train error and test error
prediction_train = model(x_train).numpy()
prediction_test  = model(x_test).numpy()

y_train = prediction_train
y_test = prediction_test 

step = 1
bounds = np.array([0, 0.015])
x0 = bounds[0] + 0.2*(bounds[1] - bounds[0])
y0 = bounds[0] + 0.8*(bounds[1] - bounds[0])
X = labels_test/(rho_p*Ut**3/g)
Y = y_test/(rho_p*Ut**3/g)
R2_train = r2_score(X, Y)

fig = plt.figure()
fig.set_size_inches(3.5,3.5)
plt.plot(X[::step], Y[::step], marker=',', markersize=0.5, linestyle = 'none', alpha=0.5)
plt.xlabel(r'$\mu_{s,\mathrm{meso}}^{*}$, fine-grid data')
plt.ylabel(r'$\mu_{s,\mathrm{meso}}^{*}$,  ANN model')
plt.xlim(bounds)
plt.ylim(bounds)
plt.plot(bounds, bounds, '-', color='tab:red', linewidth=1)
plt.text(x0 , y0, '$R^2 = {:1.2f}$'.format(R2_train), fontsize = 16)
plt.savefig(postpro_folder  + '/figures/' + 'mu_meso_ANN_scatter_plot.png', format='png',bbox_inches='tight',pad_inches=0.1)

error = (prediction_test - labels_test)/labels_test
h, bins = np.histogram(error, bins=np.linspace(-1,3,100), density=True)
val = .5*(bins[:-1]+bins[1:])
error_arr = np.array([val, h])
with open(training_folder + "error_mu_meso.csv", 'w') as file:
    np.savetxt(file, error_arr.T, fmt='%10.5f', delimiter=' ')
fig = plt.figure()
fig.set_size_inches(3.5,3.5)
plt.xlim([bins[0], bins[-1]])
plt.plot(val, h)
plt.xlabel(r'$e(\mu_s^{\mathrm{(meso)}})$')
plt.ylabel('PDF')
plt.savefig(training_folder + '/figures/' + 'error_mu_meso_hist' + '.pdf', format='pdf',bbox_inches='tight',pad_inches=0.1)


#%% Prediction of the compelete filtered stresses tensor 
mu_s      = prediction_test[:,0]
n_test    = mu_s.shape[0] 
Sij_test  = Sij[test_idx,:,:]

# Exact stresses 
tau            = data_processor.calc_dev_stresses(sigma_sgs)
P_meso         = data_processor.calc_mesoscale_pressure(sigma_sgs)
P_meso_test = P_meso[test_idx]
for i in range(num_points):
    tau[i,:,:] = tau[i,:,:]/(3*P_meso[i])
# tau_test      = tau[test_idx,:]
tau_flat      = data_processor.calc_flatten_tensor(tau)
tau_test_flat = tau_flat[test_idx,:]

# Predictions by eddy viscosity ANN model
tau_pred  = np.zeros((n_test,3,3))
for i in range(n_test):
    tau_pred[i,:,:] = 2*mu_s[i]*Sij_test[i,:,:]/(3*P_meso_test[i])
tau_pred_flat = data_processor.calc_flatten_tensor(tau_pred)

X = tau_test_flat.flatten()
Y = tau_pred_flat.flatten()

rmse = mean_squared_error(X, Y, squared=False)
with open(training_folder + "rmse_tau_ij.csv", 'w') as file:
    print('{:1.6f}'.format(rmse), file=file)

R2_train = r2_score(X, Y)
bounds = np.array([-2, 2])
x0 = bounds[0] + 0.2*(bounds[1] - bounds[0])
y0 = bounds[0] + 0.8*(bounds[1] - bounds[0])
fig = plt.figure()
fig.set_size_inches(3.5,3.5)
plt.plot(X, Y, marker=',', markersize=0.5, linestyle = 'none', alpha=0.5)
plt.xlabel(r'$\tau_{s,ij}^*$, fine-grid data')
plt.ylabel(r'$\tau_{s,ij}^*$, ANN model')
plt.xlim(bounds)
plt.ylim(bounds)
plt.plot(bounds, bounds, '-', color='tab:red', linewidth=1)
plt.text(x0 , y0, '$R^2 = {:1.2f}$'.format(R2_train), fontsize = 16)
plt.savefig(postpro_folder + '/figures/' + 'mu_meso_ANN_tau_ij_scatter_plot.png', format='png',bbox_inches='tight',pad_inches=0.1)
plt.show()
