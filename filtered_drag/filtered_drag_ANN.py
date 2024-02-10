# This Python script is dedicating to the training and validation of Artifical Neural Network models for filtered drag force in the filtered Two-Fluid Model 
# This file is part of the fTFM_ANN_modeling project (https://github.com/bahardy/fTFM_ANN_modeling.git) distributed under BSD-3-Clause license 
# Copyright (c) Baptiste HARDY. All rights reserved. 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import pickle
import math

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from terminalVelocity import getTerminalVelocity 

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.markersize'] = 1

print(tf.__version__)

# %% -------------------------- List of filtered fine-grid simulation cases -------------------------- %% # 
data_list = []
data_list.append('../data/case_1/')
data_list.append('../data/case_2/')
data_list.append('../data/case_3/')
data_list.append('../data/case_4/')
data_list.append('../data/case_5/')
data_list.append('../data/case_6/')
data_list.append('../data/case_7/')
data_list.append('../data/case_8/')
data_list.append('../data/case_9/')
#data_list.append('/case_10/')

#data_list.append('../data/case_1/')
n_cases = len(data_list)

training_folder = './models/DF_generalizedModel_training_cases1to9_subset/'
os.makedirs(training_folder,exist_ok=True)
os.makedirs(training_folder + '/figures/',exist_ok=True)

# %% -------------------------- Physical and numerical parameters -------------------------- %% # 
g        = 9.81 # m/s2
alp_max  = 0.64
rho_g    = 1.2 # kg/m3
mu_g     = 1.8e-5 #Pa.s
dp       = np.array([75e-6, 90e-6, 100e-6, 75e-6, 150e-6, 150e-6, 180e-6, 130e-6, 180e-6, 120e-6])
rho_p    = np.array([1500,  1500,  1500,    3000,  2500,   1800,   1600,   1800,   2500,   2000])
alp_mean = np.array([0.05,  0.05,  0.05,    0.05,  0.05,   0.05,   0.05,   0.05,   0.05,   0.05])
Lx       = np.array([0.0384, 0.0713, 0.1006, 0.1343, 0.2093, 0.1762, 0.2204, 0.1399, 0.2763, 0.1299])

dp       = dp[:n_cases]
rho_p    = rho_p[:n_cases]
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

# Model training options
train_model = 1     # set to 1 to start new training 
resume_training = 0 # set to 1 to resume previous training 
# if neither train_model nor resume_training is set to 1 ('true'), the previously trained model from the given folder is loaded 

# %% -------------------------- Data Loading -------------------------- %% # 
df_all = pd.DataFrame()
ndata_max    = nx*nx*nz*len(time_values)*(nb_filters-1) # maximum size of dataset for each quantity
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

                df_ = pd.DataFrame([Delta_f[i_case, nf], alp_mean[i_case], Ut[i_case], Ret[i_case], Fr[i_case]]*np.ones((ndata,1)), index=range(ndata), columns=['Delta_f', 'alp_mean', 'Ut', 'Ret', 'Fr'])
                L_cst.append(df_)

    df_base_stats = pd.concat(L_base_stats, ignore_index=True)
    df_vrz_vdz    = pd.concat(L_vrz_vdz, ignore_index=True)
    df_dpdz       = pd.concat(L_dpdz, ignore_index=True)
    df_grad_alp   = pd.concat(L_grad_alp, ignore_index=True)
    df_grad_u     = pd.concat(L_grad_u, ignore_index=True)
    df_drag_z     = pd.concat(L_drag_z, ignore_index=True)
    df_inv_tau    = pd.concat(L_inv_tau, ignore_index=True)
    df_cst        = pd.concat(L_cst, ignore_index=True)
    df            = pd.concat([df_base_stats, df_vrz_vdz, df_dpdz, df_grad_alp, df_grad_u, df_drag_z, df_inv_tau, df_cst], axis=1)

    F = (alp_mean[i_case]*rho_p[i_case] + (1-alp_mean[i_case])*rho_g)*g

    df['vrz']  = df['vrz']/df['alp']
    df['dpdz'] = df['dpdz']/df['alp'] + F #add external forcing to the perdiodic pressure gradient to get the "total" pressure gradient 

    # Scaling of input data 
    df['alp_vdz']   = df['alp_vdz']/(alp_max*Ut[i_case])
    df['alp']       = df['alp']/(alp_max)
    df['vrz']       = df['vrz']/Ut[i_case]
    df['dpdz']      = df['dpdz']/(rho_p[i_case]*g)
    df['Delta_f']   = df['Delta_f']/Lc[i_case]
    df['Igpz_filt'] = df['Igpz_filt']/(rho_p[i_case]*g)

    L_all.append(df)

df_all = pd.concat(L_all, ignore_index=True) 

print('Data is loaded') 

# %% -------------------------- ANN modeling of drift flux  -------------------------- %% # 

#Select features to predict the drift flux
dataset = df_all[['alp_vdz', 'alp', 'vrz', 'dpdz', 'Ret', 'Delta_f']] 

dataset.dropna()

#Split dataset into training and testing subsets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset  = dataset.drop(train_dataset.index)

#Isolate the target variable (here, the variance of alpha) from other features
train_features = train_dataset.copy()
test_features  = test_dataset.copy()
train_labels   = train_features.pop('alp_vdz')
test_labels    = test_features.pop('alp_vdz')
BATCH_SIZE     = 1000


if (train_model):
    # ---------------------- Build the NN model (layers, loss function, learning rate)---------------------- #
    #Normalize input features
    normalizer = tf.keras.layers.Normalization(axis=1,input_dim=train_features.shape[1]) 
    # it is necessary to specify the input_dim parameter, otherwise the build() call fails when loading the model from json config file 
    normalizer.adapt(np.array(train_features), batch_size=BATCH_SIZE)

    model = keras.Sequential([
        normalizer,
        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)
    ])

    val_fraction = 0.2
    N_val        = int(train_features.shape[0]*val_fraction)
    N_train      = train_features.shape[0] - N_val
    steps_per_epoch = math.ceil(N_train/BATCH_SIZE)

    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=steps_per_epoch*10,
    decay_rate=1,
    staircase=False)

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(lr_schedule))

    # Save model parameters                
    def myprint(s):
        with open(training_folder + 'modelsummary.txt','a') as f:
            print(s, file=f)

    model.summary(print_fn=myprint)
    # pprint.pprint(model.to_json(),stream=open(training_folder + 'model.json', 'w'))
    model_json=model.to_json()
    with open(training_folder + 'model.config.json', 'w') as json_file:
        json_file.write(model_json)

    # Creates callbacks for early stopping, log and checkpointing
    stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    checkpoint_path = training_folder + 'cp.ckpt'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_freq='epoch', 
                                                    period=10,
                                                    verbose=1)

    log_callback = keras.callbacks.CSVLogger(training_folder + 'model_fit_log.csv', append=True, separator=';')

    #Train the model and save model training history 
    history = model.fit(
        train_features,
        train_labels,
        validation_split=val_fraction,
        verbose=1, 
        callbacks=[stop_callback, cp_callback, log_callback],
        epochs=1000, 
        batch_size=BATCH_SIZE)

    model_history = history.history
    with open(training_folder + 'trainHistory', 'wb') as file_pi:
        pickle.dump(model_history, file_pi)

    # TF SavedModel format is usefull to resume training afterwards
    model.save(training_folder + 'model.tf')
    
    # H5 format saves only the weights, useful for further inference 
    model.save_weights(training_folder + 'model.weights.h5')


elif (resume_training):
    model = keras.models.load_model(training_folder + 'model.tf')
    model.summary() 

    val_fraction = 0.2
    N_val        = int(train_features.shape[0]*val_fraction)
    N_train      = train_features.shape[0] - N_val
    steps_per_epoch = math.ceil(N_train/BATCH_SIZE)

    #Callbacks for early stopping, log and checkpointing
    stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    checkpoint_path = training_folder + 'cp.ckpt'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_freq='epoch', 
                                                    period=10,
                                                    verbose=1)

    log_callback = keras.callbacks.CSVLogger(training_folder + 'model_fit_log.csv', append=True, separator=';')

    #Resume training of the model and save training history     
    history = model.fit(
        train_features,
        train_labels,
        validation_split=val_fraction,
        verbose=1, 
        callbacks=[stop_callback, cp_callback, log_callback],
        epochs=10000,
        initial_epoch=1, 
        batch_size=BATCH_SIZE)

    model_history = history.history
    with open(training_folder + 'trainHistory', 'wb') as file_pi:
        pickle.dump(model_history, file_pi)

    # TF SavedModel format is usefull to resume training afterwards
    model.save(training_folder + 'model.tf')
    
    # Earlier HDF5 format saves only the weights,can be used for further inference 
    model.save_weights(training_folder + 'model.weights.h5')

else: # inference 

    # TF SavedModel format
    #model = keras.models.load_model(training_folder + 'model.tf')

    # JSON config file + HDF5 weights 
    json_file = open(training_folder + 'model.config.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(training_folder + 'model.weights.h5')
    
    with open(training_folder + 'trainHistory', 'rb') as file_pi:
        model_history = pickle.load(file_pi)

#Prediction on test dataset
test_predictions = model.predict(test_features).flatten()

## FIGURES 
# Plot loss history
fig = plt.figure()
fig.set_size_inches(4,4)
plt.plot(model_history['loss'], label='Training loss')
plt.plot(model_history['val_loss'], label='Validation loss')
plt.yscale("log") 
plt.xlabel('Epoch')
plt.ylabel('Absolute error on drift flux')
plt.legend()
plt.grid(True)
plt.savefig(training_folder + '/figures/' + 'loss_history.pdf', format='pdf',bbox_inches='tight',pad_inches=0.1)

# Scatter plot 
freq = 50
R2 = r2_score(test_labels, test_predictions)
fig = plt.figure()
fig.set_size_inches(4,4)
ax = fig.gca()
ax.set_aspect('equal')
bounds = [-0.5, 0.05]
x0 = bounds[0] + 0.2*(bounds[1]-bounds[0])
y0 = bounds[0] + 0.8*(bounds[1]-bounds[0])
X = test_labels
Y = test_predictions
plt.plot(X[::freq], Y[::freq], marker='.', linestyle = 'none', alpha=0.8)
plt.text(x0, y0, '$R^2 = {:1.2f}$'.format(R2), fontsize = 16)
plt.xlim(bounds)
plt.ylim(bounds)
plt.plot(bounds, bounds, '-', color='tab:red', linewidth=1)
plt.xlabel(r'$ \bar \phi_s v_{d,z}/(\phi_{s,\mathrm{max}} U_t)$, fine-grid data')
plt.ylabel(r'$ \bar \phi_s v_{d,z}/(\phi_{s,\mathrm{max}} U_t)$, DF-ANN model')
plt.savefig(training_folder + '/figures/' + 'DF-ANN_drift_flux_scatterPlot.png', format='png',bbox_inches='tight',pad_inches=0.1)

# Error histogram
fig = plt.figure()
fig.set_size_inches(4,4)
error = (test_predictions - test_labels)/test_labels#relative error wrt the mean
h, bins = np.histogram(error, bins=np.linspace(-1,3,100), density=True)
val = .5*(bins[:-1]+bins[1:])
plt.xlim([bins[0], bins[-1]])
plt.plot(val, h)
plt.xlabel(r'ANN prediction relative error $\bar \phi v_{d,z}$')
plt.ylabel('PDF')
plt.savefig(training_folder + '/figures/' + 'error_drift_flux_hist.pdf', format='pdf',bbox_inches='tight',pad_inches=0.1)


# %% ----------------------------  Filtered drag evaluation ---------------------------- %% # 
Ut_arr       = df_all['Ut'].to_numpy()
vrz          = df_all['vrz'].to_numpy()*Ut_arr
alp          = df_all['alp'].to_numpy()*alp_max
alp_vrz      = alp*vrz
Igp_filt_z   = df_all['Igpz_filt'].to_numpy()
inv_taup_res = df_all['inv_tau_pf'].to_numpy()
alp_mean_arr = df_all['alp_mean'].to_numpy()

Ut_test             = Ut_arr[test_features.index]
alp_vdz_prediction  = test_predictions*(alp_max*Ut_test)
alp_filt_test       = test_features['alp'].to_numpy()
alp_vrz_test        = alp_vrz[test_features.index]
Igp_filt_z_test     = Igp_filt_z[test_features.index]
inv_taup_res_test   = inv_taup_res[test_features.index]
Ret_test            = test_features['Ret'].to_numpy()
alp_mean_test       = alp_mean_arr[test_features.index]

Igp_filt_z_model = inv_taup_res_test*(alp_vrz_test + alp_vdz_prediction)/g

X = Igp_filt_z_test
Y = Igp_filt_z_model
rmse = mean_squared_error(X, Y, squared=False)
with open(training_folder  + "rmse_filtered_drag.csv", 'w') as file:
    print('{:1.6f}'.format(rmse), file=file)
R2_filteredDrag = r2_score(X, Y)

## FIGURES 
# Scatter plot
fig = plt.figure()
fig.set_size_inches(4,4)
bounds = [0, 0.2]
plt.plot(X[::freq], Y[::freq], marker='.', linestyle = 'none', alpha=0.8)
plt.xlim(bounds)
plt.ylim(bounds)
plt.xlabel(r'$ \overline{I}_{gs,z}/(\rho_s g)$, fine-grid data')
plt.ylabel(r'$ \overline{I}_{gs,z}/(\rho_s g)$, DF-ANN model')
x0 = bounds[0] + 0.2*(bounds[1]-bounds[0])
y0 = bounds[0] + 0.8*(bounds[1]-bounds[0])
plt.text(x0, y0, '$R^2 = {:1.2f}$'.format(R2_filteredDrag), fontsize = 16)
plt.plot(bounds, bounds, color='tab:red', linewidth=1.0)
plt.savefig(training_folder + '/figures/' + 'DF-ANN_filtered_drag_scatterPlot.png', format='png',bbox_inches='tight',pad_inches=0.1)

# Error histogram
fig = plt.figure()
fig.set_size_inches(4,4)
error = (Igp_filt_z_model - Igp_filt_z_test)/Igp_filt_z_test #relative error wrt the mean
h, bins = np.histogram(error, bins=np.linspace(-1,3,100), density=True)
val = .5*(bins[:-1]+bins[1:])
error_arr = np.array([val, h])
with open(training_folder + "error_filtered_drag.csv", 'w') as file:
    np.savetxt(file, error_arr.T, fmt='%10.5f', delimiter=' ')
plt.xlim([bins[0], bins[-1]])
plt.plot(val, h)
plt.xlabel(r'$e(\overline{I}_{gs,z})$')
plt.ylabel('PDF')
plt.savefig(training_folder + '/figures/' + 'error_filtered_drag_hist.pdf', format='pdf',bbox_inches='tight',pad_inches=0.1)

plt.show()
