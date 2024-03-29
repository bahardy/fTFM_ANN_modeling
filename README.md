# fTFM_ANNmodeling
This project is dedicated to the development of Artificial Neural Network (ANN) models to close the subgrid terms in the filtered Two-Fluid Model, namely the filtered drag force and the solid phase subgrid stresses.\
Paper Title: "Machine learning approaches to close the filtered two-fluid model for gas-solid flows: Models for subgrid drag force and solid phase stress"\
Authors: "Baptiste Hardy and Stefanie Rauchenzauner and Pascal Fede and Simon Schneiderbauer and Olivier Simonin and Sankaran Sundaresan and Ali Ozel"

## Dataset 
The dataset used to train ANN models is obtained from fine-grid Two-Fluid simulation results of a tri-periodic gas-solid fluidized bed. Simulations have been performed using the NEPTUNE_CFD software. 
The fine-grid simulation results have been spatially filtered for a range of filter sizes. Only a subset of the full dataset is provided in this repository (about 6%). 

The `data` folder contains the filtered dataset obtained from 10 cases with different physical parameters detailed in the param.txt file inside each subfolder.

## Models 
### For the filtered drag force
The `filtered_drag` folder contains:
* The Python source code based on Keras API to train and validate ANN models for the filtered drag force (`filtered_drag_ANN.py`). The `terminalVelocity` module contains a routine to calculate the terminal settling velocity of a single isolated particle as a function of physical parameters.
* The `models` subfolder containing the trained models:
    * `DF_generalizedModel_training_cases1to9`: drift flux model obtained from the full dataset 
    * `DF_generalizedModel_training_cases1to9_subset`: drift flux model obtained from the partial dataset shared in this repository
  
  In both cases, the model has been saved (and can therefore be uploaded) in two ways:
  * TensorFlow Saved format: the whole model (architecture and weights) is saved in the `model.tf` folder
  * JSON-HDF5 pair format: the model architecture is saved in the `model.config.json` file, the weights are stored in the `model.weights.h5` binary file
 
The features to be fed (in this order) to the network are the following: $\frac{\bar \phi_s}{\phi_{s,max}}$, $\frac{\tilde u_{slip,z}}{U_t}$ , $\frac{\partial \bar p_g}{\partial z}$, $Re_p$, $\frac{\bar \Delta}{d_p Fr_p^{1/3}}$.
The data are automatically shifted by the mean and scaled by the standard deviation of the training sample through a Normalization preprocessing layer embedded in the network.


### For the subrid solid stress 
The `subgrid_solid_stresses` folder contains:
* The Python source code to train and validate ANN models for the subgrid solid stress. More specifically:
   * `mesoscale_pressure_ANN.py`: training and validation of an ANN model for the mesoscale pressure, i.e. the spherical part of the subrid stress (aka the subrid kinetic energy)  
   * `eddy_viscosity_ANN.py`: training and validation of an ANN model for the subgrid stress using an eddy-viscosity approach (Boussinesq hypothesis)
   * `subgrid_solid_stresses_TBNN.py`: training and validation of a Tensor-Based Neural Network (TBNN) to predict the individual components of the subgrid stress
  
  These files rely on the preprocessing files `preprocessor.py`and `turbulence_preprocessor.py` largely inspired from the sandialabs TBNN github repository: https://github.com/sandialabs/tbnn
* The `models` subfolder containing the trained models, either using the full dataset or the data subset shared in this repository when the directory is appended with the `_subset` suffix
* The c++ code `TBNN_prediction.C` to load the previously trained TensorFlow TBNN model for the subgrid stress. An example file `input.csv` contains typical values of the physical quantities needed to evaluate the input features of the TBNN model at a given spatial location (one single occurence). The TBNN model predictions (i.e. the 6 individual components of the subgrid solid stress) are exported to the `output.csv` file.

__Note__: The loading and reading of TensorFlow models relies on the `cppflow` library that can be installed from https://github.com/serizba/cppflow

## Manuscript@arXiv and Bibtex Entry
@misc{hardy2023machine,\
   title={Machine learning approaches to close the filtered two-fluid model for gas-solid flows: Models for subgrid drag force and solid phase stress},\
   author={Baptiste Hardy and Stefanie Rauchenzauner and Pascal Fede and Simon Schneiderbauer and Olivier Simonin and Sankaran Sundaresan and Ali Ozel},\
   year={2023},\
   url = {[https://arxiv.org/abs/2401.00179](https://arxiv.org/abs/2401.00179)}, \
}
## Jiang et al.'s ANN Model 
Jiang et al.'s ANM model has been used to generate Figure-3 in the manuscript. This model has been uploaded into the "JiangANNModels" folder and can be also found in https://github.com/yundij/ANN-sub-grid-Drag. 

Bibtex entries for the Jiang's studies:\
@article{jiangPowTec2019,\
   title = {Neural-network-based filtered drag model for gas-particle flows},\
   journal = {Powder Technology},\
   volume = {346},\
   pages = {403-413},\
   year = {2019},\
   doi = {https://doi.org/10.1016/j.powtec.2018.11.092}, \
   url = {https://www.sciencedirect.com/science/article/pii/S0032591018310192#s0045} \
   author = {Baptiste Hardy and Stefanie Rauchenzauner and Pascal Fede and Simon Schneiderbauer and Olivier Simonin and Sankaran Sundaresan and Ali Ozel} \
}

@article{jiangChemEngSci2021, \
title = {Development of data-driven filtered drag model for industrial-scale fluidized beds}, \
journal = {Chemical Engineering Science}, \
volume = {230}, \
pages = {116235}, \
year = {2021}, \
issn = {0009-2509}, \
doi = {https://doi.org/10.1016/j.ces.2020.116235}, \
url = {https://www.sciencedirect.com/science/article/pii/S0009250920307673}, \
author = {Yundi Jiang and Xiao Chen and Jari Kolehmainen and Ioannis G. Kevrekidis and Ali Ozel and Sankaran Sundaresan} \
}



