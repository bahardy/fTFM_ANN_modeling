# fTFM_ANNmodeling
This project is dedicated to the development of Artificial Neural Network (ANN) models to close the subgrid terms in the filtered Two-Fluid Model, namely the filtered drag force and the solid phase subgrid stresses.

## Dataset 
The dataset used to train ANN models is obtained from fine-grid Two-Fluid simulation results of a tri-periodic gas-solid fluidized bed. Simulations have been performed using the NEPTUNE_CFD software. 
The fine-grid simulation results have been filtered for a range of filter sizes. Only a subset of the full dataset is provided in this repository (about 6%). 

The `data` folder contains the filtered dataset obtained from 10 cases with different physical parameters (see Table in REF or the param.txt file inside each subfolder).

## Models 
### For the filtered drag force
The `filtered_drag` folder contains:
* The Python source code to train and validate ANN models for the filtered drag force (`filtered_drag_ANN.py`). The `terminalVelocity` module contains a routine to calculate the terminal settling velocity of a single isolated particle as a function of physical parameters.
* The `models` subfolder containing the trained models:
    * `DF_generalizedModel_training_cases1to9`: drift flux model obtained from the full dataset 
    * `DF_generalizedModel_training_cases1to9_subset`: drift flux model obtained from the partial dataset shared in this repository
  
  In both cases, the model is shared (and can therefore be uploaded) in two ways
  * TensorFlow Saved format: the whole model (architecture and weights) is saved in the `model.tf` folder
  * JSON-HDF5 combination: the model architecture is saved in the `model.config.json` file, the weights are stored in the `model.weights.h5` binary file  
