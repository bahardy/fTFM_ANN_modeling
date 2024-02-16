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
  
  In both cases, the model is shared (and can therefore be uploaded) in two ways:
  * TensorFlow Saved format: the whole model (architecture and weights) is saved in the `model.tf` folder
  * JSON-HDF5 pair format: the model architecture is saved in the `model.config.json` file, the weights are stored in the `model.weights.h5` binary file
 
The features to be fed (in this order) to the network are the following: $\frac{\bar \phi_s}{\phi_{s,max}}$, $\frac{\tilde u_{slip,z}}{U_t}$ , $\frac{\partial \bar p_g}{\partial z}$, $Re_p$, $\frac{\bar \Delta}{d_p Fr_p^{1/3}}$.
The data are automatically shifted by the mean and scaled by the standard deviation of the training sample through a Normalization preprocessing layer embedded in the network.

## Manuscript@arXiv and Bibtex Source
@misc{hardy2023machine,\
   title={Machine learning approaches to close the filtered two-fluid model for gas-solid flows: Models for subgrid drag force and solid phase stress},\
   author={Baptiste Hardy and Stefanie Rauchenzauner and Pascal Fede and Simon Schneiderbauer and Olivier Simonin and Sankaran Sundaresan and Ali Ozel},\
   year={2023},\
   eprint={2401.00179},\
   archivePrefix={arXiv},\
   primaryClass={physics.flu-dyn}
}
## Jiang et al.'s ANN Model 
Jiang et al.'s ANM model has been used to generate Figure-e in the manusctipy. This model has been uploaded into "JiangANNModels" and can be also found in https://github.com/yundij/ANN-sub-grid-Drag. 

Jiang's studies:\
@article{JIANG2019403,\
   title = {Neural-network-based filtered drag model for gas-particle flows},\
   journal = {Powder Technology},\
   volume = {346},\
   pages = {403-413},\
   year = {2019},\
   doi = {https://doi.org/10.1016/j.powtec.2018.11.092},\
   url = {https://www.sciencedirect.com/science/article/pii/S0032591018310192#s0045}\
}\



