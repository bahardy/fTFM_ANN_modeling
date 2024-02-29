#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <tensorflow/c/c_api.h>
#include "cppflow/cppflow.h"
#include "TBNN_prediction.H"

using namespace std;

int main ()
{
  double L1, L2, L3;
  double grad_u[3][3], S[3][3], R[3][3], sigma[3][3], tau[3][3];
  double T1[9], T2[9], T3[9], T4[9];
  double tke; 

  //read characteristic parameters from the input file "input.csv"
  double dp        = readFromFile("input.csv", "particle_diameter");
  double rho_p     = readFromFile("input.csv", "particle_density");
  double phi_s_max = readFromFile("input.csv", "maximum_solid_volume_fraction");
  double Ut        = readFromFile("input.csv", "terminal_velocity");
  double g         = 9.81;
  double Fr_p      = pow(Ut, 2)/dp/g;

  //read flow variables from the input file "input.csv"
  double phi_s     = readFromFile("input.csv", "solid_volume_fraction");
  double delta_f   = readFromFile("input.csv", "filter_size");
  grad_u[0][0]     = readFromFile("input.csv", "dudx");
  grad_u[0][1]     = readFromFile("input.csv", "dudy");
  grad_u[0][2]     = readFromFile("input.csv", "dudz");
  grad_u[1][0]     = readFromFile("input.csv", "dvdx");
  grad_u[1][1]     = readFromFile("input.csv", "dvdy");
  grad_u[1][2]     = readFromFile("input.csv", "dvdz");
  grad_u[2][0]     = readFromFile("input.csv", "dwdx");
  grad_u[2][1]     = readFromFile("input.csv", "dwdy");
  grad_u[2][2]     = readFromFile("input.csv", "dwdz");

  // read subgrid solid stres from input file "output.csv"
  sigma[0][0]        = readFromFile("output.csv", "sig_xx");
  sigma[0][1]        = readFromFile("output.csv", "sig_xy");
  sigma[0][2]        = readFromFile("output.csv", "sig_xz");
  sigma[1][0]        = sigma[0][1];
  sigma[1][1]        = readFromFile("output.csv", "sig_yy");
  sigma[1][2]        = readFromFile("output.csv", "sig_yz");
  sigma[2][0]        = sigma[0][2];
  sigma[2][1]        = sigma[1][2];
  sigma[2][2]        = readFromFile("output.csv", "sig_zz");

  //scale input features
  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      grad_u[i][j]  = grad_u[i][j]*Ut/g;
    }
  }   
  phi_s   = phi_s/phi_s_max;
  delta_f = delta_f/(dp*pow(Fr_p, 1.0/3.0)); 

  //scale model output
  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      sigma[i][j] = sigma[i][j]/(rho_p*pow(Ut,2));
    }
  }

  //compute scaled deviatoric stress tensor 
  tke = .5*compute_trace(sigma);
  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      tau[i][j] = sigma[i][j]/(2*tke); 
    }
      tau[i][i] -= 1./3; 
  }

  //compute scalar and tensor basis to build TBNN 
  compute_strain_rotation_rates(grad_u, S, R);
  compute_scalar_basis(S,R,L1,L2,L3);
  compute_tensor_basis(S,R,T1,T2,T3,T4);

  //define number of features to be fed into TBNN
  int num_scalars = 5;
  int num_tensors = 4; 

  //define input model objects
  //scalar basis
  vector<float> scalar_basis(num_scalars); 
  scalar_basis[0] = (float) L1;
  scalar_basis[1] = (float) L2;
  scalar_basis[2] = (float) L3;
  scalar_basis[3] = (float) phi_s;
  scalar_basis[4] = (float) delta_f;
  auto input_1 = cppflow::tensor(scalar_basis, {1,5});
  
  //tensor basis
  vector<float> tensor_basis(num_tensors*9);
  for (int i = 0; i < 9; i++) 
  {
    tensor_basis[i]    = (float) T1[i];
    tensor_basis[9+i]  = (float) T2[i];
    tensor_basis[18+i] = (float) T3[i];
    tensor_basis[27+i] = (float) T4[i];
  }
  auto input_2 = cppflow::tensor(tensor_basis, {1,4,9});

  //model inference
  cppflow::model model("./models/subgrid_stress_TBNN_model_subset/model.tf");
  auto output = model({{"serving_default_input_1:0", input_1}, {"serving_default_input_2:0", input_2}}, {"StatefulPartitionedCall:0"});

  vector<float> output_pred = output[0].get_data<float>(); 
  cout << "ANN prediction for tau_ij: \n"; 
  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){  
      cout << output_pred[3*i+j] << " ";   
    }
    cout << " \n";
  }

  float RMSE = 0; 
  for (int i=0; i<3; i++){
    for (int j=i; j<3; j++){
      RMSE += pow((output_pred[3*i+j] - tau[i][j]),2);
    }
  }
  RMSE = sqrt(RMSE/6.); 
  cout << "RMSE = " << RMSE << " \n";
}

//function for reading input variables from a file
double readFromFile(string file_name, string keyword)
{
  string key;
  string value;
  ifstream file;

  file.open(file_name);

  bool found_keyword = false;
  while(getline(file,key,',') && !found_keyword)
    {
      getline(file,value,'\n');
      if(keyword == key)
        {
          found_keyword = true;
        }
    }
  file.close();
  return atof(value.c_str());
}

void matmul(double A[3][3], double B[3][3], double C[3][3])
{
  // Initialize result matrix to zero 
  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      C[i][j] = 0;
    }
  }
  // Matrix multiplication
  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      for (int k=0; k<3; k++){
      C[i][j] = C[i][j] + A[i][k]*B[k][j] ;
      }
    }
  }
}

void compute_scalar_basis(double S[3][3], double R[3][3], double &L1, double &L2, double &L3)
{
  L1 = 0;
  L2 = 0;
  L3 = 0;

  double S_square[3][3]; 
  double S_cube[3][3]; 
  double R_square[3][3]; 

  matmul(S, S, S_square);
  matmul(S, S_square, S_cube);
  matmul(R, R, R_square);

  L1 = compute_trace(S_square); 
  L2 = compute_trace(R_square); 
  L3 = compute_trace(S_cube); 
}

void compute_tensor_basis(double S[3][3], double R[3][3], double T1_f[9], double T2_f[9], double T3_f[9], double T4_f[9])
{

  double SR[3][3], RS[3][3], SS[3][3], RR[3][3];  
  double T1[3][3], T2[3][3], T3[3][3], T4[3][3];
  double trSS, trRR; 

  matmul(S, R, SR);
  matmul(R, S, RS);
  matmul(S, S, SS);
  matmul(R, R, RR);

  trSS = compute_trace(SS);
  trRR = compute_trace(RR);

  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      T1[i][j] = S[i][j];
      T2[i][j] = SR[i][j] - RS[i][j]; 
      T3[i][j] = SS[i][j] ;
      T4[i][j] = RR[i][j];
    }
    T3[i][i] = T3[i][i] - trSS/3.;
    T4[i][i] = T4[i][i] - trRR/3.;
  }
  
  // Flat version of tensor basis for inference 
  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      T1_f[3*i+j] = T1[i][j];
      T2_f[3*i+j] = T2[i][j];
      T3_f[3*i+j] = T3[i][j];
      T4_f[3*i+j] = T4[i][j];
    }
  }
}

double compute_trace(double T[3][3])
{
  double trace = T[0][0] + T[1][1] + T[2][2]; 
  return trace; 
}

void compute_strain_rotation_rates(double grad_u[3][3], double S[3][3], double R[3][3])
{
  double trS; 

  // Compute strain rate and rotation rate tensors 
  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      S[i][j]  = .5*(grad_u[i][j]+grad_u[j][i]);
      R[i][j]  = .5*(grad_u[i][j]-grad_u[j][i]);
    }
  } 

  trS = compute_trace(S);
  
  // Deviatoric part of the strain rate tensor 
  for (int i=0; i<3; i++){
    S[i][i]  = S[i][i] - trS/3.; 
  } 
}