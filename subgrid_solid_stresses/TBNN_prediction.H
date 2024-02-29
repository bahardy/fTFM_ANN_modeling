#include <string>
using namespace std;

double readFromFile(string file_name, string keyword);

void matmul(double A[3][3], double B[3][3], double C[3][3]); 

void compute_scalar_basis(double S[3][3], double R[3][3], double &L1, double &L2, double &L3); 

void compute_tensor_basis(double S[3][3], double R[3][3], double T1_f[9], double T2_f[9], double T3_f[9], double T4_f[9]); 

void compute_strain_rotation_rates(double grad_u[3][3], double S[3][3], double R[3][3]); 

double compute_trace(double T[3][3]); 
