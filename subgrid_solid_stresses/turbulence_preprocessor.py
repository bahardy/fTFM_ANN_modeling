import numpy as np
from preprocessor import DataProcessor

"""
This file is adapted from the public repository https://github.com/sandialabs/tbnn/tree/master 
owned by Sandia Corporation. 

Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000,
there is a non-exclusive license for use of this work by or on behalf of the U.S. Government.
This software is distributed under the BSD-3-Clause license.
"""

class TurbulenceDataProcessor(DataProcessor):
    """
    Inherits from DataProcessor class.  This class is specific to processing turbulence data to predict
    the anisotropy tensor based on the mean strain rate (Sij) and mean rotation rate (Rij) tensors
    """
    @staticmethod
    def calc_Sij_Rij(grad_u, cap=float("inf")):
        """
        Calculates the strain rate and rotation rate tensors. 
        Sij = 0.5* (grad_u  + grad_u^T)
        Rij = 0.5* (grad_u  - grad_u^T)
        :param grad_u: num_points X 3 X 3
        :return: Sij, Rij: num_points X 3 X 3 tensors
        """

        num_points = grad_u.shape[0]
        Sij = np.zeros((num_points, 3, 3))
        Rij = np.zeros((num_points, 3, 3))
        for i in range(num_points):
            Sij[i, :, :] = 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :])) - 1./3. * np.eye(3)*np.trace(Sij[i, :, :])
            Rij[i, :, :] = 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        return Sij, Rij

    def calc_scalar_basis(self, Sij, Rij, num_invariants, is_train=False, is_scale=True):
        """
        Given the non-dimensionalized mean strain rate and mean rotation rate tensors Sij and Rij,
        this returns a set of normalized scalar invariants
        :param Sij: 0.5 * (du_i/dx_j + du_j/dx_i)
        :param Rij: 0.5 * (du_i/dx_j - du_j/dx_i)
        :param num_invariants: Number of scalar invariants to keep in the basis
        :return: invariants: The num_points X num_scalar_invariants numpy matrix of scalar invariants
        """
        DataProcessor.calc_scalar_basis(self, Sij, is_train=is_train)
        num_points = Sij.shape[0]
        # num_invariants = 5
        invariants = np.zeros((num_points, num_invariants))

        if (num_invariants == 1): # Use only first invariants tr(S2)
            for i in range(num_points):
                invariants[i, 0] = np.trace(np.matmul(Sij[i, :, :], Sij[i, :, :]))

        elif(num_invariants == 2): 
            for i in range(num_points): #Use invariants linked to S, i.e. tr(S2), tr(S3)
                invariants[i, 0] = np.trace(np.matmul(Sij[i, :, :], Sij[i, :, :]))
                invariants[i, 1] = np.trace(np.matmul(Sij[i, :, :], np.matmul(Sij[i, :, :], Sij[i, :, :])))

        elif(num_invariants == 3): 
            for i in range(num_points): #Use invariants : tr(S2), tr(R2), tr(S3)
                invariants[i, 0] = np.trace(np.matmul(Sij[i, :, :], Sij[i, :, :]))
                invariants[i, 1] = np.trace(np.matmul(Rij[i, :, :], Rij[i, :, :]))
                invariants[i, 2] = np.trace(np.matmul(Sij[i, :, :], np.matmul(Sij[i, :, :], Sij[i, :, :])))

        elif(num_invariants == 5): 
            for i in range(num_points): #Use invariants linked to S and R 
                invariants[i, 0] = np.trace(np.matmul(Sij[i, :, :], Sij[i, :, :]))
                invariants[i, 1] = np.trace(np.matmul(Rij[i, :, :], Rij[i, :, :]))
                invariants[i, 2] = np.trace(np.matmul(Sij[i, :, :], np.matmul(Sij[i, :, :], Sij[i, :, :])))
                invariants[i, 3] = np.trace(np.matmul(Rij[i, :, :], np.matmul(Rij[i, :, :], Sij[i, :, :])))
                invariants[i, 4] = np.trace(np.matmul(np.matmul(Rij[i, :, :], Rij[i, :, :]), np.matmul(Sij[i, :, :], Sij[i, :, :])))
       
        return invariants

    def calc_tensor_basis(self, Sij, Rij, num_tensor_basis=4, is_scale=False):
        """
        Given Sij and Rij, it calculates the tensor basis
        :param Sij: scaled strain rate tensor
        :param Rij: scaled rotation rate tensor
        :param num_tensor_basis: number of tensors to preserved in the tensor basis 
        :return: T_flat: num_points X num_tensor_basis X 9 numpy array of tensor basis.
                        Ordering is 11, 12, 13, 21, 22, ...
        """
        num_points = Sij.shape[0]

        T = np.zeros((num_points, num_tensor_basis, 3, 3))
        for i in range(num_points):
            sij = Sij[i, :, :]
            rij = Rij[i, :, :]
            T[i, 0, :, :] = sij
            if(num_tensor_basis==2):
                T[i, 0, :, :] = sij
                T[i, 1, :, :] = np.dot(sij, sij) - 1./3.*np.eye(3)*np.trace(np.dot(sij, sij))
            if(num_tensor_basis > 2):
                T[i, 0, :, :] = sij
                T[i, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
                T[i, 2, :, :] = np.dot(sij, sij) - 1./3.*np.eye(3)*np.trace(np.dot(sij, sij))
            if(num_tensor_basis > 3):
                T[i, 3, :, :] = np.dot(rij, rij) - 1./3.*np.eye(3)*np.trace(np.dot(rij, rij))
            if(num_tensor_basis > 4):
                T[i, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
                T[i, 5, :, :] = np.dot(rij, np.dot(rij, sij)) \
                                + np.dot(sij, np.dot(rij, rij)) \
                                - 2./3.*np.eye(3)*np.trace(np.dot(sij, np.dot(rij, rij)))
                T[i, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
                T[i, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
                T[i, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) \
                                + np.dot(np.dot(sij, sij), np.dot(rij, rij)) \
                                - 2./3.*np.eye(3)*np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
                T[i, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) \
                                - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))
            # Enforce zero trace for anisotropy
            for j in range(num_tensor_basis):
                T[i, j, :, :] = T[i, j, :, :] - 1./3.*np.eye(3)*np.trace(T[i, j, :, :])

        # Scale down to promote convergence
        if is_scale:
            scale_factor = [10, 100, 100, 100, 1000, 1000, 10000, 10000, 10000, 10000]
            for i in range(num_tensor_basis):
                T[:, i, :, :] /= scale_factor[i]

        # Flatten:
        T_flat = np.zeros((num_points, num_tensor_basis, 9))
        for i in range(3):
            for j in range(3):
                T_flat[:, :, 3*i+j] = T[:, :, i, j]
        return T_flat

    def calc_output(self, stresses):
        """
        Given subgrid stress tensor (num_points X 3 X 3), return flattened non-dimensional anisotropy tensor
        :param stresses: subgrid stress tensor
        :return: anisotropy_flat: (num_points X 9) anisotropy tensor.  aij = (uiuj)/2k - 1./3. * delta_ij
        """
        num_points = stresses.shape[0]
        anisotropy = np.zeros((num_points, 3, 3))

        for i in range(3):
            for j in range(3):
                tke = 0.5 * (stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2])
                tke = np.maximum(tke, 1e-8)
                anisotropy[:, i, j] = stresses[:, i, j]/(2.0 * tke)
            anisotropy[:, i, i] -= 1./3.
        anisotropy_flat = np.zeros((num_points, 9))
        for i in range(3):
            for j in range(3):
                anisotropy_flat[:, 3*i+j] = anisotropy[:, i, j]
        return anisotropy_flat
    
    def calc_dev_stresses(self, stresses):
        """
        Given the full subgrid stress tensor (num_points X 3 X 3), return its deviatoric part 
        :param stresses: subgrid stress tensor
        :return: tau: (num_points X 3 x 3) tau_ij = sigma_ij - 1/3 sigma_kk delta_ij 
        """
        num_points = stresses.shape[0]
        tau = np.zeros((num_points, 3, 3))

        for i in range(3):
            for j in range(3):
                tke = 0.5 * (stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2])
                tke = np.maximum(tke, 1e-8)
                tau[:, i, j] = stresses[:, i, j]
            tau[:, i, i] -= 2./3.*tke

        return tau
    
    def calc_flatten_tensor(self, tau):
        """
        Given a symmetric second-order tensor (num_points X 3 X 3), return the flatten equivalent (num_points X 6)
        :return: tau_flat: (num_points X 6) tau_ij
        """
        num_points = tau.shape[0]
        tau_flat = np.zeros((num_points, 6))

        tau_flat[:,0]  = tau[:,0,0]  
        tau_flat[:,1]  = tau[:,0,1]
        tau_flat[:,2]  = tau[:,0,2]
        tau_flat[:,3]  = tau[:,1,1]
        tau_flat[:,4]  = tau[:,1,2]
        tau_flat[:,5]  = tau[:,2,2]

        return tau_flat
    
    def calc_eddy_viscosity(self, stresses, strain_rate):
        """
        Given the sub-grid stress and filtered strain-rate tensors (num_points X 3 X 3), returns the local eddy viscosity
        :param stresses: sub-grid stress tensor
        :param strain_rate: filtered srain-rate tensor
        :return: mu_s (num_points,) : mu_s = sqrt(tau:tau)/(2*sqrt(S:S))
        """
        num_points = stresses.shape[0]
        mu_s       = np.zeros((num_points,))

        for i in range(num_points):
            stresses_norm = np.sqrt(np.trace(np.matmul(stresses[i, :, :], stresses[i, :, :])))
            S_norm        = np.sqrt(np.trace(np.matmul(strain_rate[i, :, :], strain_rate[i, :, :])))
            mu_s[i]       = stresses_norm/(2*S_norm)

        return mu_s

    def calc_mesoscale_pressure(self, stresses):
        """
        Given the sub-grid stress, returns the mesoscale pressure 
        :param stresses: sub-grid stress tensor
        :return: P_s (num_points,) : P_s = (1./3.)*trace(sigma_sgs)
        """     
        
        Ps = (stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2])/3.
        Ps = np.maximum(Ps, 1e-8)
        return Ps


    @staticmethod
    def calc_rans_anisotropy(grad_u, tke, eps):
        """
        Calculate the Reynolds stress anisotropy tensor (num_points X 9) that RANS would have predicted
        given a linear eddy viscosity hypothesis: a_ij = -2*nu_t*Sij/(2*k) = - C_mu * k / eps * Sij
        :param grad_u: velocity gradient tensor
        :param tke: turbulent kinetic energy
        :param eps: turbulent dissipation rate
        :return: rans_anisotropy
        """
        sij, _ = TurbulenceDataProcessor.calc_Sij_Rij(grad_u, tke, eps, cap=np.infty)
        c_mu = 0.09

        # Calculate anisotropy tensor (num_points X 3 X 3)
        # Note: Sij is already non-dimensionalized with tke/eps
        rans_anisotropy_matrix = - c_mu * sij

        # Flatten into num_points X 9 array
        num_points = sij.shape[0]
        rans_anisotropy = np.zeros((num_points, 9))
        for i in range(3):
            for j in range(3):
                rans_anisotropy[:, i*3+j] = rans_anisotropy_matrix[:, i, j]
        return rans_anisotropy

    @staticmethod
    def make_realizable(labels):
        """
        This function is specific to turbulence modeling.
        Given the anisotropy tensor, this function forces realizability
        by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
        Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
        constraints from first step, so this function should be called iteratively to get convergence
        to a realizable state.
        :param labels: the predicted anisotropy tensor (num_points X 9 array)
        """
        numPoints = labels.shape[0]
        A = np.zeros((3, 3))
        for i in range(numPoints):
            # Scales all on-diags to retain zero trace
            if np.min(labels[i, [0, 4, 8]]) < -1./3.:
                labels[i, [0, 4, 8]] *= -1./(3.*np.min(labels[i, [0, 4, 8]]))
            if 2.*np.abs(labels[i, 1]) > labels[i, 0] + labels[i, 4] + 2./3.:
                labels[i, 1] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
                labels[i, 3] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
            if 2.*np.abs(labels[i, 5]) > labels[i, 4] + labels[i, 8] + 2./3.:
                labels[i, 5] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
                labels[i, 7] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
            if 2.*np.abs(labels[i, 2]) > labels[i, 0] + labels[i, 8] + 2./3.:
                labels[i, 2] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
                labels[i, 6] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])

            # Enforce positive semidefinite by pushing evalues to non-negative
            A[0, 0] = labels[i, 0]
            A[1, 1] = labels[i, 4]
            A[2, 2] = labels[i, 8]
            A[0, 1] = labels[i, 1]
            A[1, 0] = labels[i, 1]
            A[1, 2] = labels[i, 5]
            A[2, 1] = labels[i, 5]
            A[0, 2] = labels[i, 2]
            A[2, 0] = labels[i, 2]
            evalues, evectors = np.linalg.eig(A)
            if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/2.:
                evalues = evalues*(3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/(2.*np.max(evalues))
                A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
                for j in range(3):
                    labels[i, j] = A[j, j]
                labels[i, 1] = A[0, 1]
                labels[i, 5] = A[1, 2]
                labels[i, 2] = A[0, 2]
                labels[i, 3] = A[0, 1]
                labels[i, 7] = A[1, 2]
                labels[i, 6] = A[0, 2]
            if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
                evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
                A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
                for j in range(3):
                    labels[i, j] = A[j, j]
                labels[i, 1] = A[0, 1]
                labels[i, 5] = A[1, 2]
                labels[i, 2] = A[0, 2]
                labels[i, 3] = A[0, 1]
                labels[i, 7] = A[1, 2]
                labels[i, 6] = A[0, 2]

        return labels
