# This file contains a routine to compute the terminal settling velocity of a free particle at finite Reynolds number  
# This file is part of the fTFM_ANN_modeling project (https://github.com/bahardy/fTFM_ANN_modeling.git) distributed under BSD-3-Clause license 
# Copyright (c) Baptiste HARDY. All rights reserved.

import math 

def getTerminalVelocity(rho_s, rho_g, dp, g, mu):
    CD = 1 #start value for the drag coefficient in the iterative process 
    tol = 1e-8
    delta = 2*tol
    count = 0
    count_max = 200
    while (delta > tol) & (count < count_max): 
        Ut = math.sqrt(4/3*(rho_s-rho_g)/rho_g*dp*g/CD)
        Ret = rho_g*Ut*dp/mu 
        if(Ret < 1000):
            CD_new = 24/Ret * (1+0.15*Ret**0.687) #Schiller-Naumann drag relation
        else:
            CD_new = 0.44
        delta = abs(CD_new-CD)
        CD = CD_new
        count = count+1 
        #print("Ut = {:1.3f}, Ret = {:1.3f}, Cd = {:1.3f} \n".format(Ut, Ret, CD))
    print("Iterative process has converged in {:d} iterations".format(count))    
    return Ut 
