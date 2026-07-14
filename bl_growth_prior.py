import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from scipy import stats

import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

import sys
import warnings
from datetime import datetime
from pathlib import Path
import traceback

#geometry 
#just doing a round tube with constant area
r = 45e-3 #m
Dh = 2 * r
A = np.pi * r**2 #m^2
L = 0.5 #m
Cf = 0.003
bl_h = 0.5e-3 #m #this is just half of the height 


#Initial Conditions 
M = 1
P = 7 * 1e6

def eff_Area (bl_growth):
    bl_height = bl_h * bl_growth
    return np.pi * (r - bl_height)**2

def dAdX(bl_growth):
    return 0

#mach num
def mNum(v,a): #mach number 
    M = v/a
    return M

#speed of sound
def soS(T, R,gamma): #solving for a using variable gamma and Cp 
    a = np.sqrt(gamma * R * T)
    return a

#cantera funcs
gas = ct.Solution('h2_air.yaml')

def get_Y(comp_string):
    gas.TPX = 300, 101325, comp_string
    return gas.Y.copy()

Y_air = get_Y("O2:0.21, N2:0.79")

def gas_properties(T,P,Y):
    gas.TPY = T, P, Y
    cp = gas.cp_mass
    h = gas.enthalpy_mass
    gamma = gas.cp_mass/gas.cv_mass
    R_specific = gas.cp_mass - gas.cv_mass
    s = gas.entropy_mass
    return{
        "cp": cp,
        "h": h,
        "gamma": gamma,
        "R_specific": R_specific,
        "s": s
    }



def dVdx (V,A,M,T,P,x,dx,bl_h,Cf,bl_growth):
    gamma = gas_properties(T,P,Y_air)["gamma"]
    A = eff_Area(bl_growth)
    dAdx = dAdX(bl_growth)

    term1 = (- V / ((1 - M**2) * A)) * dAdx
    term2 = ((2 * Cf * gamma * V * M**2)/((1-M**2) * Dh))
    return term1 + term2

def dPdx (V,A,M,T,P,x,dx,bl_h,Cf,bl_growth):
    gamma = gas_properties(T,P,Y_air)["gamma"]
    A = eff_Area(bl_growth)
    dAdx = dAdX(bl_growth)

    term1 = (gamma * M**2 * P)/ ((1 - M**2) * A) * dAdx
    term2 = ((2 * gamma * Cf * M**2 * (1 + (gamma -1) * M**2))/((1 - M**2) * Dh))
    return term1 - term2



def pressureStagFunc(P,M,gamma):
    Pstag = P * (1 + (gamma - 1)/2 * M**2)**(gamma/(gamma-1))
    return Pstag

def temperatureStagFunc(T,M,gamma):
    Tstag = T * (1 + (gamma - 1)/2 * M**2)
    return Tstag

def choked_massFlow(Pstag,Astar,Tstag, gamma):
    mdot_choke = (Pstag * Astar/np.sqrt(Tstag)) * np.sqrt(gamma / R_mix) * ((gamma + 1)/2)**(-(gamma + 1)/(2*(gamma-1)))
    return mdot_choke

def pstag_predicted(mdot,Astar,Tstag,gamma):
    Pstag_pred = mdot * (np.sqrt(Tstag)/Astar) / (np.sqrt(gamma / R_mix) * ((gamma + 1)/2)**(-(gamma + 1)/(2*(gamma-1))))
    return Pstag_pred

def stagtostatic(Pstag,Tstag, M, gamma):
    middleTerm = (1 + ((gamma -1)/2) * M**2)
    P = Pstag * middleTerm **(- gamma/(gamma - 1))
    T = Tstag * middleTerm ** (-1)
    return P,T

def residualT(T_new,T_old,uOld,uNew,P):
    T_gasProperties_old = gas_properties(T_old, P, Y_air)
    T_gasProperties_new = gas_properties(T_new, P, Y_air)
    ht_old = T_gasProperties_old["h"]
    ht_new = T_gasProperties_new["h"]
    term1 = (ht_new - ht_old)
    term2 = (uNew**2 - uOld**2)/2
    term3 = 0
    return term1 + term2 - term3

def newtonRaphson_T(T_Guess, T_old, xOld, uOld, uNew, dx,P,):
    numIters = 0
    tol = 1e-8
    E = residualT(T_Guess, T_old, xOld, uOld, uNew, dx,P)

    while abs(E) >= tol and numIters <= 100:
        deltaT = max(abs(T_Guess)*1e-6, 1e-6)
        dEdT = (residualT(T_Guess + deltaT, T_old, xOld, uOld, uNew,P) - E)/deltaT

        if not np.isfinite(dEdT) or abs(dEdT) < 1e-14:
            raise RuntimeError("Bad temperature Newton derivative")

        lamda = 1.0
        accepted = False

        while lamda > 1e-3:
            T_new = T_Guess - lamda * E/dEdT

            if T_new <= 0 or not np.isfinite(T_new) or T_new > 3*T_old:
                lamda *= 0.5
                continue

            E_new = residualT(T_new, T_old, xOld, uOld, uNew,P,)

            if np.isfinite(E_new) and abs(E_new) < abs(E):
                accepted = True
                break

            lamda *= 0.5

        if not accepted:
            raise RuntimeError("damping for temp Newton Raphson failed")    
        
        T_Guess = T_new
        E = E_new
        
        numIters += 1

    if numIters > 100 or not np.isfinite(T_Guess) or T_Guess <= 0:
        raise RuntimeError("Newton-Raphson solve failed")

    return T_Guess

def pressureResidual(Pstag,P_guess,T,gamma):
    u = (mdot_i * R_mix * T)/(P_guess * preburner_area)
    M = mNum(u, soS(T, R_mix, gamma))
    Pstatic = Pstag / (1 + 0.5 * (gamma - 1) * M**2)**(gamma/(gamma-1))

    return Pstatic - P_guess

def newtonRaphson_P(P_guess, Pstag, T, gamma):
    numIters = 0
    tol = 1e-8
    E = pressureResidual(Pstag, P_guess, T, gamma)

    while abs(E) >= tol and numIters <= 100:
        deltaP = max(abs(P_guess)*1e-6, 1e-6)
        dEdP = (pressureResidual(Pstag, P_guess + deltaP, T, gamma) - E)/deltaP

        if not np.isfinite(dEdP) or abs(dEdP) < 1e-14:
            raise RuntimeError("Bad pressure Newton derivative")

        lamda = 1.0
        accepted = False

        while lamda > 1e-3:
            P_new = P_guess - lamda * E/dEdP
            if P_new <= 0 or not np.isfinite(P_new) or P_new > 3*Pstag:
                lamda *= 0.5
                continue
            E_new = pressureResidual(Pstag, P_new, T, gamma)
            if np.isfinite(E_new) and abs(E_new) < abs(E):
                accepted = True
                break
            lamda *= 0.5
        if not accepted:
            raise RuntimeError("damping for pressure Newton Raphson failed")    
        
        P_guess = P_new
        E = E_new
        numIters += 1

    if numIters > 100 or not np.isfinite(P_guess) or P_guess <= 0:
        raise RuntimeError("Newton-Raphson solve failed")

    return P_guess

#rk45

def rk45Step(V,P, T_preburner,h, x, Cf,bl_growth): #add stages for each mdot 3
    accepted = False 

    local_tol = 1e-6
    h_max =1e-3

    h = min(h,h_max)
    attempts = 0

    while accepted != True:
        attempts += 1

        if attempts > 200:
            raise RuntimeError(
                f"rk45Step failed to accept step: "
                f"x={x}, h={h}, V={V}, P={P}, T={T_preburner}, ")

        if h < 1e-14:
            raise RuntimeError(f"RK45 step size got too small at x = {x:.4f}")
            
        
        x1 = x
        Cf1 = cf_location(x1,Cf_dnz)
        A1 = geom_Area(x1,bl_h,bl_growth)
        V1 = V
        P1 =  P

        try:
            T1 = T_preburner
            a1 = soS(T1,R_mix,gas_properties(T1, P1, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue
        mdot_1Cur = mdotFuncX(x1)
        mdot_1Prev = mdotFuncX(x-h)  
        M1 = mNum(V1,a1)
        k1V = h * dVdX(V1,A1,M1,T1,P1,mdot_1Cur,delMdotdx(mdot_1Cur,mdot_1Prev,x1,x1-h),Cf1,x1,h,eta_total,combustion_end,bl_h,bl_growth)
        k1P = h * dPdX(V1,A1,M1,T1,P1,mdot_1Cur,delMdotdx(mdot_1Cur,mdot_1Prev,x1,x1-h),Cf1,x1,h,eta_total,combustion_end,bl_h,bl_growth)

        x2 = x1 + 1/5 * h
        Cf2 = cf_location(x2,Cf_dnz)
        A2 = geom_Area(x2,bl_h,bl_growth)
        V2 = V + 1/5 * k1V 
        P2 = P + 1/5 * k1P
        try:
            T2 = newtonRaphson_T(T1, T1, x1, V1, V2, 1/5 * h,eta_total,P2,combustion_end)
            a2 = soS(T2,R_mix,gas_properties(T2, P2, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue
        M2 = mNum(V2,a2)
        mdot_2Cur = mdotFuncX(x2)
        mdot_2Prev = mdotFuncX(x1) 
        k2V = h * dVdX(V2,A2,M2,T2,P2,mdot_2Cur,delMdotdx(mdot_2Cur,mdot_2Prev,x2,x1),Cf2,x2,1/5 * h,eta_total,combustion_end,bl_h,bl_growth)
        k2P = h * dPdX(V2,A2,M2,T2,P2,mdot_2Cur,delMdotdx(mdot_2Cur,mdot_2Prev,x2,x1),Cf2,x2,1/5 * h,eta_total,combustion_end,bl_h,bl_growth)

        x3 = x1 + 3/10 * h
        Cf3 = cf_location(x3,Cf_dnz)

        A3 = geom_Area(x3,bl_h,bl_growth)
        V3 = V + 3/40 * k1V + 9/40 * k2V
        P3 = P + 3/40 * k1P + 9/40 * k2P
        try:
            T3 = newtonRaphson_T(T1, T1, x1, V1, V3, 3/10 * h,eta_total,P3,combustion_end) 
            a3 = soS(T3,R_mix,gas_properties(T3, P3, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue
        M3 = mNum(V3,a3)
        mdot_3Cur = mdotFuncX(x3)
        mdot_3Prev = mdotFuncX(x1) 
        k3V = h * dVdX(V3,A3,M3,T3,P3,mdot_3Cur,delMdotdx(mdot_3Cur,mdot_3Prev,x3,x2),Cf3,x3,3/10 * h,eta_total,combustion_end,bl_h,bl_growth)
        k3P = h * dPdX(V3,A3,M3,T3,P3,mdot_3Cur,delMdotdx(mdot_3Cur,mdot_3Prev,x3,x2),Cf3,x3,3/10 * h,eta_total,combustion_end,bl_h,bl_growth)

        x4 = x1 + 4/5 * h
        Cf4 = cf_location(x4,Cf_dnz)

        A4 = geom_Area(x4,bl_h,bl_growth)
        V4 = V + 44/45 * k1V - 56/15 * k2V + 32/9 * k3V
        P4 = P + 44/45 * k1P - 56/15 * k2P + 32/9 * k3P
        try:
            T4 = newtonRaphson_T(T1, T1, x1, V1, V4, 4/5 * h,eta_total,P4,combustion_end) 
            a4 = soS(T4,R_mix,gas_properties(T4, P4, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue
        M4 = mNum(V4,a4)
        mdot_4Cur = mdotFuncX(x4)
        mdot_4Prev = mdotFuncX(x1) 
        k4V = h * dVdX(V4,A4,M4,T4,P4,mdot_4Cur,delMdotdx(mdot_4Cur,mdot_4Prev,x4,x3),Cf4,x4,4/5 * h,eta_total,combustion_end,bl_h,bl_growth)
        k4P = h * dPdX(V4,A4,M4,T4,P4,mdot_4Cur,delMdotdx(mdot_4Cur,mdot_4Prev,x4,x3),Cf4,x4,4/5 * h,eta_total,combustion_end,bl_h,bl_growth)

        x5 = x1 + 8/9 * h
        Cf5 = cf_location(x5,Cf_dnz)
        A5 = geom_Area(x5,bl_h,bl_growth)
        V5 = V + 19372/6561 * k1V - 25360/2187 * k2V + 64448/6561 * k3V - 212/729 * k4V
        P5 = P + 19372/6561 * k1P - 25360/2187 * k2P + 64448/6561 * k3P - 212/729 * k4P
        try:
            T5 = newtonRaphson_T(T1, T1, x1, V1, V5, 8/9 * h,eta_total,P5,combustion_end)
            a5 = soS(T5,R_mix,gas_properties(T5, P5, Y_mix)["gamma"])

        except:
            h *= 0.5
            continue

        M5 = mNum(V5,a5)
        mdot_5Cur = mdotFuncX(x5)
        mdot_5Prev = mdotFuncX(x1) 
        k5V = h * dVdX(V5,A5,M5,T5,P5,mdot_5Cur,delMdotdx(mdot_5Cur,mdot_5Prev,x5,x1),Cf5,x5,8/9 * h,eta_total,combustion_end,bl_h,bl_growth)
        k5P = h * dPdX(V5,A5,M5,T5,P5,mdot_5Cur,delMdotdx(mdot_5Cur,mdot_5Prev,x5,x1),Cf5,x5,8/9 * h,eta_total,combustion_end,bl_h,bl_growth)

        x6 = x1 + h
        Cf6 = cf_location(x6,Cf_dnz)
        A6 = geom_Area(x6,bl_h,bl_growth)
        V6 = V + 9017/3168 * k1V - 355/33 * k2V + 46732/5247 * k3V + 49/176 * k4V - 5103/18656 * k5V
        P6 = P + 9017/3168 * k1P - 355/33 * k2P + 46732/5247 * k3P + 49/176 * k4P - 5103/18656 * k5P
        try:
            T6 = newtonRaphson_T(T1, T1, x1, V1, V6, 1 * h,eta_total,P6,combustion_end)
            a6 = soS(T6,R_mix,gas_properties(T6, P6, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue

        M6 = mNum(V6,a6)
        mdot_6Cur = mdotFuncX(x6)
        mdot_6Prev = mdotFuncX(x1) 
        k6V = h * dVdX(V6,A6,M6,T6,P6,mdot_6Cur,delMdotdx(mdot_6Cur,mdot_6Prev,x6,x1),Cf6,x6,h,eta_total,combustion_end,bl_h,bl_growth)
        k6P = h * dPdX(V6,A6,M6,T6,P6,mdot_6Cur,delMdotdx(mdot_6Cur,mdot_6Prev,x6,x1),Cf6,x6,h,eta_total,combustion_end,bl_h,bl_growth)

        #5th order solution 
        v_5Order = V + 35/384 * k1V + 500/1113 * k3V + 125/192 * k4V - 2187/6784 * k5V + 11/84 * k6V
        p_5Order = P + 35/384 * k1P + 500/1113 * k3P + 125/192 * k4P - 2187/6784 * k5P + 11/84 * k6P

        x7 = x1 + h
        Cf7 = cf_location(x7,Cf_dnz)
        A7 = geom_Area(x7,bl_h,bl_growth)
        V7 = v_5Order
        P7 = p_5Order
        try:
            T7 = newtonRaphson_T(T1, T1, x1, V1, V7, 1 * h,eta_total,P7,combustion_end)
            a7 = soS(T7,R_mix,gas_properties(T7, P7, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue       
        M7 = mNum(V7,a7)
        mdot_7Cur = mdotFuncX(x7)
        mdot_7Prev = mdotFuncX(x1) 
        k7V = h * dVdX(V7,A7,M7,T7,P7,mdot_7Cur,delMdotdx(mdot_7Cur,mdot_7Prev,x7,x1),Cf7,x7,h,eta_total,combustion_end,bl_h,bl_growth)
        k7P = h * dPdX(V7,A7,M7,T7,P7,mdot_7Cur,delMdotdx(mdot_7Cur,mdot_7Prev,x7,x1),Cf7,x7,h,eta_total,combustion_end,bl_h,bl_growth)

        #4th order solution
        v_4Order = V + 5179/57600 * k1V + 7571/16695 * k3V + 393/640 * k4V - 92097/339200 * k5V + 187/2100 * k6V + 1/40 * k7V
        p_4Order = P + 5179/57600 * k1P + 7571/16695 * k3P + 393/640 * k4P - 92097/339200 * k5P + 187/2100 * k6P + 1/40 * k7P

        #error estimate 
        errorV = abs(v_5Order - v_4Order)
        errorP = abs(p_5Order - p_4Order)

        errorRatioV = errorV/(abs(V) * local_tol)
        errorRatioP = errorP/(abs(P) * local_tol)
        errorRatio = max(errorRatioP,errorRatioV)

        if errorRatio > 1: 
            accepted = False 

            s = 0.5 * errorRatio**(-1/5)
            h = min(s * h, h_max)
            continue #this just restarts the loop with the updated h value
            
        else: #if both are smaller than tol then I am accepting the time step and then making it bigger 
            accepted = True

            Vnext, Pnext = v_5Order, p_5Order
            xNext = x1 + h
            Tnext = newtonRaphson_T(T1, T1, x1, V1, Vnext, 1 * h,eta_total,Pnext,combustion_end)

            s = 1.2 if errorRatio == 0 else min(2, 0.9 * errorRatio**(-1/5))

            h_next = min(s * h, h_max)
            break
    
    return xNext, Vnext, Pnext, Tnext,h_next, location


