import numpy as np
import matplotlib.pyplot as plt

import cantera as ct

from scipy.optimize import fsolve
from scipy import stats
from scipy.optimize import least_squares

import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

import sys
import warnings
from datetime import datetime
from pathlib import Path
import traceback
warnings.filterwarnings("ignore", category=FutureWarning)

Vinj = None 
injMdot = 0

#geometry 
preburner_area = 45e-3 * 45e-3 #m2 
preburner_length = 0.42 #m

nozzle_area_ratio = 25 
conv_Nozzle_length = 0.08 #m 
div_Nozzle_length = 0.140 #m

throat_loc = preburner_length + conv_Nozzle_length
nozzle_exit = throat_loc + div_Nozzle_length
throat_Area = preburner_area / nozzle_area_ratio
exit_Area = preburner_area
x_injLocation = 0.15 * preburner_length #m


def smoothstep(xi):
    return 6*xi**5 - 15*xi**4 + 10*xi**3

def geometry_regions(x):
    if x <= preburner_length:
        return "Preburner"
    elif throat_loc <= x <= throat_loc + 0.005: 
        return "Throat"
    elif x <= throat_loc:
        return "Conv Nozzle"
    elif x <= nozzle_exit:
        return "Div Nozzle"
    else:
        return "Test Section"

def geom_Area(x):
    if x <= preburner_length:
        return preburner_area
    elif x <= throat_loc:
        xi = (x - preburner_length)/conv_Nozzle_length
        return preburner_area + smoothstep(xi) * (throat_Area - preburner_area)
    elif x <= nozzle_exit:
        xi = (x - throat_loc)/(div_Nozzle_length)
        return throat_Area + smoothstep(xi) * (exit_Area - throat_Area)
    else:
        return exit_Area
    
def Dh(x):
    return np.sqrt(geom_Area(x))

def dAdx(x, tol = 1e-3):
    region = geometry_regions(x)
    if region == "Preburner" or region == "Test Section":
        return 0.0
    elif region == "Throat" or region == "Conv Nozzle" or region == "Div Nozzle":
        xCurrent = x
        xPrev = x - max((x*tol),1e-9)
        dA = geom_Area(xCurrent) - geom_Area(xPrev)
        return dA/(xCurrent - xPrev)

def pressureTap(x_old, p_old, x_new, p_new, PT_locations):
    location = None
    p_tap = None

    for location in PT_locations[:]:
        locationCrossed = (x_old <= location <= x_new)

        if locationCrossed:
            if x_new != x_old:
                frac = (location - x_old)/(x_new - x_old)
                p_tap = p_old + frac  * (p_new - p_old)

                PT_locations.remove(location)
                return p_tap,x_new 
    
    return None,None
    
def cf_location(x,Cf_dnz, Cf_cnz):
    region = geometry_regions(x)
    Cf_preburner = 0.0025
    if region == "Preburner":
        return Cf_preburner
    
    elif region == "Throat" or region == "Conv Nozzle":
        return Cf_cnz
    elif region == "Div Nozzle" or region == "Test Section":
        return Cf_dnz    
    

def mNum(v,a): #mach number 
    M = v/a
    return M

def soS(T, R,gamma): #solving for a using variable gamma and Cp 
    a = np.sqrt(gamma * R * T)
    return a

#Species and species properties

gas = ct.Solution('h2_air.yaml')
# Y is mass fraction for cantera 

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

def get_Y(comp_string):
    gas.TPX = 300, 101325, comp_string
    return gas.Y.copy()

def Ymix(YA,mdotAir, YB, mdotH2):
    Ymix = (mdotAir * YA + mdotH2 * YB)/(mdotAir + mdotH2)
    return Ymix

#smarts model
hpr_h2 = 120e6 #J/kg
fst = 0.029
phi = 0.2306
theta = 1
x_react = 0.1
def x_norm(x,combustion_end):
    return (x - x_react)/(combustion_end - x_react)

def eta(x,eta_total,combustion_end):
    eta = eta_total * (theta * x_norm(x,combustion_end)/(1 + (theta - 1) * x_norm(x,combustion_end)))
    return eta

def dPHI(x,dx,eta_total,combustion_end):
    xcurrent = x
    xprev = x - dx
    dPHI = phi * (eta(xcurrent,eta_total,combustion_end) - eta(xprev,eta_total,combustion_end))
    return dPHI

def dHtdx(x,dx,eta_total,combustion_end):
    if x <= preburner_length:
        return (dPHI(x,dx,eta_total,combustion_end) * hpr_h2 * fst)/dx
    else:
        return 0

def residualT(T_new,T_old,xOld,uOld,uNew,dx,eta_total,P,combustion_end):
    T_gasProperties_old = gas_properties(T_old, P, Y_mix)
    T_gasProperties_new = gas_properties(T_new, P, Y_mix)
    ht_old = T_gasProperties_old["h"]
    ht_new = T_gasProperties_new["h"]
    term1 = (ht_new - ht_old)
    term2 = (uNew**2 - uOld**2)/2
    term3 = dHtdx(xOld,dx,eta_total,combustion_end) * dx
    return term1 + term2 - term3

#initial conditions

#using pb3
dir_air = 0.229/39.37 #meters
d_h2 = 0.034/39.37  #meters

A_airInjs = (np.pi * (dir_air/2)**2) 
A_H2Injs = (np.pi * (d_h2/2)**2)

P1 = 7.708339*1e6 #Pa
P2 = 7.708339*1e6 #Pa
P3 = 8.315077*1e6 #Pa

P_air = P1 #Pa. Pa is equal to P1 and P2 because they are the same injector and they are connected to the same plenum.
P_H2 = P3 #Pa

T_air = 300 #K
T_air2= 300 #K
T_H2 = 300 #K

M1 = 0.999
M2 = 0.999
M3 = 0.999

mdot1_Air = 0.4430/2
mdot2_Air = 0.4430/2
mdot3_H2 = 0.003

mdotAir = mdot1_Air + mdot2_Air #injector 1 and 2 are the same so can just add them together
mdotH2 = mdot3_H2 #big injector
mdot_i = mdotAir + mdotH2

Y_air = get_Y("O2:0.21, N2:0.79")
Y_H2 = get_Y("H2:1.0")
Y_mix = Ymix(Y_air, mdotAir, Y_H2, mdotH2)

R_air = gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["R_specific"]
R_H2 = gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["R_specific"]
R_mix = gas_properties(300, 101325, Y_mix)["R_specific"]
#print("R_air", R_air, "R_H2", R_H2, "R_mix", R_mix)

a1 = soS(T_air, R_air, gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"])
a2 = soS(T_air2, R_air, gas_properties(T_air2, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"])
a3 = soS(T_H2, R_H2, gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"])
#print("a1", a1, "a2", a2, "a3", a3)

uA = M1 * a1
uA_2 = M2 * a2
uB = M3 * a3
#print("uA", uA, "uA_2", uA_2, "uB", uB)

TstagA = T_air * (1 + ((gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"] - 1)/2) * M1**2)
TstagB = T_H2 * (1 + ((gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"] - 1)/2) * M3**2)

Pstag_Air = P_air * (1 + (gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"] - 1)/2 * M1**2)**(gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"]/(gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"]-1))
Pstag_H2 = P_H2 * (1 + (gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"] - 1)/2 * M3**2)**(gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"]/(gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"]-1))

rho1 = mdot1_Air/(A_airInjs * uA)
rho2 = mdot2_Air/(A_airInjs * uA_2)
rho3 = mdot3_H2/(A_H2Injs * uB)

M_h2 = (mdot3_H2/(rho3 * A_H2Injs))/a3
#print("M_h2", M_h2)
#print("rho1", rho1, "rho2", rho2, "rho3", rho3)

A_CV_END = preburner_area #area at the end of the CV is the same as the area at the start of the preburner inlet.
    
def delMdotdx(mdotn1, mdotn,x1,x): #dmdot/dx function
    return (mdotn1 - mdotn)/(x1-x)

def mdotFuncX (x):
    if x < x_injLocation: #pre injector mdot
        return mdot_i
    else:   #post injector mdot
        return mdot_i + injMdot 




#1st order ODE Functions
def dVdX (V,A,M,T,P,mdot,dmdotDX, Cf, x,dx,eta_total,combustion_end): #first 4 parts of sharpios 1d flow eqn converted to dV/dx

    gas_Prop = gas_properties(T, P, Y_mix)
    cp = gas_Prop["cp"]
    gamma = gas_Prop["gamma"]

    term1 = ((-V)/(A * (1 - M**2)))* dAdx(x)
    term2 = ((V/((1-M**2) * cp * T)) * dHtdx(x,dx,eta_total,combustion_end))
    term3 = ((gamma *M**2)/(2 * (1 - M**2)))
    term4 = ((((4 * Cf * V)/Dh(x))) - (2*(Vinj/mdot) * dmdotDX))
    term5 = (((V*(1 + gamma * M**2))/((1-M**2)*mdot)) * (dmdotDX))
    return term1 + term2 + (term3*term4) + term5

def dPdX (V,A,M,T,P,mdot,dmdotDX, Cf, x,dx,eta_total,combustion_end): #first 4 parts of sharpios 1d flow eqn converted to dP/dx
    gas_Prop = gas_properties(T, P, Y_mix)
    cp = gas_Prop["cp"]
    gamma = gas_Prop["gamma"]


    term1 = ((gamma * M**2 * P)/(A * (1 - M**2))) * dAdx(x)
    term2 = -(((gamma * M**2 * P)/((1-M**2) * cp * T)) * dHtdx(x,dx,eta_total,combustion_end))
    term3  = -((gamma * M**2 * (1 + (gamma-1) * M**2))/(2 * (1 - M**2)))
    term4 = (((4 * Cf * (P/Dh(x)))) - (2 * ((Vinj * P)/(mdot * V)) * (dmdotDX)))
    term5 = -(((2 * gamma * M**2 * (1 + ((gamma-1)/2) *M**2)*P)/((1-M**2)*mdot)) * (dmdotDX))
    return term1 + term2 + (term3 * term4) + term5

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





#Mixing Equations
def E1_CV(ui,Ti,uA,uB,TA,TB):
    return ui - (mdotAir/mdot_i) * uA - (mdotH2/mdot_i) * uB - (mdotAir * R_air * TA)/(mdot_i * uA) - (mdotH2 * R_H2 * TB)/(mdot_i * uB) + (R_mix * Ti)/ui

def E2_CV(ui,Ti,uA,uB,TA,TB):

    hi = gas_properties(Ti, 101325, Y_mix)["h"]
    hA = gas_properties(TA, 101325, Y_air)["h"]
    hB = gas_properties(TB, 101325, Y_H2)["h"]
    return (hi + ui**2/2) - (mdotAir/mdot_i) * (hA + uA**2/2) - (mdotH2/mdot_i) * (hB + uB**2/2)

def E3_InjA_CV(PstagA_2, uA_2, TA_2): #third cv equation check power point for indepth breakdown
    preinjA_gasProperties = gas_properties(TA_2, PstagA_2, get_Y("O2:0.21, N2:0.79"))
    part1 = (PstagA_2/(R_air * TstagA))
    part2_partial = ((preinjA_gasProperties["gamma"] - 1)/2) * ((uA_2)**2)/(soS(TA_2, R_air, preinjA_gasProperties["gamma"])**2)
    part2 = (1 + (part2_partial))
    part3 = (1 - (preinjA_gasProperties["gamma"]/(preinjA_gasProperties["gamma"] - 1)))
    rhoA = part1 * part2 **part3
    return rhoA * uA_2 * A_airInjs - mdotAir

def E4_InjB_CV(PstagB_2, uB_2, TB_2): #third cv equation check power point for indepth breakdown
    preinjB_gasProperties = gas_properties(TB_2, PstagB_2, get_Y("H2:1.0"))
    part1 = (PstagB_2/(R_H2 * TstagB))
    part2 = (1 + ((preinjB_gasProperties["gamma"] - 1)/2) * ((uB_2/soS(TB_2, R_H2, preinjB_gasProperties["gamma"]))**2))
    part3 = (1 - 1*(preinjB_gasProperties["gamma"]/(preinjB_gasProperties["gamma"]-1)))
    rhoB = part1 * part2**part3
    return rhoB * uB_2 * A_H2Injs - mdotH2

def E5_InjA_CV(TA_2,uA_2):
    preinjA_gasProperties = gas_properties(TA_2, 101325, get_Y("O2:0.21, N2:0.79"))
    part1_partial = (((preinjA_gasProperties["gamma"] - 1)/2) * ((uA_2)**2)/(soS(TA_2, R_air, preinjA_gasProperties["gamma"])**2))
    part1 = (1 + part1_partial)
    return (TA_2 * part1) - TstagA

def E6_InjB_CV(TB_2,uB_2):
    preinjB_gasProperties = gas_properties(TB_2, 101325, get_Y("H2:1.0"))
    part1_partial = (((preinjB_gasProperties["gamma"] - 1)/2) * ((uB_2)**2)/(soS(TB_2, R_H2, preinjB_gasProperties["gamma"])**2))
    part1 = (1 + part1_partial)
    return (TB_2 * part1) - TstagB

#NewtonRaphson Solvers

#the two following functions are my function where i basically seperate the injectors into 2 states.
# state 1 is initial and state 2 is post "induced loss"
#it is a multi var newton raphson method with damping that finds the roots - u and T - of the my two constraining equations.
#E5 and E6 are my mass converscation equations and E3 and E4 are my energy conservation equations (just using stag temp)
#this allows me to induce or guess a stag pressure loss pre mixing the two streams and then easily solve the two streams mixing after a loss has been applied
#thermodynamics (energy, mass, etc) is conserved so that is the reason this works 

def CV_toPreburner(u2,T2,uA,uB,TA,TB): #this is newton raphson for the CV it goes from state 1 (once gasses have mixed) to state 2 (preburner inlet) 
    numIters = 0                         #cut down the system of equations to 2 equations and 2 unkowns so just solving till im under tolorence 
    tol = 1e-8
    uB = uB/4
    print(u2,T2,uA,uB,TA,TB)
    print("____________________")

    E1 = E1_CV(u2,T2,uA,uB,TA,TB)
    E2 = E2_CV(u2,T2,uA,uB,TA,TB)

    E_vec = np.array([E1, E2])

    while(np.linalg.norm(E_vec, 2) >= tol and numIters <= 100):
        #numerical jacobian 

        deltaU = u2/1e6  #the delta or perturbation will be updating as u2 and T2 update to make sure its not too big or too small.
        deltaT = T2/1e6

        dE1du = (E1_CV(u2 + deltaU, T2, uA, uB, TA, TB) - E1)/deltaU
        dE1dT = (E1_CV(u2, T2 + deltaT, uA, uB, TA, TB) - E1)/deltaT
        dE2du = (E2_CV(u2 + deltaU, T2, uA, uB, TA, TB) - E2)/deltaU
        dE2dT = (E2_CV(u2, T2 + deltaT, uA, uB, TA, TB) - E2)/deltaT

        J = np.array([[dE1du, dE1dT], [dE2du, dE2dT]])
        
        deltas = np.linalg.solve(J, -E_vec)
        
        u2 += deltas[0]
        T2 += deltas[1]

        E1 = E1_CV(u2,T2,uA,uB,TA,TB)   #updating E1 and E2 values after updating u2 and T2 to check for convergence and to move
        E2 = E2_CV(u2,T2,uA,uB,TA,TB)   #the method forward
        E_vec = np.array([E1, E2])   

        numIters += 1 #just counting num of iterations 

    if numIters > 100 or not np.isfinite(u2) or not np.isfinite(T2) or T2 <= 0:
        raise RuntimeError("CV to Preburner solve failed")

    return u2, T2

def newtonRaphson_T(T_Guess, T_old, xOld, uOld, uNew, dx,eta_total,P,combustion_end):
    numIters = 0
    tol = 1e-8
    E = residualT(T_Guess, T_old, xOld, uOld, uNew, dx,eta_total,P,combustion_end)

    while abs(E) >= tol and numIters <= 100:
        deltaT = max(abs(T_Guess)*1e-6, 1e-6)
        dEdT = (residualT(T_Guess + deltaT, T_old, xOld, uOld, uNew, dx,eta_total,P,combustion_end) - E)/deltaT

        if not np.isfinite(dEdT) or abs(dEdT) < 1e-14:
            raise RuntimeError("Bad temperature Newton derivative")

        lamda = 1.0
        accepted = False

        while lamda > 1e-3:
            T_new = T_Guess - lamda * E/dEdT

            if T_new <= 0 or not np.isfinite(T_new) or T_new > 3*T_old:
                lamda *= 0.5
                continue

            E_new = residualT(T_new, T_old, xOld, uOld, uNew, dx,eta_total,P,combustion_end)

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

def rk45Step(V,P,Cf_dnz, Cf_cnz, h, x, T_preburner,eta_total,combustion_end): #add stages for each mdot 3
    accepted = False 
    location = geometry_regions(x)

    if location == "Preburner":
        local_tol = 1e-2
        h_max = 1e-1
    elif location == "Conv Nozzle" or location == "Div Nozzle":
        local_tol = 1e-6
        h_max =5e-3
    elif location == "Throat":
        local_tol = 1e-8
        h_max = 1e-4

    h = min(h,h_max)
    while(accepted !=True):

        if h < 1e-14:
            raise RuntimeError(f"RK45 step size got too small at x = {x:.4f}")
        mdot_Current = mdotFuncX(x)
        mdot_Prev = mdotFuncX(x-h)
        x1 = x
        Cf1 = cf_location(x1,Cf_dnz, Cf_cnz)
        A1 = geom_Area(x1)
        V1 = V
        P1 =  P
        try:
            T1 = T_preburner
            a1 = soS(T1,R_mix,gas_properties(T1, P1, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue
        M1 = mNum(V1,a1)
        k1V = h * dVdX(V1,A1,M1,T1,P1,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x1,x1-h),Cf1,x1,h,eta_total,combustion_end)
        k1P = h * dPdX(V1,A1,M1,T1,P1,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x1,x1-h),Cf1,x1,h,eta_total,combustion_end)

        x2 = x1 + 1/5 * h
        Cf2 = cf_location(x2,Cf_dnz, Cf_cnz)
        A2 = geom_Area(x2)
        V2 = V + 1/5 * k1V 
        P2 = P + 1/5 * k1P
        try:
            T2 = newtonRaphson_T(T1, T1, x1, V1, V2, 1/5 * h,eta_total,P2,combustion_end)
            a2 = soS(T2,R_mix,gas_properties(T2, P2, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue
        M2 = mNum(V2,a2)
        k2V = h * dVdX(V2,A2,M2,T2,P2,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x2,x1),Cf2,x2,1/5 * h,eta_total,combustion_end)
        k2P = h * dPdX(V2,A2,M2,T2,P2,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x2,x1),Cf2,x2,1/5 * h,eta_total,combustion_end)

        x3 = x1 + 3/10 * h
        Cf3 = cf_location(x3,Cf_dnz, Cf_cnz)

        A3 = geom_Area(x3)
        V3 = V + 3/40 * k1V + 9/40 * k2V
        P3 = P + 3/40 * k1P + 9/40 * k2P
        try:
            T3 = newtonRaphson_T(T1, T1, x1, V1, V3, 3/10 * h,eta_total,P3,combustion_end) 
            a3 = soS(T3,R_mix,gas_properties(T3, P3, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue
        M3 = mNum(V3,a3)
        k3V = h * dVdX(V3,A3,M3,T3,P3,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x3,x1),Cf3,x3,3/10 * h,eta_total,combustion_end)
        k3P = h * dPdX(V3,A3,M3,T3,P3,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x3,x1),Cf3,x3,3/10 * h,eta_total,combustion_end)

        x4 = x1 + 4/5 * h
        Cf4 = cf_location(x4,Cf_dnz, Cf_cnz)

        A4 = geom_Area(x4)
        V4 = V + 44/45 * k1V - 56/15 * k2V + 32/9 * k3V
        P4 = P + 44/45 * k1P - 56/15 * k2P + 32/9 * k3P
        try:
            T4 = newtonRaphson_T(T1, T1, x1, V1, V4, 4/5 * h,eta_total,P4,combustion_end) 
            a4 = soS(T4,R_mix,gas_properties(T4, P4, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue
        M4 = mNum(V4,a4)
        k4V = h * dVdX(V4,A4,M4,T4,P4,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x4,x1),Cf4,x4,4/5 * h,eta_total,combustion_end)
        k4P = h * dPdX(V4,A4,M4,T4,P4,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x4,x1),Cf4,x4,4/5 * h,eta_total,combustion_end)

        x5 = x1 + 8/9 * h
        Cf5 = cf_location(x5,Cf_dnz, Cf_cnz)
        A5 = geom_Area(x5)
        V5 = V + 19372/6561 * k1V - 25360/2187 * k2V + 64448/6561 * k3V - 212/729 * k4V
        P5 = P + 19372/6561 * k1P - 25360/2187 * k2P + 64448/6561 * k3P - 212/729 * k4P
        try:
            T5 = newtonRaphson_T(T1, T1, x1, V1, V5, 8/9 * h,eta_total,P5,combustion_end)
            a5 = soS(T5,R_mix,gas_properties(T5, P5, Y_mix)["gamma"])

        except:
            h *= 0.5
            continue

        M5 = mNum(V5,a5)
        k5V = h * dVdX(V5,A5,M5,T5,P5,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x5,x1),Cf5,x5,8/9 * h,eta_total,combustion_end)
        k5P = h * dPdX(V5,A5,M5,T5,P5,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x5,x1),Cf5,x5,8/9 * h,eta_total,combustion_end)

        x6 = x1 + h
        Cf6 = cf_location(x6,Cf_dnz, Cf_cnz)
        A6 = geom_Area(x6)
        V6 = V + 9017/3168 * k1V - 355/33 * k2V + 46732/5247 * k3V + 49/176 * k4V - 5103/18656 * k5V
        P6 = P + 9017/3168 * k1P - 355/33 * k2P + 46732/5247 * k3P + 49/176 * k4P - 5103/18656 * k5P
        try:
            T6 = newtonRaphson_T(T1, T1, x1, V1, V6, 1 * h,eta_total,P6,combustion_end)
            a6 = soS(T6,R_mix,gas_properties(T6, P6, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue

        M6 = mNum(V6,a6)
        k6V = h * dVdX(V6,A6,M6,T6,P6,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x6,x1),Cf6,x6,h,eta_total,combustion_end)
        k6P = h * dPdX(V6,A6,M6,T6,P6,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x6,x1),Cf6,x6,h,eta_total,combustion_end)

        #5th order solution 
        v_5Order = V + 35/384 * k1V + 500/1113 * k3V + 125/192 * k4V - 2187/6784 * k5V + 11/84 * k6V
        p_5Order = P + 35/384 * k1P + 500/1113 * k3P + 125/192 * k4P - 2187/6784 * k5P + 11/84 * k6P

        x7 = x1 + h
        Cf7 = cf_location(x7,Cf_dnz, Cf_cnz)
        A7 = geom_Area(x7)
        V7 = v_5Order
        P7 = p_5Order
        try:
            T7 = newtonRaphson_T(T1, T1, x1, V1, V7, 1 * h,eta_total,P7,combustion_end)
            a7 = soS(T7,R_mix,gas_properties(T7, P7, Y_mix)["gamma"])
        except:
            h *= 0.5
            continue       
        M7 = mNum(V7,a7)
        k7V = h * dVdX(V7,A7,M7,T7,P7,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x7,x1),Cf7,x7,h,eta_total,combustion_end)
        k7P = h * dPdX(V7,A7,M7,T7,P7,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x7,x1),Cf7,x7,h,eta_total,combustion_end)

        #4th order solution
        v_4Order = V + 5179/57600 * k1V + 7571/16695 * k3V + 393/640 * k4V - 92097/339200 * k5V + 187/2100 * k6V + 1/40 * k7V
        p_4Order = P + 5179/57600 * k1P + 7571/16695 * k3P + 393/640 * k4P - 92097/339200 * k5P + 187/2100 * k6P + 1/40 * k7P

        #error estimate 
        errorV = abs(v_5Order - v_4Order)
        errorP = abs(p_5Order - p_4Order)
        err = max(errorV, errorP)

        if errorV > V * local_tol or errorP > P * local_tol: #comparing error if either error are > than tol it means that step is too big so i am making it smaller 
            accepted = False 
            sV = 2 if errorV == 0 else 0.5*(local_tol/errorV)**(1/5)
            sP = 2 if errorP == 0 else 0.5*(local_tol/errorP)**(1/5)
            s = min(sV, sP)
            h = min(s * h, h_max)
            continue #this just restarts the loop with the updated h value
            
        else: #if both are smaller than tol then I am accepting the time step and then making it bigger 
            accepted = True

            Vnext, Pnext = v_5Order, p_5Order
            xNext = x1 + h
            Tnext = newtonRaphson_T(T1, T1, x1, V1, Vnext, 1 * h,eta_total,Pnext,combustion_end)

            errorRatio = max(
                errorV/(abs(V) * local_tol), errorP/(abs(P) * local_tol))
            
            s = 1.2 if errorRatio == 0 else min(2, 0.9 * errorRatio**(-1/5))

            h_next = min(s * h, h_max)
            break

    
    return xNext, Vnext, Pnext, Tnext,h_next, location


#Full Solver
def solver(Preburner_TStag,Cf_dnz, Cf_cnz,eta_total,combustion_end,scale, acceptedScale, postThroatSolve):
    global mdot,Vinj #making them global so that i can use them in rk45 and ode functions
    Vinj = 0

    Preburner_T = Preburner_TStag #k

    Preburner_predictedPStag = pstag_predicted(mdot_i, throat_Area, Preburner_TStag, gas_properties(Preburner_TStag, 101325, Y_mix)["gamma"])
    og_Preburner_P = newtonRaphson_P(Preburner_predictedPStag,Preburner_predictedPStag, Preburner_T, gas_properties(Preburner_T, 101325, Y_mix)["gamma"])

    if acceptedScale == False:
        if scale ==1:
            Preburner_P = og_Preburner_P
        else:
            Preburner_P = og_Preburner_P * scale

        Preburner_U = mdot_i/(Preburner_P * preburner_area / (R_mix * Preburner_T))
        Preburner_gasProperties = gas_properties(Preburner_T, Preburner_P, Y_mix)

        M_Preburner_Inlet = Preburner_U/soS(Preburner_T,R_mix,Preburner_gasProperties["gamma"])
        rho_preburner = mdot_i/(preburner_area * Preburner_U)
        Preburner_Pstag = Preburner_predictedPStag

    elif acceptedScale == True:
        Preburner_P = og_Preburner_P * scale
        Preburner_gasProperties = gas_properties(Preburner_T, Preburner_P, Y_mix)
        Preburner_U = mdot_i/(Preburner_P * preburner_area / (R_mix * Preburner_T))
        M_Preburner_Inlet = Preburner_U/soS(Preburner_T,R_mix,Preburner_gasProperties["gamma"])
        rho_preburner = mdot_i/(preburner_area * Preburner_U)
        Preburner_Pstag = pressureStagFunc(Preburner_P,M_Preburner_Inlet,Preburner_gasProperties["gamma"])


    temp = [Preburner_T]                # creating fresh arrays in function 
    velocities = [Preburner_U]
    pressure = [Preburner_P]
    pStag = [Preburner_Pstag]
    tStag = [Preburner_TStag]
    machNum = [M_Preburner_Inlet]
    density = [rho_preburner]
    areaList = [geom_Area(0)]
    dAdxList = [0.0]
    areaRatio = [1.0]
    xList = [0.0] #this list starts at the preburner 
    stepList = [1e-1]
    mdotList = [mdot_i]
    
    pt_location = []
    pt_pressures = []
    PT_locations = [0.1,0.2,0.3,0.4,0.495,0.5,0.505,0.51,0.6,0.64]


    sInitial = gas_properties(Preburner_T, Preburner_P,Y_mix)["s"]
    entropy = [sInitial]

    mdotReconstructed = [mdot_i] #recontruction array to check if calcs are correct 
    throatP = 0
    pb_count = 0
    throat_count = 0
    conv_count = 0
    div_count = 0
    while (xList[-1] < nozzle_exit ): #actual for loop for solving everything. from start of preburner to throat 

        xPrev = xList[-1]     
        hPrev = stepList[-1] #from step 0 to step 1 and then step 1 to step 2 etc 
        
        Vbefore = velocities[-1]
        Pbefore = pressure[-1]
        Tbefore = temp [-1]
        xNext, VCurrent, PCurrent, TCurrent, hNext, location = rk45Step(Vbefore,Pbefore,Cf_dnz, Cf_cnz,hPrev, xPrev,Tbefore,eta_total,combustion_end)


        if location == "Preburner":
            pb_count +=1
        elif location == "Throat":
            throat_count +=1 
        elif location == "Conv Nozzle":
            conv_count+=1
        elif location == "Div Nozzle":
            div_count +=1 

        currentMix_properties = gas_properties(TCurrent, PCurrent,Y_mix)
        currentMix_gamma = currentMix_properties["gamma"]

        xList.append(xNext)
        xCurrent = xList[-1]


        areaList.append(geom_Area(xCurrent))
        dAdxList.append(dAdx(xCurrent))

        mdotlocal = mdotFuncX(xCurrent)
        stepList.append(hNext)

        velocities.append(VCurrent)
        pressure.append(PCurrent)

        rhoCurrent = mdotlocal/(geom_Area(xCurrent) * VCurrent) 
        density.append(rhoCurrent)

        temp.append(TCurrent)

        mdotReconstructed.append(rhoCurrent * VCurrent * geom_Area(xCurrent))
        mdotList.append(mdotFuncX(xCurrent))

        aCurrent = soS(TCurrent,R_mix,currentMix_gamma)
        MCurrent = mNum(VCurrent,aCurrent)
        machNum.append(MCurrent)

        Pstag_current = pressureStagFunc(PCurrent, MCurrent, currentMix_gamma)
        pStag.append(Pstag_current)

        Tstag_current = temperatureStagFunc(TCurrent, MCurrent, currentMix_gamma)
        tStag.append(Tstag_current)

        sCurrent = currentMix_properties["s"]
        entropy.append(sCurrent)
        if postThroatSolve == True:
            pt_P, pt_x= pressureTap(xList[-2],pressure[-2],xCurrent, PCurrent,PT_locations)

            if pt_P is not None:
                pt_location.append(pt_x)
                pt_pressures.append(pt_P)

        if MCurrent >= 0.99 and postThroatSolve == False:
            break

        elif geometry_regions(xCurrent) == "Throat" and acceptedScale == True and postThroatSolve == True:
            throatP = pressure[-1]
            throatT = temp[-1]
            throatPstag = pStag[-1]
            throatTstag = tStag[-1]
            MachN = 1.001
            throatMix_properties = gas_properties(throatT, throatP,Y_mix)
            throatMix_gamma = throatMix_properties["gamma"]
            entropy_throat = throatMix_properties["s"]

            P_new, T_New = stagtostatic(throatPstag,throatTstag,MachN,throatMix_gamma)
            V_new = MachN * soS(T_New,R_mix,throatMix_gamma)
            xEndofThroat = throat_loc + 0.005
            rho_New = mdotlocal/(geom_Area(xEndofThroat) * V_new)

            mdotReconstructed.append(rho_New * V_new * geom_Area(xEndofThroat))
            mdotList.append(mdotFuncX(xEndofThroat))
            stepList.append(0.001)
            xList.append(xEndofThroat)
            pressure.append(P_new)
            velocities.append(V_new)
            temp.append(T_New)
            density.append(rho_New)
            machNum.append(MachN)
            pStag.append(throatPstag)
            tStag.append(throatTstag)
            entropy.append(entropy_throat)
            areaList.append(geom_Area(xEndofThroat))
            dAdxList.append(dAdx(xEndofThroat))
            pt_P, pt_x = pressureTap(xList[-2],pressure[-2],xEndofThroat, P_new,PT_locations)
            if pt_P is not None:

                pt_location.append(pt_x)
                pt_pressures.append(pt_P)
        else:
            continue

    #converting to np arrays
    V_List = np.array(velocities)
    P_List = np.array(pressure)
    T_List = np.array(temp)
    rho_List = np.array(density)
    M_List = np.array(machNum)
    AreaRatio_List = np.array(areaRatio)
    pStag_List = np.array(pStag)
    tStag_List = np.array(tStag)
    x_used_List = np.array(xList[:len(V_List)])
    Area_List = np.array(areaList)
    dAdx_List = np.array(dAdxList)
    pt_PressureList = np.array(pt_pressures)
    pt_locationList = np.array(pt_location)

    mdotReconsturcted_List = np.array(mdotReconstructed)
    mdot_List = mdotList
    entropy_List = np.array(entropy)

    return {
        "velocity": V_List,
        "pressure": P_List,
        "temperature": T_List,
        "density": rho_List,
        "Mach": M_List,
        "pressure_stag": pStag_List,
        "temperature_stag": tStag_List,
        "x": x_used_List,
        "Area": Area_List,
        "dAdx": dAdx_List,
        "mdot": mdot_List,
        "mdot_reconstructed": mdotReconsturcted_List,
        "entropy": entropy_List,
        "xChoked": xCurrent,
        "Area Ratio": AreaRatio_List, 
        "Choked Area Ratio": Area_List[0]/Area_List[-1], 
        "Initial Pstag" : pStag_List[0],
        "Initial Tstag" : tStag_List[0],
        "Preburner Count": pb_count,
        "Conv Count": conv_count,
        "throat Count": throat_count,
        "Throat Pressure": throatP,
        "PT_P": pt_PressureList,
        "PT_X": pt_locationList

    }

#Sweeping

def chokedLocationResiduals(scale, Cf_dnz, Cf_cnz,eta_total,combustion_end):
    results = solver(TstagA,Cf_dnz, Cf_cnz,eta_total,combustion_end,scale,False,False)
    x_Choke = results["x"][-1]
    residual = throat_loc - x_Choke  #we want this to be zero
    return residual
    
def eval_scale(scale,Cf_dnz, Cf_cnz,eta_total,combustion_end):
    #scale, Cf = args
    try:
        res = chokedLocationResiduals(scale, Cf_dnz, Cf_cnz,eta_total,combustion_end)
        return scale, res
    except Exception as eS:
        print("Scale Failed ", scale, eS)
        traceback.print_exc()
        return scale, np.nan
   
def scaling_InletPressure_NOTPar(Cf_dnz, Cf_cnz,eta_total,combustion_end):

    max_scale = 1.0
    max_res = chokedLocationResiduals(max_scale, Cf_dnz, Cf_cnz,eta_total,combustion_end)

    if max_res > 0:
        direction = 1
    elif max_res < 0:
        direction = -1
    else:
        return max_scale, max_scale

    prev_scale = max_scale
    prev_res = max_res

    for i in range(1, 21):
        try:
            cur_scale = max_scale + direction * (i/10)
            cur_scale, cur_res = eval_scale(cur_scale,Cf_dnz, Cf_cnz,eta_total,combustion_end)
        except Exception as eS:
            print(f"Failed because of: {eS}")
            traceback.print_exc()

            continue

        if not np.isfinite(cur_res):
            continue
        if cur_res * prev_res < 0:
            return cur_scale, prev_scale, cur_res, prev_res
        
        prev_scale = cur_scale
        prev_res = cur_res


def scale_HybridNewBisec(scale_low, scale_high,res_low, res_high,Cf_dnz, Cf_cnz,eta_total,combustion_end):

    tol = 1e-6
    maxIters = 100
    min_fraction = 0.10   # reject secant points too close to bracket edges

    best_scale = scale_low if abs(res_low) < abs(res_high) else scale_high
    best_res = res_low if abs(res_low) < abs(res_high) else res_high

    scale_candidate = best_scale
    res_candidate = best_res

    for i in range(maxIters):

        width = scale_high - scale_low

        # Secant / false-position proposal
        secant_denom = res_high - res_low

        if abs(secant_denom) > 1e-14:
            scale_candidate = (
                scale_high
                - res_high * (scale_high - scale_low) / secant_denom
            )

            # Guard: reject bad or endpoint-hugging secant proposals
            if (
                not np.isfinite(scale_candidate)
                or not (scale_low < scale_candidate < scale_high)
                or scale_candidate < scale_low + min_fraction * width
                or scale_candidate > scale_high - min_fraction * width
            ):
                scale_candidate = 0.5 * (scale_low + scale_high)

        else:
            scale_candidate = 0.5 * (scale_low + scale_high)

        res_candidate = chokedLocationResiduals(scale_candidate, Cf_dnz, Cf_cnz,eta_total,combustion_end)

        # Track best residual seen
        if abs(res_candidate) < abs(best_res):
            best_scale = scale_candidate
            best_res = res_candidate

        # Residual convergence
        if abs(res_candidate) < tol:
            return scale_candidate, res_candidate

        # Update bracket by sign
        if res_low * res_candidate < 0:
            scale_high = scale_candidate
            res_high = res_candidate
        else:
            scale_low = scale_candidate
            res_low = res_candidate

        # Bracket-size convergence
        if abs(scale_high - scale_low) < tol:
            return best_scale, best_res

    return best_scale, best_res

#MCMC set up
#using black box approach 

# wrapper for Op var. basically tells log_likelihood func that i am just putting in a double prec scalar and will
# output a double precision scalar 
# this is just so that i can easily pas my prior into my function 

#MCMC functions

@as_op(itypes=[pt.dscalar,pt.dscalar,pt.dscalar,pt.dscalar,pt.dvector],otypes=[pt.dscalar]) 
def log_likelihood(Cf_dnz, Cf_cnz,eta_total,combustion_end,true_PTPressure):    
    Cf_dnz = float(Cf_dnz)
    Cf_cnz = float(Cf_cnz)
    eta_total = float(eta_total)
    combustion_end = float(combustion_end)

    try:
        high_scale,low_scale,high_res, low_res = scaling_InletPressure_NOTPar(Cf_dnz, Cf_cnz,eta_total,combustion_end) #finding bracket 
        final_scale,final_res = scale_HybridNewBisec(low_scale,high_scale, low_res,high_res,Cf_dnz, Cf_cnz,eta_total,combustion_end)   # finding exact scale 
        resultsAtCorrectScale = solver(TstagA,Cf_dnz, Cf_cnz,eta_total,combustion_end,final_scale,True,True) #getting exact values at correct scale 
        Predicted_PTPressure = resultsAtCorrectScale["PT_P"]

        predicted_error = Predicted_PTPressure - true_PTPressure
        percent_uncertainty = 0.01 
        sigma_i = np.sqrt((percent_uncertainty * true_PTPressure)**2) 

        log_prob = np.sum(stats.norm.logpdf(predicted_error,loc = 0.0,scale = sigma_i))

        return np.array(log_prob, dtype=np.float64)
    
    except Exception as e:
            print(f"Failed because of: {e}")
            traceback.print_exc()
            return np.array(-1.0e10, dtype=np.float64)
    

def generatingTrueValues(True_Cf_pb,True_Cf_nz,True_eta_total,True_x_react):

    high_scale,low_scale,high_res, low_res = scaling_InletPressure_NOTPar(True_Cf_pb,True_Cf_nz,True_eta_total,True_x_react) #finding bracket 
    final_scale,final_res = scale_HybridNewBisec(low_scale,high_scale, low_res,high_res,True_Cf_pb,True_Cf_nz,True_eta_total,True_x_react)   # finding exact scale 
    resultsAtCorrectScale = solver(TstagA,True_Cf_pb,True_Cf_nz,True_eta_total,True_x_react,final_scale,True,True) #getting exact values at correct scale 

    true_PTPressure = resultsAtCorrectScale["PT_P"]
    return true_PTPressure

'''
def likelihoodPlotting(frozenVar1, frozenVar2, frozenVar3, MovingVar):

    logplist = []
    count = 0 
    Cf_nz = frozenVar1
    eta_total = frozenVar2
    combustion_end = frozenVar3
    
    movingVar_grid = np.linspace(MovingVar - MovingVar * 0.2, MovingVar + MovingVar * 0.2, 50)

    True_Cf_pb = 0.002
    True_Cf_nz = 0.005
    True_eta_Total = 0.8
    True_x_react = 0.2

    true_PTPressure = generatingTrueValues(True_Cf_pb,True_Cf_nz, True_eta_Total,True_x_react)

    count = 0

    for movingVariable in movingVar_grid:

        try:
            count+=1

            high_scale,low_scale,high_res, low_res = scaling_InletPressure_NOTPar(movingVariable,Cf_nz,eta_total,combustion_end) #finding bracket 
            final_scale,final_res = scale_HybridNewBisec(low_scale,high_scale, low_res,high_res,movingVariable,Cf_nz,eta_total,combustion_end)   # finding exact scale 
            resultsAtCorrectScale = solver(TstagA,movingVariable,Cf_nz,eta_total,combustion_end,final_scale,True,True) #getting exact values at correct scale 
            Predicted_PTPressure = resultsAtCorrectScale["PT_P"]

            predicted_error = Predicted_PTPressure - true_PTPressure
            percent_uncertainty = 0.01 
            sigma_i = np.sqrt((percent_uncertainty * true_PTPressure)**2) 

            log_prob = np.sum(stats.norm.logpdf(predicted_error,loc = 0.0,scale = sigma_i))
            print(count)
            logplist.append(log_prob)
        
        except Exception as e:
                print(f"Failed because of: {e}")
        


    logp_array = np.array(logplist)
    movingVar_list = movingVar_grid[:len(logp_array)]
    
    plt.figure()
    plt.plot(movingVar_list,logp_array)
    plt.xlabel("Values")
    plt.ylabel("Log Likelihood")
    plt.grid()
    plt.show()

'''
#MCMC Model

def run_MCMC_case(caseConfig):

    param_names = caseConfig["Parameters"]
    
    set_True_Cf_dNz = caseConfig["True_Cf_dNz"]
    set_Cf_dNz_Prior_mu = caseConfig["Cf_dNz_Prior_mu"]
    set_Cf_dNz_Prior_sigma = caseConfig["Cf_dNz_Prior_sigma"]
    set_Cf_dNz_scale = caseConfig["Cf_dNz_Scale"]
    set_Cf_dNz_scaling = caseConfig["Cf_dNz_Scaling"]

    set_True_Cf_cNz = caseConfig["True_Cf_cNz"]
    set_Cf_cNz_Prior_mu = caseConfig["Cf_cNz_Prior_mu"]
    set_Cf_cNz_Prior_sigma = caseConfig["Cf_cNz_Prior_sigma"]
    set_Cf_cNz_scale = caseConfig["Cf_cNz_Scale"]
    set_Cf_cNz_scaling = caseConfig["Cf_cNz_Scaling"]

    set_True_eta_Total = caseConfig["True_eta_Total"]
    set_eta_Total_Prior_mu = caseConfig["eta_Total_Prior_mu"]
    set_eta_Total_Prior_sigma = caseConfig["eta_Total_Prior_sigma"]
    set_eta_Total_scale = caseConfig["eta_Total_Scale"]
    set_eta_Total_scaling = caseConfig["eta_Total_Scaling"]

    set_True_combustion_end = caseConfig["True_combustion_end"]
    set_combustion_end_Prior_mu = caseConfig["combustion_end_Prior_mu"]
    set_combustion_end_Prior_sigma = caseConfig["combustion_end_Prior_sigma"]
    set_combustion_end_scale = caseConfig["combustion_end_Scale"]
    set_combustion_end_scaling = caseConfig["combustion_end_Scaling"]

    set_draws = caseConfig["Draws"]
    set_tune = caseConfig["Tune"]
    set_chains = caseConfig["Chains"]
    set_cores = caseConfig["Cores"]

    with pm.Model() as model:
      
            timestamp = datetime.now().strftime("%H_%d_%m_%Y")

            run_label = (
                f"{param_names}_"
                f"{caseConfig['Case_Name']}_"
                f"{timestamp}"
            )
            nameofCase = caseConfig["Case_Name"]

            results_root = Path("MCMC Results")
            results_root.mkdir(exist_ok = True)

            case_folder = results_root / run_label
            case_folder.mkdir(exist_ok = True)

            diagnostics_folder = case_folder / "Diagnostics"
            diagnostics_folder.mkdir(exist_ok = True)

            with open(case_folder/f"{nameofCase}_config.txt", "w") as f:
                for key,value in caseConfig.items():
                    f.write(f"{key}: {value}\n")


            true_PTPressure = generatingTrueValues(set_True_Cf_dNz,set_True_Cf_cNz,set_True_eta_Total,set_True_combustion_end)
            prior_Cf_dNz = pm.TruncatedNormal("Cf_dNz", mu=set_Cf_dNz_Prior_mu,  sigma=set_Cf_dNz_Prior_sigma,lower = 0,initval=set_Cf_dNz_Prior_mu,default_transform=None)
            prior_Cf_cNz = pm.TruncatedNormal("Cf_cNz", mu=set_Cf_cNz_Prior_mu,  sigma=set_Cf_cNz_Prior_sigma,lower = 0,initval=set_Cf_cNz_Prior_mu,default_transform=None)

            prior_eta_Total = pm.TruncatedNormal("eta_Total", mu=set_eta_Total_Prior_mu, sigma = set_eta_Total_Prior_sigma, lower = 0, upper = 1, initval=set_eta_Total_Prior_mu, default_transform=None)
            prior_combustion_end = pm.TruncatedNormal("combustion_end", mu=set_combustion_end_Prior_mu, sigma = set_combustion_end_Prior_sigma, lower = 0, initval=set_combustion_end_Prior_mu, default_transform=None)

            log_like = log_likelihood(prior_Cf_dNz,prior_Cf_cNz,prior_eta_Total,prior_combustion_end,
                                      pt.as_tensor_variable(true_PTPressure, dtype="float64"))

            pm.Potential("Error Likelihood",log_like)
            
            step = pm.DEMetropolisZ(
                vars = [prior_Cf_dNz,prior_Cf_cNz,prior_eta_Total,prior_combustion_end],
                S= np.array([set_Cf_dNz_scale,set_Cf_cNz_scale,set_eta_Total_scale,set_combustion_end_scale]), 
                scaling = np.array([set_Cf_dNz_scaling,set_Cf_cNz_scaling,set_eta_Total_scaling,set_combustion_end_scaling]),  #Initial scale factor for how aggressive the sampler noise moves around 
                tune="scaling",
                tune_interval=100,
                tune_drop_fraction=0.9
            )

            trace = pm.sample(
                draws=set_draws,
                tune=set_tune,
                step = step,
                chains=set_chains,
                cores = set_cores, 
                random_seed=42,
                progressbar=True,
                return_inferencedata=True,
                compute_convergence_checks=True,
            )

            print("Sample stats")
            print(list(trace.sample_stats.data_vars))

            acceptance_rate = None
            accepted = None

            if "accepted" in trace.sample_stats.data_vars:
                accepted = trace.sample_stats["accepted"].values
                acceptance_rate = np.mean(accepted)

                print("Overall acceptance rate:", acceptance_rate)

    summary = az.summary(trace)

    cf_dNz_samples = trace.posterior["Cf_dNz"].values.flatten()
    cf_cNz_samples = trace.posterior["Cf_cNz"].values.flatten()
    eta_Total_samples = trace.posterior["eta_Total"].values.flatten()
    combustion_end_samples = trace.posterior["combustion_end"].values.flatten()
    
    with open(case_folder/ f"{nameofCase}_MCMC_Report.txt", "w") as f:
        f.write(f"Parameters Included in Model = {param_names}\n")

        f.write(f"{caseConfig['Case_Name']} Values \n")

        f.write(f"\nTrue Diverging Nozzle Cf = {set_True_Cf_dNz}\n")
        f.write(f"Diverging Nozzle Cf Prior Mean = {set_Cf_dNz_Prior_mu}\n")
        f.write(f"Diverging Nozzle Cf Prior Sigma = {set_Cf_dNz_Prior_sigma}\n")
        f.write(f"Diverging Nozzle Cf Scale = {set_Cf_dNz_scale}\n")
        f.write(f"Diverging Nozzle Cf Scaling = {set_Cf_dNz_scaling}\n")

        f.write(f"\nTrue Converging Nozzle Cf = {set_True_Cf_cNz}\n")
        f.write(f"Converging Nozzle Cf Prior Mean = {set_Cf_cNz_Prior_mu}\n")
        f.write(f"Converging Nozzle Cf Prior Sigma = {set_Cf_cNz_Prior_sigma}\n")
        f.write(f"Converging Nozzle Cf Scale = {set_Cf_cNz_scale}\n")
        f.write(f"Converging Nozzle Cf Scaling = {set_Cf_cNz_scaling}\n")

        f.write(f"\nTrue Eta Total = {set_True_eta_Total}\n")
        f.write(f"Eta Total Prior Mean = {set_eta_Total_Prior_mu}\n")
        f.write(f"Eta Total Prior Sigma = {set_eta_Total_Prior_sigma}\n")
        f.write(f"Eta Total Scale = {set_eta_Total_scale}\n")
        f.write(f"Eta Total Scaling = {set_eta_Total_scaling}\n")

        f.write(f"\nTrue Combustion End = {set_True_combustion_end}\n")
        f.write(f"Combustion End Prior Mean = {set_combustion_end_Prior_mu}\n")
        f.write(f"Combustion End Prior Sigma = {set_combustion_end_Prior_sigma}\n")
        f.write(f"Combustion End Scale = {set_combustion_end_scale}\n")
        f.write(f"Combustion End Scaling = {set_combustion_end_scaling}\n")
        
        f.write("\nARVIZ Summary\n")
        f.write("---------------------\n")
        f.write(summary.to_string())

        f.write("\nOther Results and metrics \n")
        f.write(f"Acceptance Rate: {acceptance_rate}\n")

        f.write("\nSettings:\n")
        f.write(f"Draws = {set_draws}\n")
        f.write(f"Tune = {set_tune}\n")
        f.write(f"Chains = {set_chains}\n")
        f.write(f"Cores = {set_cores}\n")

    param_labels = {
            "Cf_dNz": "Diverging Nozzle Friction Coefficient (Cf)",
            "Cf_cNz": "Converging Nozzle Friction Coefficient (Cf)",
            "eta_Total": "Combustion Efficiency (\u03b7_Total)",
            "combustion_end": "Combustion End Location"
        }
    
    true_values = {
            "Cf_dNz": set_True_Cf_dNz,
            "Cf_cNz": set_True_Cf_cNz,
            "eta_Total": set_True_eta_Total,
            "combustion_end" : set_True_combustion_end
        }

    for param in summary.index:
        label = param_labels.get(param, param)
        true_val = true_values.get(param)

        az.plot_trace(trace, var_names=[param])
        plt.suptitle(f"Trace Plot \u2014 {label} | {nameofCase}", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_Trace.png", dpi=200)
        plt.close()

        az.plot_dist(trace, var_names=[param])
        if true_val is not None:
            plt.axvline(true_val, color="red", linestyle="--", linewidth=1.5, label=f"True = {true_val}")
            plt.legend(fontsize=9)

        plt.xlabel(label, fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.title(f"Posterior Distribution \u2014 {label} | {nameofCase}", fontsize=11)
        plt.tight_layout()
        plt.savefig(case_folder / f"{param}_{nameofCase}_Posterior.png", dpi=200)
        plt.close()

        az.plot_autocorr(trace, var_names=[param])
        plt.suptitle(f"Autocorrelation \u2014 {label} | {nameofCase}", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_Autocorr.png", dpi=200)
        plt.close()

        az.plot_ess(trace, var_names=[param])
        plt.suptitle(f"ESS Plot \u2014 {label} | {nameofCase}", fontsize=11)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_Ess.png", dpi=200)
        plt.close()

        az.plot_rank(trace, var_names=[param])
        plt.suptitle(f"rank Plot \u2014 {label} | {nameofCase}", fontsize=11)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_rank.png", dpi=200)
        plt.close()


    running_mean_cf_dNz = np.cumsum(cf_dNz_samples) / np.arange(1, len(cf_dNz_samples) + 1)
    running_mean_cf_cNz = np.cumsum(cf_cNz_samples) / np.arange(1, len(cf_cNz_samples) + 1)
    running_mean_eta_Total = np.cumsum(eta_Total_samples) / np.arange(1, len(eta_Total_samples) + 1)
    running_mean_combustion_end = np.cumsum(combustion_end_samples) / np.arange(1, len(combustion_end_samples) + 1)

    #initialize a side-by-side figure layout (1 row, 4 columns)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Cf_div Nozzle
    axs[0].plot(running_mean_cf_dNz, color="steelblue", label="Running mean")
    axs[0].axhline(set_True_Cf_dNz, color="red", linestyle="--", linewidth=1.5, label=f"True Diverging Nozzle Cf = {set_True_Cf_dNz}")
    axs[0].set_xlabel("Sample", fontsize=11)
    axs[0].set_ylabel("Cf", fontsize=11)
    axs[0].set_title(f"Running Mean \u2014 Friction Coefficient | {nameofCase}", fontsize=11)
    axs[0].legend(fontsize=10)
    axs[0].grid(True, alpha=0.4)

    # Cf_conv nozzle
    axs[1].plot(running_mean_cf_cNz, color="steelblue", label="Running mean")
    axs[1].axhline(set_True_Cf_cNz, color="red", linestyle="--", linewidth=1.5, label=f"True Converging Nozzle Cf = {set_True_Cf_cNz}")
    axs[1].set_xlabel("Sample", fontsize=11)
    axs[1].set_ylabel("Cf", fontsize=11)
    axs[1].set_title(f"Running Mean \u2014 Friction Coefficient | {nameofCase}", fontsize=11)
    axs[1].legend(fontsize=10)
    axs[1].grid(True, alpha=0.4)

    # Combustion Eff
    axs[2].plot(running_mean_eta_Total, color="darkorange", label="Running mean")
    axs[2].axhline(set_True_eta_Total, color="red", linestyle="--", linewidth=1.5, label=f"True \u03b7 = {set_True_eta_Total}")
    axs[2].set_xlabel("Sample", fontsize=11)
    axs[2].set_ylabel("\u03b7_Total", fontsize=11)
    axs[2].set_title(f"Running Mean \u2014 Combustion Efficiency | {nameofCase}", fontsize=11)
    axs[2].legend(fontsize=10)
    axs[2].grid(True, alpha=0.4)

    # combustion_end
    axs[3].plot(running_mean_combustion_end, color="darkorange", label="Running mean")
    axs[3].axhline(set_True_combustion_end, color="red", linestyle="--", linewidth=1.5, label=f"True x = {set_True_combustion_end}")
    axs[3].set_xlabel("Sample", fontsize=11)
    axs[3].set_ylabel("combustion_end", fontsize=11)
    axs[3].set_title(f"Running Mean of Combustion End Location | {nameofCase}", fontsize=11)
    axs[3].legend(fontsize=10)
    axs[3].grid(True, alpha=0.4)


    plt.tight_layout()
    plt.savefig(case_folder / f"Combined_Metrics_{nameofCase}_Running_mean.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    az.plot_pair(trace,var_names= ["Cf_dNz","Cf_cNz","eta_Total","combustion_end"])
    plt.suptitle("Joint Posterior", y = 1.02)
    plt.tight_layout()
    plt.savefig(case_folder / f"PairPlot_{nameofCase}.png", dpi=150, bbox_inches = "tight")
    plt.close()

#cases and running model 
if __name__ == "__main__":
   
    case_name = sys.argv[1]

    all_cases = {
        
        "4_Param_Base_Case": {
            "Case_Name": "4_Param_Base_Case",
            "Parameters": ["Cf_dNz","Cf_cNz", "eta_Total","combustion_end"],

            "True_Cf_dNz": 0.008,
            "Cf_dNz_Prior_mu" : 0.0019,
            "Cf_dNz_Prior_sigma" : (0.05 * 0.002),
            "Cf_dNz_Scale" : 0.001,
            "Cf_dNz_Scaling" : 0.001,

            "True_Cf_cNz": 0.005,
            "Cf_cNz_Prior_mu" : 0.0076,
            "Cf_cNz_Prior_sigma" : (0.008 * 0.05),
            "Cf_cNz_Scale" : 0.001,
            "Cf_cNz_Scaling" : 0.001,

            "True_eta_Total" : 0.8,
            "eta_Total_Prior_mu": 0.76,
            "eta_Total_Prior_sigma": (0.05 * 0.8),
            "eta_Total_Scale" : 0.1,
            "eta_Total_Scaling":0.001,

            "True_combustion_end": preburner_length,
            "combustion_end_Prior_mu" : preburner_length - 0.05 * preburner_length,
            "combustion_end_Prior_sigma" : (0.05 * preburner_length),
            "combustion_end_Scale" : 0.1,
            "combustion_end_Scaling" : 0.001,

            "Draws" : 500,
            "Tune" : 250,
            "Chains" : 10 ,
            "Cores" : 10
            },
    
    }
  

    case = all_cases[case_name]

    run_MCMC_case(case)

