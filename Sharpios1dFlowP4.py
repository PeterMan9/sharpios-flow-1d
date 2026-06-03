import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import cantera as ct
from scipy import stats
from scipy.optimize import least_squares
import time
import pymc as pm

f_darcy = 0.02
Cf = f_darcy/4

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

def dsmoothstep_dxi(xi):
    return 30*xi**4 - 60*xi**3 + 30*xi**2

def geometry_regions(x):
    if x <= preburner_length:
        return "Preburner"
    elif throat_loc - 0.005 <= x <= throat_loc + 0.005: 
        return "Throat"
    elif x <= throat_loc:
        return "Conv Nozzle"
    elif x <= nozzle_exit:
        return "Div Nozzle"
    else:
        return "outside"

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
    if geometry_regions(x) == "Preburner":
        return 0.0
    elif geometry_regions(x) == "Throat":
        xCurrent = x
        xPrev = x - (x*tol)
        dA = geom_Area(xCurrent) - geom_Area(xPrev)
        return dA/(xCurrent - xPrev)
    
    elif geometry_regions(x) == "Conv Nozzle":
        xCurrent = x
        xPrev = x - (x*tol)
        dA = geom_Area(xCurrent) - geom_Area(xPrev)
        return dA/(xCurrent - xPrev)
    
    elif geometry_regions(x) == "Div Nozzle":
        xCurrent = x
        xPrev = x - (x*tol)
        dA = geom_Area(xCurrent) - geom_Area(xPrev)
        return dA/(xCurrent - xPrev)
    else:
        return 0.0
    
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
eta_total = 1
theta = 1
x_react = 0

def x_norm(x):
    return (x - x_react)/(preburner_length - x_react)

def eta(x):
    eta = eta_total * (theta * x_norm(x)/(1 + (theta - 1) * x_norm(x)))
    return eta

def dPHI(x,dx):
    xcurrent = x
    xprev = x - dx
    dPHI = phi * (eta(xcurrent) - eta(xprev))
    return dPHI

def dHtdx(x,dx):
    return (dPHI(x,dx) * hpr_h2 * fst)/dx

def residualT(T_new,T_old,xOld,uOld,uNew,dx):
    T_gasProperties_old = gas_properties(T_old, 101325, Y_mix)
    T_gasProperties_new = gas_properties(T_new, 101325, Y_mix)
    ht_old = T_gasProperties_old["h"]
    ht_new = T_gasProperties_new["h"]
    term1 = (ht_new - ht_old)
    term2 = (uNew**2 - uOld**2)/2
    term3 = dHtdx(xOld,dx) * dx
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
print("R_air", R_air, "R_H2", R_H2, "R_mix", R_mix)

a1 = soS(T_air, R_air, gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"])
a2 = soS(T_air2, R_air, gas_properties(T_air2, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"])
a3 = soS(T_H2, R_H2, gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"])
print("a1", a1, "a2", a2, "a3", a3)

uA = M1 * a1
uA_2 = M2 * a2
uB = M3 * a3
print("uA", uA, "uA_2", uA_2, "uB", uB)

TstagA = T_air * (1 + ((gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"] - 1)/2) * M1**2)
TstagB = T_H2 * (1 + ((gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"] - 1)/2) * M3**2)

Pstag_Air = P_air * (1 + (gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"] - 1)/2 * M1**2)**(gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"]/(gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"]-1))
Pstag_H2 = P_H2 * (1 + (gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"] - 1)/2 * M3**2)**(gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"]/(gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"]-1))

rho1 = mdot1_Air/(A_airInjs * uA)
rho2 = mdot2_Air/(A_airInjs * uA_2)
rho3 = mdot3_H2/(A_H2Injs * uB)

M_h2 = (mdot3_H2/(rho3 * A_H2Injs))/a3
print("M_h2", M_h2)
print("rho1", rho1, "rho2", rho2, "rho3", rho3)

A_CV_END = preburner_area #area at the end of the CV is the same as the area at the start of the preburner inlet.
    
def delMdotdx(mdotn1, mdotn,x1,x): #dmdot/dx function
    return (mdotn1 - mdotn)/(x1-x)

def mdotFuncX (x):
    if x < x_injLocation: #pre injector mdot
        return mdot_i
    else:   #post injector mdot
        return mdot_i + injMdot 

#1st order ODE Functions
def dVdX (V,A,M,T,P,mdot,dmdotDX, Cf, x,dx): #first 4 parts of sharpios 1d flow eqn converted to dV/dx
    gas_Prop = gas_properties(T, P, Y_mix)
    cp = gas_Prop["cp"]
    gamma = gas_Prop["gamma"]

    term1 = ((-V)/(A * (1 - M**2)))* dAdx(x)
    term2 = ((V/((1-M**2) * cp * T)) * dHtdx(x,dx))
    term3 = ((gamma *M**2)/(2 * (1 - M**2)))
    term4 = ((((4 * Cf * V)/Dh(x))) - (2*(Vinj/mdot) * dmdotDX))
    term5 = (((V*(1 + gamma * M**2))/((1-M**2)*mdot)) * (dmdotDX))
    return term1 + term2 + (term3*term4) + term5

def dPdX (V,A,M,T,P,mdot,dmdotDX, Cf, x,dx): #first 4 parts of sharpios 1d flow eqn converted to dP/dx
    gas_Prop = gas_properties(T, P, Y_mix)
    cp = gas_Prop["cp"]
    gamma = gas_Prop["gamma"]

    term1 = ((gamma * M**2 * P)/(A * (1 - M**2))) * dAdx(x)
    term2 = -(((gamma * M**2 * P)/((1-M**2) * cp * T)) * dHtdx(x,dx))
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
    preinjA_gasProperties = gas_properties(TA_2, None, get_Y("O2:0.21, N2:0.79"))
    part1_partial = (((preinjA_gasProperties["gamma"] - 1)/2) * ((uA_2)**2)/(soS(TA_2, R_air, preinjA_gasProperties["gamma"])**2))
    part1 = (1 + part1_partial)
    return (TA_2 * part1) - TstagA

def E6_InjB_CV(TB_2,uB_2):
    preinjB_gasProperties = gas_properties(TB_2, None, get_Y("H2:1.0"))
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



def InjA_Loss_CV(Pstag_A2,uA_2, TA_2):
    numIters = 0
    tol = 1e-8

    E5 = E5_InjA_CV(TA_2,uA_2)
    E3 = E3_InjA_CV(Pstag_A2,uA_2, TA_2)

    E_vec = np.array([E5, E3])

    while(np.linalg.norm(E_vec, 2) >= tol and numIters <= 1000):

        deltaU = max(abs(uA_2)*1e-6, 1e-3)
        deltaT = max(abs(TA_2)*1e-6, 1e-3)

        #partial derivatives for numerical jacobian
        dE5du = (E5_InjA_CV(TA_2, uA_2+ deltaU) - E5)/deltaU
        dE5dT = (E5_InjA_CV(TA_2 + deltaT, uA_2) - E5)/deltaT
        dE3du = (E3_InjA_CV(Pstag_A2, uA_2 + deltaU, TA_2) - E3)/deltaU
        dE3dT = (E3_InjA_CV(Pstag_A2, uA_2, TA_2 + deltaT) - E3)/deltaT
 
        J = np.array([[dE5du, dE5dT], [dE3du, dE3dT]])
        deltas = np.linalg.solve(J, -E_vec)
        old_norm = np.linalg.norm(E_vec, 2)

        lamda = 1 # setting up damped newton raphson method so that I dont overshoot my initial couple steps 

        while lamda > 1e-3:

            uA_2Trial = uA_2 + lamda * deltas[0]
            TA_2Trial = TA_2 + lamda * deltas[1]

            if uA_2Trial <= 0 or TA_2Trial <= 0: #if statement to check to make sure i dont overstep and if i do, i dampin the step
                lamda *=0.5
                continue #this basically stops the code and forces it to restart at the top of the loop with an updated lamda 

            E5_trial = E5_InjA_CV(TA_2Trial, uA_2Trial) #updating E5 and E3 values after updating u2 and T2 to check for convergence 
            E3_trial = E3_InjA_CV(Pstag_A2, uA_2Trial, TA_2Trial) 
            trial_norm = np.linalg.norm([E5_trial,E3_trial],2) #taking the L2 norm of this vector to basically get the magnitude of the residual equations  

            #if i get a norm that is a number and it is smaller than my previous pre dampning norm i exit loop .
            #if not i just make my lamda smaller
            if np.isfinite(trial_norm) and trial_norm < old_norm: 
                break                                               
            
            lamda *= 0.5
        #updating values (u,T and residual eq and vector) from good newton step
        uA_2 = uA_2Trial 
        TA_2 = TA_2Trial
        E5 = E5_trial
        E3 = E3_trial
        E_vec = np.array([E5, E3])

        numIters += 1

        if numIters > 1000 or not np.isfinite(uA_2) or not np.isfinite(TA_2) or TA_2 <= 0:
            print("uA_2", uA_2, "TA_2", TA_2, "numIters", numIters)
            raise RuntimeError("InjA solve failed")

    return uA_2, TA_2

def InjB_Loss_CV(Pstag_B2,uB_2, TB_2):
    numIters = 0
    tol = 1e-6

    E4 = E4_InjB_CV(Pstag_B2,uB_2, TB_2)
    E6 = E6_InjB_CV(TB_2,uB_2)
    E_vec = np.array([E6, E4])

    while(np.linalg.norm(E_vec, 2) >= tol and numIters <= 500):
        deltaU = max(abs(uB_2)*1e-6, 1e-3)
        deltaT = max(abs(TB_2)*1e-6, 1e-3)

        #partial derivatives for numerical jacobian
        dE6du = (E6_InjB_CV(TB_2, uB_2 + deltaU) - E6)/deltaU
        dE6dT = (E6_InjB_CV(TB_2 + deltaT, uB_2) - E6)/deltaT
        dE4du = (E4_InjB_CV(Pstag_B2, uB_2 + deltaU, TB_2) - E4)/deltaU
        dE4dT = (E4_InjB_CV(Pstag_B2, uB_2, TB_2 + deltaT) - E4)/deltaT

        J = np.array([[dE6du, dE6dT], [dE4du, dE4dT]])
        deltas = np.linalg.solve(J, -E_vec)
        old_norm = np.linalg.norm(E_vec, 2)

        lamda = 1 # same as for inj A loss cv. just my dampning factor

        while lamda > 1e-3:

            uB_2Trial = uB_2 + lamda * deltas[0]
            TB_2Trial = TB_2 + lamda * deltas[1]

            if uB_2Trial <= 0 or TB_2Trial <= 0: #if statement to check to make sure i dont overstep and if i do, i dampin the step
                lamda *=0.5
                continue #this basically stops the code and forces it to restart at the top of the loop with an updated lamda 

            E6_trial = E6_InjB_CV(TB_2Trial, uB_2Trial) #updating E6 and E4 values after updating u2 and T2 to check for convergence 
            E4_trial = E4_InjB_CV(Pstag_B2, uB_2Trial, TB_2Trial) 
            trial_norm = np.linalg.norm([E6_trial,E4_trial],2) #taking the L2 norm of this vector to basically get the magnitude of the residual equations  

            #if i get a norm that is a number and it is smaller than my previous pre dampning norm i exit loop .
            #if not i just make my lamda smaller
            if np.isfinite(trial_norm) and trial_norm < old_norm: 
                break                                               
            
            lamda *= 0.5
        #updating values (u,T and residual eq and vector) from good newton step
        uB_2 = uB_2Trial 
        TB_2 = TB_2Trial
        E6 = E6_trial
        E4 = E4_trial
        E_vec = np.array([E6, E4])

        numIters += 1

        if numIters > 500 or not np.isfinite(uB_2) or not np.isfinite(TB_2) or TB_2 <= 0:
         print("uB_2", uB_2, "TB_2", TB_2, "numIters", numIters)
         raise RuntimeError("InjB solve failed")
        
    return uB_2, TB_2

def newtonRaphson_T(T_Guess, T_old, xOld, uOld, uNew, dx):
    numIters = 0
    tol = 1e-8
    E = residualT(T_Guess, T_old, xOld, uOld, uNew, dx)

    while abs(E) >= tol and numIters <= 100:
        deltaT = max(abs(T_Guess)*1e-6, 1e-6)
        dEdT = (residualT(T_Guess + deltaT, T_old, xOld, uOld, uNew, dx) - E)/deltaT

        if not np.isfinite(dEdT) or abs(dEdT) < 1e-14:
            raise RuntimeError("Bad temperature Newton derivative")

        lamda = 1.0
        accepted = False

        while lamda > 1e-3:
            T_new = T_Guess - lamda * E/dEdT

            if T_new <= 0 or not np.isfinite(T_new) or T_new > 3*T_old:
                lamda *= 0.5
                continue

            E_new = residualT(T_new, T_old, xOld, uOld, uNew, dx)

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

def rk45Step(V,P,Cf, h, x, T_preburner): #add stages for each mdot 3
    accepted = False 
    tol = 1e-6
    location = geometry_regions(x)

    if location == "Preburner":
        h = min(h, 1e-1)
    elif location == "Conv Nozzle" or location == "Div Nozzle":
        h = min(h, 5e-3)
    elif location == "Throat":
        h = min(h, 1e-3)
        
    while (accepted != True):
        mdot_Current = mdotFuncX(x)
        mdot_Prev = mdotFuncX(x-h)
        x1 = x
        A1 = geom_Area(x1)
        V1 = V
        P1 =  P
        T1 = T_preburner
        a1 = soS(T1,R_mix,gas_properties(T1, P1, Y_mix)["gamma"])
        M1 = mNum(V1,a1)
        k1V = h * dVdX(V1,A1,M1,T1,P1,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x1,x1-h),Cf,x1,h)
        k1P = h * dPdX(V1,A1,M1,T1,P1,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x1,x1-h),Cf,x1,h)

        x2 = x1 + 1/5 * h
        A2 = geom_Area(x2)
        V2 = V + 1/5 * k1V 
        P2 = P + 1/5 * k1P
        T2 = newtonRaphson_T(T1, T1, x1, V1, V2, 1/5 * h) 
        a2 = soS(T2,R_mix,gas_properties(T2, P2, Y_mix)["gamma"])
        M2 = mNum(V2,a2)
        k2V = h * dVdX(V2,A2,M2,T2,P2,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x2,x1),Cf,x2,1/5 * h)
        k2P = h * dPdX(V2,A2,M2,T2,P2,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x2,x1),Cf,x2,1/5 * h)

        x3 = x1 + 3/10 * h
        A3 = geom_Area(x3)
        V3 = V + 3/40 * k1V + 9/40 * k2V
        P3 = P + 3/40 * k1P + 9/40 * k2P
        T3 = newtonRaphson_T(T1, T1, x1, V1, V3, 3/10 * h) 
        a3 = soS(T3,R_mix,gas_properties(T3, P3, Y_mix)["gamma"])
        M3 = mNum(V3,a3)
        k3V = h * dVdX(V3,A3,M3,T3,P3,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x3,x1),Cf,x3,3/10 * h)
        k3P = h * dPdX(V3,A3,M3,T3,P3,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x3,x1),Cf,x3,3/10 * h)

        x4 = x1 + 4/5 * h
        A4 = geom_Area(x4)
        V4 = V + 44/45 * k1V - 56/15 * k2V + 32/9 * k3V
        P4 = P + 44/45 * k1P - 56/15 * k2P + 32/9 * k3P
        T4 = newtonRaphson_T(T1, T1, x1, V1, V4, 4/5 * h) 
        a4 = soS(T4,R_mix,gas_properties(T4, P4, Y_mix)["gamma"])
        M4 = mNum(V4,a4)
        k4V = h * dVdX(V4,A4,M4,T4,P4,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x4,x1),Cf,x4,4/5 * h)
        k4P = h * dPdX(V4,A4,M4,T4,P4,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x4,x1),Cf,x4,4/5 * h)

        x5 = x1 + 8/9 * h
        A5 = geom_Area(x5)
        V5 = V + 19372/6561 * k1V - 25360/2187 * k2V + 64448/6561 * k3V - 212/729 * k4V
        P5 = P + 19372/6561 * k1P - 25360/2187 * k2P + 64448/6561 * k3P - 212/729 * k4P
        T5 = newtonRaphson_T(T1, T1, x1, V1, V5, 8/9 * h)
        a5 = soS(T5,R_mix,gas_properties(T5, P5, Y_mix)["gamma"])
        M5 = mNum(V5,a5)
        k5V = h * dVdX(V5,A5,M5,T5,P5,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x5,x1),Cf,x5,8/9 * h)
        k5P = h * dPdX(V5,A5,M5,T5,P5,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x5,x1),Cf,x5,8/9 * h)

        x6 = x1 + h
        A6 = geom_Area(x6)
        V6 = V + 9017/3168 * k1V - 355/33 * k2V + 46732/5247 * k3V + 49/176 * k4V - 5103/18656 * k5V
        P6 = P + 9017/3168 * k1P - 355/33 * k2P + 46732/5247 * k3P + 49/176 * k4P - 5103/18656 * k5P
        T6 = newtonRaphson_T(T1, T1, x1, V1, V6, 1 * h)
        a6 = soS(T6,R_mix,gas_properties(T6, P6, Y_mix)["gamma"])
        M6 = mNum(V6,a6)
        k6V = h * dVdX(V6,A6,M6,T6,P6,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x6,x1),Cf,x6,h)
        k6P = h * dPdX(V6,A6,M6,T6,P6,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x6,x1),Cf,x6,h)

        #5th order solution 
        v_5Order = V + 35/384 * k1V + 500/1113 * k3V + 125/192 * k4V - 2187/6784 * k5V + 11/84 * k6V
        p_5Order = P + 35/384 * k1P + 500/1113 * k3P + 125/192 * k4P - 2187/6784 * k5P + 11/84 * k6P

        x7 = x1 + h
        A7 = geom_Area(x7)
        V7 = v_5Order
        P7 = p_5Order
        T7 = newtonRaphson_T(T1, T1, x1, V1, V7, 1 * h)
        a7 = soS(T7,R_mix,gas_properties(T7, P7, Y_mix)["gamma"])
        M7 = mNum(V7,a7)
        k7V = h * dVdX(V7,A7,M7,T7,P7,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x7,x1),Cf,x7,h)
        k7P = h * dPdX(V7,A7,M7,T7,P7,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x7,x1),Cf,x7,h)

        #4th order solution
        v_4Order = V + 5179/57600 * k1V + 7571/16695 * k3V + 393/640 * k4V - 92097/339200 * k5V + 187/2100 * k6V + 1/40 * k7V
        p_4Order = P + 5179/57600 * k1P + 7571/16695 * k3P + 393/640 * k4P - 92097/339200 * k5P + 187/2100 * k6P + 1/40 * k7P

        #error estimate 
        errorV = abs(v_5Order - v_4Order)
        errorP = abs(p_5Order - p_4Order)
        err = max(errorV, errorP)

        if errorV > V * tol or errorP > P * tol: #comparing error if either error are > than tol it means that step is too big so i am making it smaller 
            accepted = False 
            sV = 0.9*(tol/errorV)**(1/5)
            sP = 0.9*(tol/errorP)**(1/5)
            s = min(sV, sP)
            hUpdated = min(s * h, 1e-4)
            h = hUpdated
            continue #this just restarts the loop with the updated h value
            
        else: #if both are smaller than tol then I am accepting the time step and then making it bigger 
            accepted = True
            Vnext, Pnext = v_5Order, p_5Order
            xNext = x1 + h
            Tnext = newtonRaphson_T(T1, T1, x1, V1, Vnext, 1 * h)
            s = 2

        if err == 0:
            s = 2

        hUpdated = min(s * h, 1e-4)
    
    return xNext, Vnext, Pnext, Tnext,hUpdated, accepted


#Full Solver
print("__________________________________________________")
def solver(Preburner_TStag,Cf,scale, acceptedScale):
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
    stepList = [preburner_length/1e2]
    mdotList = [mdot_i]
    
    sInitial = gas_properties(Preburner_T, Preburner_P,Y_mix)["s"]
    entropy = [sInitial]

    mdotReconstructed = [mdot_i] #recontruction array to check if calcs are correct 

    while (xList[-1] < nozzle_exit ): #actual for loop for solving everything. from start of preburner to throat 

        xCurrent = xList[-1]     
        hCurrent = stepList[-1] #from step 0 to step 1 and then step 1 to step 2 etc 
        
        Vbefore = velocities[-1]
        Pbefore = pressure[-1]
        Tbefore = temp [-1]
        
        xNext, VCurrent, PCurrent, TCurrent, hNext, accepted = rk45Step(Vbefore,Pbefore, Cf,hCurrent, xCurrent,Tbefore)
        currentMix_properties = gas_properties(TCurrent, PCurrent,Y_mix)
        currentMix_gamma = currentMix_properties["gamma"]

        xList.append(xNext)
        xCurrent = xList[-1]

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

        if MCurrent >= 0.99:
            print("choked at x = ", xCurrent)
            print("choked at region = ", geometry_regions(xCurrent))
            break
        
    if machNum[-1] <0.99:
        print("flow did NOT CHOKE. final Mach number: ", machNum[-1])
        print("x at end of solve: ", xList[-1])

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


    mdotReconsturcted_List = np.array(mdotReconstructed)
    mdot_List = mdotList
    entropy_List = np.array(entropy)

    return {
        "velocity": V_List,
        "pressure": P_List,
        "temperature": T_List,
        "density": rho_List,
        "mach_number": M_List,
        "pressure_stag": pStag_List,
        "temperature_stag": tStag_List,
        "x": x_used_List,
        "area": Area_List,
        "dAdx": dAdx_List,
        "mdot": mdot_List,
        "mdot_reconstructed": mdotReconsturcted_List,
        "entropy": entropy_List,
        "xChoked": xCurrent,
        "Area Ratio": AreaRatio_List, 
        "Choked Area Ratio": Area_List[0]/Area_List[-1], 
        "Initial Pstag" : pStag_List[0],
        "Initial Tstag" : tStag_List[0],
    }

#Sweeping

def chokedLocationResiduals(scale, Cf):
    results = solver(TstagA,Cf,scale,False)
    x_Choke = results["x"][-1]
    residual = throat_loc - x_Choke  #we want this to be zero
    return residual
    
def scaling_InletPressure(Cf):
    max_scale = 1
    max_res = chokedLocationResiduals(max_scale, Cf)
    
    if max_res > 0:
        dir = 1
    elif max_res < 0:
        dir = -1

    for i in range(20):
        
        cur_scale = max_scale + dir * (i/10)
        
        cur_res = chokedLocationResiduals(cur_scale, Cf)
        if cur_res * max_res < 0:
            print("bracket has been found")
            final_scale = cur_scale
            print("scale", cur_scale, "residual", cur_res)
            return final_scale, prev_scale
        
        else:
            print("Scale = ", cur_scale, "Residual = ", cur_res)
            prev_scale = cur_scale #doing this so that i can then return it when i find a bracket 
            final_scale = None

    return final_scale,prev_scale    

Start_Sweep = time.perf_counter()

high_scale,low_scale = scaling_InletPressure(0.005)

def scale_bisec(scale_low, scale_high, Cf):
    res_low = chokedLocationResiduals(scale_low,Cf)
    res_high = chokedLocationResiduals(scale_high,Cf)
    tol = 1e-6

    maxIters = 100

    for i in range(maxIters):

        scale_mid = 0.5 * (scale_low + scale_high)
        res_mid = chokedLocationResiduals(scale_mid,Cf)

        if abs(res_mid) < tol:
            print("Scale", scale_mid, "Residual", res_mid)
            return scale_mid, res_mid
        
        if res_low * res_mid < 0:
            res_high = res_mid
            scale_high = scale_mid
        else:
            res_low = res_mid
            scale_low = scale_mid

final_scale,final_res = scale_bisec(low_scale,high_scale, 0.005)


resultsAtCorrectScale = solver(TstagA,Cf,final_scale,True)
print("Preburner Inlet Conditions")
print("Inlet Velocity", resultsAtCorrectScale["velocity"][0], "m/s")
print("Inlet Pressure", resultsAtCorrectScale["pressure"][0] * 1e-6, "MPa")
print("Inlet Mach Num", resultsAtCorrectScale["mach_number"][0])




End_Sweep = time.perf_counter()

print("Total time for residual sweep: ", End_Sweep - Start_Sweep, "seconds")
'''
plt.figure()
plt.plot(resultsAtCorrectScale["x"], resultsAtCorrectScale["mach_number"])
plt.xlabel("x (m)")
plt.ylabel("Mach Number")
plt.title("Mach Number vs x")
plt.grid()

plt.figure()
plt.plot(resultsAtCorrectScale["x"], resultsAtCorrectScale["temperature_stag"])
plt.xlabel("x (m)")
plt.ylabel("Stagnation Temperature")
plt.title("Stagnation Temperature vs x")
plt.grid()

plt.figure()
plt.plot(resultsAtCorrectScale["x"], resultsAtCorrectScale["pressure_stag"])
plt.xlabel("x (m)")
plt.ylabel("Stagnation Pressure")
plt.title("Stagnation Pressure vs x")
plt.grid()

plt.figure()
plt.plot(resultsAtCorrectScale["x"], resultsAtCorrectScale["entropy"])
plt.xlabel("x (m)")
plt.ylabel("Entropy")
plt.title("Entropy vs x")
plt.grid()
plt.show()


'''
#MCMC
#using black box approach 
chamber_P_Paper = 4.82633 * 1e6

def log_likelihood(Cf):
    high_scale,low_scale = scaling_InletPressure(Cf) #finding bracket 
    final_scale,final_res = scale_bisec(low_scale,high_scale, Cf)   # finding exact scale 
    resultsAtCorrectScale = solver(TstagA,Cf,final_scale,True) #getting exact values at correct scale 
    chamber_P_Predicted =  resultsAtCorrectScale["pressure"][0]

    log_prob = np.sum(stats.norm.logpdf(chamber_P_Paper,loc = chamber_P_Predicted,scale = chamber_P_Predicted * 0.2))
    return log_prob


with pm.Model() as model:
    #prior for Cf
    prior_Cf = pm.Normal("Cf", mu=0.005, sigma=0.00125,lower = 0)
    
    log_like = log_likelihood(prior_Cf)

    pm.Potential("Ini Chamber Pressure Likelihood",log_like)
    
    step = pm.DEMetropolisZ()

    trace = pm.sample(
        draws=2000,
        tune=1000,
        step = step,
        chains=4
    )




#Other Stuff

# Creating Injector Array and Adding Mass Flow from Injector to global mdot array 
'''
Vinj = 0 # m/s speed of N2 being injected (alr converted to x direction)
Dinj = 0.003175 #m Injector diameter
Ainj = np.pi * (Dinj/2)**2 #m^2 
injMdot = rho_preburner * Vinj * Ainj #kg/s


injIndex = int(x_injLocation/dx) #index of the center of the injector 
injIndexRange = int((Dinj/2)/dx) #range is +- so this is only half of total inj diameter
inj_array = np.zeros(len(xList_rk45)) #array to hold injector locations (0 means no injector 1 means injector, 2 means post injector)

startInj = max(0, int(injIndex - injIndexRange)) #start index of injector
endInj = min(len(xList_rk45)-1, int(injIndex + injIndexRange)) #end index of injector
inj_array[startInj:endInj+1] = 1 #mark injector location, +1 to include end index
inj_array[endInj+1:] = 2    #mark post injector locations +1 to start 1 after that end index #going to go over wtf this means 

mdot[startInj:endInj+1] = mdot_i + injMdot #add injector mass flow to main flow at injector location
mdot[endInj+1:] = mdot_i + injMdot #post injector mass flow


#setting up basic mcmc model for just the friction coeff 
def log_prior(Cf):
    if Cf < 0:
        return -np.inf
    else:
        return stats.norm.logpdf(Cf, loc=0.005, scale=0.00125) 
    
def log_likelihood(Cf):
    solve = consecutive_solves(Pstag_Air, Pstag_H2, uA, T_air, uB, T_H2, n_solves, Cf)
    return np.sum(stats.norm.logpdf(imposed_Mdot, loc=solve["Choked Mdot"], scale=0.1*imposed_Mdot)) #likelihood function based on how close the choked mass flow from the solve is to the imposed mass flow, with a standard deviation of 10% of the imposed mass flow

def log_posterior(Cf):
    logPrior = log_prior(Cf)
    if logPrior == -np.inf:
        return -np.inf
    logLikelihood = log_likelihood(Cf)
    return logPrior + logLikelihood

Cf_grid = np.linspace(0.003, 0.007, 100) #grid of friction coeffs to evaluate

iters = 1000
Cf_initial = 0.0025
Cf_chain = np.zeros(iters)
Cf_chain[0] = Cf_initial

for i in range(1, iters):
    Cf_current = Cf_chain[i-1]
    Cf_guess = np.random.normal(Cf_current, 0.0005) #small random walk proposal distribution

    log_posterior_current = log_posterior(Cf_current)
    log_posterior_guess = log_posterior(Cf_guess)
    acceptance_ratio = np.exp(log_posterior_guess - log_posterior_current)

    if acceptance_ratio >= np.random.uniform(0, 1):
        Cf_chain[i] = Cf_guess
    else:
        Cf_chain[i] = Cf_current

log_posterior_values = [log_posterior(Cf) for Cf in Cf_grid]
posterior_values = np.exp(log_posterior_values - np.max(log_posterior_values)) #subtracting max log posterior for numerical stability

plt.figure()
plt.plot(Cf_grid, posterior_values, 'o-')
plt.xlabel("Friction Coefficient (Cf)")
plt.ylabel("Log Posterior")
plt.title("Log Posterior vs Friction Coefficient")
plt.grid()

plt.figure()
plt.hist(Cf_chain, bins=50, density=True, alpha=0.6, label="MCMC samples")

posterior_curve = np.exp(log_posterior_values - np.max(log_posterior_values))
posterior_curve = posterior_curve / np.trapezoid(posterior_curve, Cf_grid)

plt.plot(Cf_grid, posterior_curve, label="Posterior curve")
plt.xlabel("Pipe Friction Factor (Cf)")
plt.ylabel("Density")
plt.legend()
plt.show()

'''
