import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import cantera as ct
from scipy import stats

#preburner inlet conditions 
R = 296.8 #J/kgK
L_tochoke = 0.5 #m 
L_total = 0.7   #m
f_darcy = 0.02
Cf = f_darcy/4
gamma0 = 1.4

Vinj = None 
injMdot = 0

r0 = 0.05 #m
D0 = 2 * r0
A0 = np.pi * r0**2
Astar = A0/25
rStar = np.sqrt(Astar/np.pi)
r_exit = r0

x1_conv = 0.5 * L_tochoke #defining range where the area starts to converge 
x2_conv = L_tochoke #where convergence ends and area is minimum
x_injLocation = 0.15 * L_tochoke #m



x1_div = L_tochoke #defining range where area starts to diverge
x2_div = L_total

def smoothstep(xi):
    return 6*xi**5 - 15*xi**4 + 10*xi**3

def dsmoothstep_dxi(xi):
    return 30*xi**4 - 60*xi**3 + 30*xi**2

def radius(x):
    if x <= x1_conv: #constant area section before convergence starts
        return r0
    elif x <= x2_conv: #converging section 
        xi = (x - x1_conv) / (x2_conv - x1_conv)

        return r0 - (r0 - rStar) * smoothstep(xi)
    
    elif x <= x2_div: #diverging section
        xi = (x - x1_div) / (x2_div - x1_div)
        return rStar + (r_exit - rStar) * smoothstep(xi)
    else:
        return r_exit

def Area(x):
    r = radius(x)
    return np.pi * r**2

def Dh(x):
    return 2 * radius(x)

def dAdx(x):
    if x < x1_conv:
        return 0.0
    elif x <= x2_conv:
        xi = (x - x1_conv) / (x2_conv - x1_conv)
        drdx = -(r0 - rStar) * dsmoothstep_dxi(xi) / (x2_conv - x1_conv)
        r = radius(x)
        return 2 * np.pi * r * drdx
    elif x <= x2_div:
        xi = (x - x1_div) / (x2_div - x1_div)
        drdx = (r_exit - rStar) * dsmoothstep_dxi(xi) / (x2_div - x1_div)
        r = radius(x)
        return 2 * np.pi * r * drdx
    else:
        return 0.0

def mNum(v,a): #mach number 
    M = v/a
    return M

def soS(T): #solving for a using variable gamma and Cp 
    a = np.sqrt(gamma(T) * R * T)
    return a

#dhtdx profile
qtotal = 200e3
x_s = 0

def dHtdx(x):#dht/dx profile - quadtratic Ht
    return (qtotal/L_tochoke) * (x - x_s)

# Nasa Polynomials

def CpNasa(T): #solving variable Cp with NASA polynomials for N2 
    return (0.02926640*10**2 + 0.14879768E-02 * T - 0.05684760E-05 * T**2 + 0.10097038E-09 * T**3 - 0.06753351E-13 * T**4) * R

def gamma(T):#solving for gamma using 
    SHC = CpNasa(T) / (CpNasa(T) - R)
    return SHC
     
def hTNasa(T): #solving for static enthalpy using NASA polynomials for N2
    return (0.02926640*10**2 + (0.14879768E-02 * T)/2 - (0.05684760E-05 * T**2)/3 + (0.10097038E-09 * T**3)/4 - (0.06753351E-13 * T**4)/5) * R * T

#initial conditions and mixing solver  
d1 = 0.25#in
d2 = d1
d3 = 0.25

r1 = d1 / 2
r2 = r1
r3 = d3 / 2

A1 = (r1/39.37)**2 * np.pi #m^2
A2 = (r2/39.37)**2 * np.pi #m^2
A3 = (r3/39.37)**2 * np.pi #m^2
A_A = A1 + A2 #m^2
A_airInjs = 3 * A_A
A_H2Injs = 3 * A3

P1 = 20*1e6 #Pa
P2 = 20*1e6 #Pa
P3 = 20*1e6 #Pa
PA = P1 #Pa. Pa is equal to P1 and P2 because they are the same injector and they are connected to the same plenum.
PB = P3 #Pa

TA = 300 #K
TA_2 = 300 #K
TB = 300 #K

M1 = 0.95
M2 = 0.95
M3 = 0.95

a1 = soS(TA)
a2 = soS(TA_2)
a3 = soS(TB)

uA = M1 * a1
uA_2 = M2 * a2
uB = M3 * a3

TstagA = TA * (1 + (gamma(TA) - 1)/2 * M1**2)
TstagB = TB * (1 + (gamma(TB) - 1)/2 * M3**2)

PstagA = PA * (1 + (gamma(TA) - 1)/2 * M1**2)**(gamma(TA)/(gamma(TA)-1))
PstagB = PB * (1 + (gamma(TB) - 1)/2 * M3**2)**(gamma(TB)/(gamma(TB)-1))

rho1 = P1/(R*TA) #will have to define R for diff species etc - but for now they are all the same
rho2 = P2/(R*TA_2)
rho3 = P3/(R*TB)

print("rho1", rho1)

mdot1 = rho1 * A_airInjs * uA
mdot2 = rho2 * A_airInjs * uA_2
mdot3 = rho3 * A_H2Injs * uB

mdotA = mdot1 + mdot2 #injector 1 and 2 are the same so can just add them together
mdotB = mdot3 #big injector
mdot_i = mdotA + mdotB

A_CV_END = A0 #area at the end of the CV is the same as the area at the start of the preburner inlet.
def delMdotdx(mdotn1, mdotn,x1,x): #dmdot/dx function
    return (mdotn1 - mdotn)/(x1-x)

def mdotFuncX (x):
    if x < x_injLocation: #pre injector mdot
        return mdot_i
    else:   #post injector mdot
        return mdot_i + injMdot 

# Functions for solving ODEs -functions that solve for dV/dx and dP/dx - based off of 1d sharpios flow equations 

def dVdX (V,A,M,cp,T,dAdX,localdHtdx,mdot,DMDOTDX,Dh, Cf): #first 4 parts of sharpios 1d flow eqn converted to dV/dx
    gammA = gamma(T)
    term1 = ((-V)/(A * (1 - M**2)))* dAdX
    term2 = ((V/((1-M**2) * cp * T)) * localdHtdx)
    term3 = ((gammA *M**2)/(2 * (1 - M**2)))
    term4 = ((((4 * Cf * V)/Dh)) - (2*(Vinj/mdot) * DMDOTDX))
    term5 = (((V*(1 + gammA * M**2))/((1-M**2)*mdot)) * (DMDOTDX))
    return term1 + term2 + (term3*term4) + term5

def dPdX (P,V,A,M,cp,T,DADX,localdHtdx,mdot,DMDOTDX,Dh, Cf): #first 4 parts of sharpios 1d flow eqn converted to dP/dx
    gammA = gamma(T)
    term1 = ((gammA * M**2 * P)/(A * (1 - M**2))) * DADX
    term2 = -(((gammA * M**2 * P)/((1-M**2) * cp * T)) * localdHtdx)
    term3  = -((gammA * M**2 * (1 + (gammA-1) * M**2))/(2 * (1 - M**2)))
    term4 = (((4 * Cf * (P/Dh))) - (2 * ((Vinj * P)/(mdot * V)) * (DMDOTDX)))
    term5 = -(((2 * gammA * M**2 * (1 + ((gammA-1)/2) *M**2)*P)/((1-M**2)*mdot)) * (DMDOTDX))
    return term1 + term2 + (term3 * term4) + term5

def pressureStagFunc(P,M,T):
    gammA = gamma(T)
    Pstag = P * (1 + (gammA - 1)/2 * M**2)**(gammA/(gammA-1))
    return Pstag
def temperatureStagFunc(T,M):
    gammA = gamma(T)
    Tstag = T * (1 + (gammA - 1)/2 * M**2)
    return Tstag

def choked_massFlow(Pstag,T,Astar,Tstag):
    mdot_choke = (Pstag * Astar/np.sqrt(Tstag)) * np.sqrt(gamma(T) / R) * ((gamma(T) + 1)/2)**(-(gamma(T) + 1)/(2*(gamma(T)-1)))
    return mdot_choke

def pstag_predicted(mdot,T,Astar,Tstag):
    Pstag_pred = mdot * (np.sqrt(Tstag)/Astar) / (np.sqrt(gamma(T) / R) * ((gamma(T) + 1)/2)**(-(gamma(T) + 1)/(2*(gamma(T)-1))))
    return Pstag_pred



def E1_CV(ui,Ti,uA,uB,TA,TB):
    return ui - (mdotA/mdot_i) * uA - (mdotB/mdot_i) * uB - (mdotA * R * TA)/(mdot_i * uA) - (mdotB * R * TB)/(mdot_i * uB) + (R * Ti)/ui

def E2_CV(ui,Ti,uA,uB,TA,TB):
    hi = hTNasa(Ti)
    hA = hTNasa(TA)
    hB = hTNasa(TB)
    return (hi + ui**2/2) - (mdotA/mdot_i) * (hA + uA**2/2) - (mdotB/mdot_i) * (hB + uB**2/2)

def E3_InjA_CV(PstagA_2, uA_2, TA_2): #third cv equation check power point for indepth breakdown
    part1 = (PstagA_2/(R * TstagA))
    part2 = (1 + ((gamma(TA_2) - 1)/2) * ((uA_2/soS(TA_2))**2))
    part3 = (1 - 1*(gamma(TA_2)/(gamma(TA_2)-1)))
    rhoA = part1 * part2**part3
    return rhoA * uA_2 * A_airInjs - mdotA

def E4_InjB_CV(PstagB_2, uB_2, TB_2): #third cv equation check power point for indepth breakdown
    part1 = (PstagB_2/(R * TstagB))
    part2 = (1 + ((gamma(TB_2) - 1)/2) * ((uB_2/soS(TB_2))**2))
    part3 = (1 - 1*(gamma(TB_2)/(gamma(TB_2)-1)))
    rhoB = part1 * part2**part3
    return rhoB * uB_2 * A_H2Injs - mdotB

def E5_InjA_CV(TA_2,uA_2):
    return TA_2 * (1 + (gamma(TA_2) - 1)/2 * (uA_2/soS(TA_2))**2) - TstagA

def E6_InjB_CV(TB_2,uB_2):
    return TB_2 * (1 + (gamma(TB_2) - 1)/2 * (uB_2/soS(TB_2))**2) - TstagB


def CV_toPreburner(u2,T2,uA,uB,TA,TB): #this is newton raphson for the CV it goes from state 1 (once gasses have mixed) to state 2 (preburner inlet) 
    numIters = 0        #cut down the system of equations to 2 equations and 2 unkowns so just solving till im under tolorence 
    tol = 1e-8

    E1 = E1_CV(u2,T2,uA,uB,TA,TB)
    E2 = E2_CV(u2,T2,uA,uB,TA,TB)
    E_vec = np.array([E1, E2])

    while(np.linalg.norm(E_vec, 2) >= tol and numIters <= 100):
        #numerical jacobian 

        deltaU = u2/1e8  #the delta or perturbation will be updating as u2 and T2 update to make sure its not too big or too small.
        deltaT = T2/1e8


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


    E3 = E3_InjA_CV(Pstag_A2,uA_2, TA_2)
    E5 = E5_InjA_CV(TA_2,uA_2)
    E_vec = np.array([E5, E3])

    while(np.linalg.norm(E_vec, 2) >= tol and numIters <= 100):
        deltaU = max(abs(uA_2)*1e-6, 1e-6)
        deltaT = max(abs(TA_2)*1e-6, 1e-6)

        #partial derivatives for numerical jacobian
        dE5du = (E5_InjA_CV(TA_2, uA_2+ deltaU) - E5)/deltaU
        dE5dT = (E5_InjA_CV(TA_2 + deltaT, uA_2) - E5)/deltaT
        dE3du = (E3_InjA_CV(Pstag_A2, uA_2 + deltaU, TA_2) - E3)/deltaU
        dE3dT = (E3_InjA_CV(Pstag_A2, uA_2, TA_2 + deltaT) - E3)/deltaT

        J = np.array([[dE5du, dE5dT], [dE3du, dE3dT]])
        deltas = np.linalg.solve(J, -E_vec)

        uA_2 += deltas[0]
        TA_2 += deltas[1]
        
        E5 = E5_InjA_CV(TA_2, uA_2)   #updating E5 and E3 values after updating u2 and T2 to check for convergence and to move
        E3 = E3_InjA_CV(Pstag_A2, uA_2, TA_2)   
        E_vec = np.array([E5, E3])   

        numIters += 1
        if numIters > 100 or not np.isfinite(uA_2) or not np.isfinite(TA_2) or TA_2 <= 0:
         raise RuntimeError("InjA solve failed")

    return uA_2, TA_2

def InjB_Loss_CV(Pstag_B2,uB_2, TB_2):
    numIters = 0
    tol = 1e-8


    E4 = E4_InjB_CV(Pstag_B2,uB_2, TB_2)
    E6 = E6_InjB_CV(TB_2,uB_2)
    E_vec = np.array([E6, E4])

    while(np.linalg.norm(E_vec, 2) >= tol and numIters <= 100):
        deltaU = max(abs(uB_2)*1e-6, 1e-6)
        deltaT = max(abs(TB_2)*1e-6, 1e-6)

        #partial derivatives for numerical jacobian
        dE6du = (E6_InjB_CV(TB_2, uB_2 + deltaU) - E6)/deltaU
        dE6dT = (E6_InjB_CV(TB_2 + deltaT, uB_2) - E6)/deltaT
        dE4du = (E4_InjB_CV(Pstag_B2, uB_2 + deltaU, TB_2) - E4)/deltaU
        dE4dT = (E4_InjB_CV(Pstag_B2, uB_2, TB_2 + deltaT) - E4)/deltaT

        J = np.array([[dE6du, dE6dT], [dE4du, dE4dT]])
        deltas = np.linalg.solve(J, -E_vec)

        uB_2 += deltas[0]
        TB_2 += deltas[1]

        E6 = E6_InjB_CV(TB_2, uB_2)   #updating E6 and E4 values after updating uB_2 and B_2 to check for convergence and to move
        E4 = E4_InjB_CV(Pstag_B2, uB_2, TB_2)   
        E_vec = np.array([E6, E4])   
        

        numIters += 1

        if numIters > 100 or not np.isfinite(uB_2) or not np.isfinite(TB_2) or TB_2 <= 0:
         raise RuntimeError("InjB solve failed")
    
    return uB_2, TB_2


#rk45
def rk45Step(V,P,Cf, h, x): #add stages for each mdot 
    accepted = False 
    tol = 1e-6
    while (accepted != True):
        mdot_Current = mdotFuncX(x)
        mdot_Prev = mdotFuncX(x-h)

        x1 = x
        dHtdx1 = dHtdx(x1)
        A1 = Area(x1)
        print("A1", A1)
        V1 = V
        P1 =  P
        rho1 = mdot_Current/(A1 * V1)
        T1 = P1/(rho1 * R)
        cp1 = CpNasa(T1)
        a1 = soS(T1)
        M1 = mNum(V1,a1)
        k1V = h * dVdX(V1,A1,M1,cp1,T1,dAdx(x1),dHtdx1,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x1,x1-h),Dh(x1), Cf)
        k1P = h * dPdX(P1,V1,A1,M1,cp1,T1,dAdx(x1),dHtdx1,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x1,x1-h),Dh(x1), Cf)
        print("V1", V1, "P1", P1,"T1", T1, "rho1", rho1)

        x2 = x1 + 1/5 * h
        dHtdx2 = dHtdx(x2)
        A2 = Area(x2)
        V2 = V + 1/5 * k1V 
        P2 = P + 1/5 * k1P

        rho2 = mdot_Current/(A2 * V2)
        T2 = P2/(rho2 * R)
        cp2 = CpNasa(T2)
        a2 = soS(T2)
        M2 = mNum(V2,a2)
        k2V = h * dVdX(V2,A2,M2,cp2,T2,dAdx(x2),dHtdx2,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x2,x1),Dh(x2), Cf)
        k2P = h * dPdX(P2,V2,A2,M2,cp2,T2,dAdx(x2),dHtdx2,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x2,x1),Dh(x2), Cf)
        print("V2", V2, "P2", P2,"T2", T2)


        x3 = x1 + 3/10 * h
        dHtdx3 = dHtdx(x3)
        A3 = Area(x3)
        V3 = V + 3/40 * k1V + 9/40 * k2V
        P3 = P + 3/40 * k1P + 9/40 * k2P
        rho3 = mdot_Current/(A3 * V3)
        T3 = P3/(rho3 * R)
        cp3 = CpNasa(T3)
        a3 = soS(T3)
        M3 = mNum(V3,a3)
        k3V = h * dVdX(V3,A3,M3,cp3,T3,dAdx(x3),dHtdx3,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x3,x1),Dh(x3), Cf)
        k3P = h * dPdX(P3,V3,A3,M3,cp3,T3,dAdx(x3),dHtdx3,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x3,x1),Dh(x3), Cf)
      
        x4 = x1 + 4/5 * h
        dHtdx4 = dHtdx(x4)
        A4 = Area(x4)
        V4 = V + 44/45 * k1V - 56/15 * k2V + 32/9 * k3V
        P4 = P + 44/45 * k1P - 56/15 * k2P + 32/9 * k3P
        rho4 = mdot_Current/(A4 * V4)
        T4 = P4/(rho4 * R)
        cp4 = CpNasa(T4)
        a4 = soS(T4)
        M4 = mNum(V4,a4)
        k4V = h * dVdX(V4,A4,M4,cp4,T4,dAdx(x4),dHtdx4,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x4,x1),Dh(x4), Cf)
        k4P = h * dPdX(P4,V4,A4,M4,cp4,T4,dAdx(x4),dHtdx4,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x4,x1),Dh(x4), Cf)

        x5 = x1 + 8/9 * h
        dHtdx5 = dHtdx(x5)
        A5 = Area(x5)
        V5 = V + 19372/6561 * k1V - 25360/2187 * k2V + 64448/6561 * k3V - 212/729 * k4V
        P5 = P + 19372/6561 * k1P - 25360/2187 * k2P + 64448/6561 * k3P - 212/729 * k4P
        rho5 = mdot_Current/(A5 * V5)
        T5 = P5/(rho5 * R)
        cp5 = CpNasa(T5)
        a5 = soS(T5)
        M5 = mNum(V5,a5)
        k5V = h * dVdX(V5,A5,M5,cp5,T5,dAdx(x5),dHtdx5,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x5,x1),Dh(x5), Cf)
        k5P = h * dPdX(P5,V5,A5,M5,cp5,T5,dAdx(x5),dHtdx5,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x5,x1),Dh(x5), Cf)

        x6 = x1 + h
        dHtdx6 = dHtdx(x6)
        A6 = Area(x6)
        V6 = V + 9017/3168 * k1V - 355/33 * k2V + 46732/5247 * k3V + 49/176 * k4V - 5103/18656 * k5V
        P6 = P + 9017/3168 * k1P - 355/33 * k2P + 46732/5247 * k3P + 49/176 * k4P - 5103/18656 * k5P
        rho6 = mdot_Current/(A6 * V6)
        T6 = P6/(rho6 * R)
        cp6 = CpNasa(T6)
        a6 = soS(T6)
        M6 = mNum(V6,a6)
        k6V = h * dVdX(V6,A6,M6,cp6,T6,dAdx(x6),dHtdx6,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x6,x1),Dh(x6), Cf)
        k6P = h * dPdX(P6,V6,A6,M6,cp6,T6,dAdx(x6),dHtdx6,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x6,x1),Dh(x6), Cf)

        #5th order solution 
        v_5Order = V + 35/384 * k1V + 500/1113 * k3V + 125/192 * k4V - 2187/6784 * k5V + 11/84 * k6V
        p_5Order = P + 35/384 * k1P + 500/1113 * k3P + 125/192 * k4P - 2187/6784 * k5P + 11/84 * k6P


        x7 = x1 + h
        dHtdx7 = dHtdx(x7)
        A7 = Area(x7)
        V7 = v_5Order
        P7 = p_5Order
        rho7 = mdot_Current/(A7 * V7)
        T7 = P7/(rho7 * R)
        cp7 = CpNasa(T7)
        a7 = soS(T7)
        M7 = mNum(V7,a7)
        k7V = h * dVdX(V7,A7,M7,cp7,T7,dAdx(x7),dHtdx7,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x7,x1),Dh(x7), Cf)
        k7P = h * dPdX(P7,V7,A7,M7,cp7,T7,dAdx(x7),dHtdx7,mdot_Current,delMdotdx(mdot_Current,mdot_Prev,x7,x1),Dh(x7), Cf)

        #4th order solution
        v_4Order = V + 5179/57600 * k1V + 7571/16695 * k3V + 393/640 * k4V - 92097/339200 * k5V + 187/2100 * k6V + 1/40 * k7V
        p_4Order = P + 5179/57600 * k1P + 7571/16695 * k3P + 393/640 * k4P - 92097/339200 * k5P + 187/2100 * k6P + 1/40 * k7P

        #error estimate 
        errorV = abs(v_5Order - v_4Order)
        errorP = abs(p_5Order - p_4Order)
        err = max(errorV, errorP)
        if errorV > V * tol or errorP > P * tol: #comparing error if either error are > than tol it means that step is too big so i am making it smaller 
            accepted = False 
            Vnext,Pnext = V, P
            xNext = x1
            sV = 0.9*(tol/errorV)**(1/5)
            sP = 0.9*(tol/errorP)**(1/5)
            s = min(sV, sP)
            
        else: #if both are smaller than tol then I am accepting the time step and then making it bigger 
            accepted = True
            Vnext, Pnext = v_5Order, p_5Order
            xNext = x1 + h
            s = 2
        if err == 0:
            s = 2

        hUpdated = s * h
        h = hUpdated
    
    return xNext, Vnext, Pnext, hUpdated, accepted

# Solving/Defining inital conditons for the preburner inlet - initializing arrays to store values etc

#consecutive solves
def consecutive_solves(pstagA_2,pstagB_2, u2_guess_A, T2_guess_A, u2_guess_B, T2_guess_B, numSolves, Cf):
    global mdot,Vinj #making them global so that i can use them in rk45 and ode functions
    Vinj = 0

    u_InjA_2, T_InjA_2 = InjA_Loss_CV(pstagA_2, u2_guess_A, T2_guess_A)
    u_InjB_2, T_InjB_2 = InjB_Loss_CV(pstagB_2, u2_guess_B, T2_guess_B)

    u_preburner_guess = (u_InjA_2 * A_airInjs + u_InjB_2 * A_H2Injs)/(A_CV_END)
    T_preburner_guess = (T_InjA_2 * mdotA + T_InjB_2 * mdotB)/(mdot_i)

    u_preburner, T_preburner = CV_toPreburner(u_preburner_guess, T_preburner_guess, u_InjA_2, u_InjB_2, T_InjA_2, T_InjB_2)
    M_Preburner_Inlet = u_preburner/np.sqrt(gamma(T_preburner) * R * T_preburner)
    
    Tstag_Preburner = temperatureStagFunc(T_preburner, M_Preburner_Inlet)

    Pstag_Preburner = pstagA_2 #*12

    P_preburner = (1 + (gamma(T_preburner) - 1)/2 * M_Preburner_Inlet**2)**(-gamma(T_preburner)/(gamma(T_preburner)-1)) * Pstag_Preburner
    rho_preburner = P_preburner/(R * T_preburner)

    mdot_preburner = rho_preburner * u_preburner * A_CV_END
    
    #print("Preburner Mach Number:", M_Preburner_Inlet)
    print("Preburner Pressure:", P_preburner * 1e-6, "MPa")
    print("Preburner Temperature:", T_preburner)
    print("Preburner velocity:", u_preburner)
    print("Preburner rho", rho_preburner)
    print("Imposed mdot", mdot_i, "solved mdot", mdot_preburner)


    temp = [T_preburner]                # creating fresh arrays in function 
    velocities = [u_preburner]
    pressure = [P_preburner]
    pStag = [Pstag_Preburner]
    tStag = [Tstag_Preburner]
    machNum = [M_Preburner_Inlet]
    density = [rho_preburner]
    areaList = [Area(0)]
    dAdxList = [0.0]
    areaRatio = [1.0]

    xList = [0.0] #this list starts at the preburner 
    stepList = [L_tochoke/1000]
    mdotList = [mdot_i]
    
    gas = ct.Solution('gri30.yaml')
    gas.TPX = T_preburner, P_preburner, {'N2': 1.0}
    sInitial = gas.entropy_mass
    entropy = [sInitial]


    mdotReconstructed = [mdot_preburner] #recontruction array to check if calcs are correct 

    #solving flow through Preburner
    
    while (xList[-1] < L_tochoke *.0005): #actual for loop for solving everything. 

        xCurrent = xList[-1]     
        hCurrent = stepList[-1] #from step 0 to step 1 and then step 1 to step 2 etc 
        
        Vbefore = velocities[-1]
        Pbefore = pressure[-1]
        Tbefore = temp [-1]

        print("Vbefore", Vbefore, "Pbefore", Pbefore * 1e-6, "Tbefore", Tbefore)
        xNext, VCurrent, PCurrent, hNext, accepted = rk45Step(Vbefore,Pbefore, Cf,hCurrent, xCurrent)
      
        xList.append(xNext)
        xCurrent = xList[-1]

        mdotlocal = mdotFuncX(xCurrent)
        stepList.append(hNext)

        velocities.append(VCurrent)
        pressure.append(PCurrent)

        rhoCurrent = mdotlocal/(Area(xCurrent) * VCurrent) 
        density.append(rhoCurrent)

        TCurrent = PCurrent/(rhoCurrent * R)
        temp.append(TCurrent)

        mdotReconstructed.append(rhoCurrent * VCurrent * Area(xCurrent))
        mdotList.append(mdotFuncX(xCurrent))

        aCurrent = soS(TCurrent)
        MCurrent = mNum(VCurrent,aCurrent)
        machNum.append(MCurrent)

        Pstag_current = pressureStagFunc(PCurrent, MCurrent, TCurrent)
        pStag.append(Pstag_current)

        Tstag_current = temperatureStagFunc(TCurrent, MCurrent)
        tStag.append(Tstag_current)

        gas.TP = TCurrent, PCurrent
        sCurrent = gas.entropy_mass
        entropy.append(sCurrent)
       
        print("#################################################")
        print("V:", Vbefore, "->", VCurrent)
        print("P:", Pbefore * 1e-6, "->", PCurrent * 1e-6)
        print("T:", TCurrent)
        print("Rho:", rhoCurrent)

        if MCurrent >= 0.99:
            print("choked at x = ", xCurrent)
            break
        
        
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
    
    choked_mdot = choked_massFlow(pStag_List[-1], T_List[-1], Astar, tStag_List[-1])
    StagPressure_Predicted = pstag_predicted(mdot_List[-1], T_List[-1], Astar, tStag_List[-1])
    #print("Predicted choked Pstag: ", StagPressure_Predicted * 1e-6, "MPa")
    #print("Pstag_Preburner: ", Pstag_Preburner * 1e-6, "MPa")

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
        "Choked Mdot" : choked_mdot,
        "Predicted Pstag from Mdot" : StagPressure_Predicted,
        "Number of Solves": numSolves
    }
n_solves = 1 #number of solves. 
results = consecutive_solves(PstagA, PstagB, uA, TA, uB, TB, n_solves, 0.005)
'''
plt.figure()
plt.plot(results["x"],results["mdot"], label = "Imposed Mdot")
plt.plot(results["x"],results["mdot_reconstructed"], label = "Reconstructed Mdot")
plt.xlabel("X (m)")
plt.ylabel("Mdot (kg/s)")
plt.legend()
plt.grid()


plt.figure()
plt.plot(results["x"],results["mach_number"])
plt.xlabel("X (m)")
plt.ylabel("Mach Num ")
plt.grid()


plt.figure()
plt.plot(results["x"],results["temperature"])
plt.xlabel("X (m)")
plt.ylabel("temperature ")
plt.grid()

plt.figure()
plt.plot(results["x"],results["pressure"])
plt.xlabel("X (m)")
plt.ylabel("pressure ")
plt.grid()
plt.show()
'''

imposed_Mdot = mdot_i #m

'''
#setting up basic mcmc model for just the friction coeff 
def log_prior(Cf):
    if Cf < 0:
        return -np.inf
    else:
        return stats.norm.logpdf(Cf, loc=0.005, scale=0.00125) 
    
def log_likelihood(Cf):
    solve = consecutive_solves(PstagA, PstagB, uA, TA, uB, TB, n_solves, Cf)
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


#function to basically sweep through a bunch of diff scales to see if I can find a good bracket for bisection method

def safe_chokedResiduals(scale,numSolves,Cf):
    try: #i am using try and except to catch any errors such as cantera errors etc, and then just returning None for those cases so that the code does not crash
        results = consecutive_solves(PstagA*scale, PstagB*scale, uA, TA, uB, TB, numSolves, Cf)
        mdotChoke = results["Choked Mdot"]
        Predicted_stagPressure = results["Predicted Pstag from Mdot"]
        FinalStagPressure = results["pressure_stag"][-1]
        residual = imposed_Mdot - mdotChoke  #we want this to be zero

        if not np.isfinite(mdotChoke):
            return None, None, None, False, None, None

        if not np.isfinite(residual):
            return None, None, None, False, None, None
        
        return residual, mdotChoke, results,True, Predicted_stagPressure, FinalStagPressure
    
    except:
        return None, None, None, False, None, None
    
#bisection method using choked mass flow rate 
#bisection method 

def hybrid_newton_bisection(scale_low, scale_high, tol=1e-6, max_iters=100):

    low_res, low_mdotChoke, low_results, low_ok, low_Predicted_stagPressure, low_FinalStagPressure = \
        safe_chokedResiduals(scale_low, n_solves, Cf)

    high_res, high_mdotChoke, high_results, high_ok, high_Predicted_stagPressure, high_FinalStagPressure = \
        safe_chokedResiduals(scale_high, n_solves, Cf)

    if (not low_ok) or (not high_ok):
        raise RuntimeError("Initial bracket has failed flow solves.")

    if not np.isfinite(low_res) or not np.isfinite(high_res):
        raise RuntimeError("Initial bracket contains invalid residuals.")

    if low_res * high_res > 0:
        raise RuntimeError("No sign change in initial bracket.")

    history = []

    # start in the middle of the bracket
    scale_current = 0.5 * (scale_low + scale_high)
    current_res, current_mdotChoke, current_results, current_ok, current_Predicted_stagPressure, current_FinalStagPressure = \
        safe_chokedResiduals(scale_current, n_solves, Cf)

    if (not current_ok) or (not np.isfinite(current_res)):
        scale_current = scale_low if abs(low_res) < abs(high_res) else scale_high
        current_res = low_res if abs(low_res) < abs(high_res) else high_res

    for i in range(max_iters):

        bracket_width = abs(scale_high - scale_low)

        print(
            "Iter", i,
            "scale_current:", scale_current,
            "res:", current_res,
            "bracket:", scale_low, scale_high,
            "width:", bracket_width
        )

        history.append({
            "iter": i,
            "scale_low": scale_low,
            "scale_high": scale_high,
            "scale_current": scale_current,
            "res_low": low_res,
            "res_high": high_res,
            "res_current": current_res,
            "bracket_width": bracket_width
        })

        if abs(current_res) < tol:
            print("Hybrid method converged by residual.")
            return scale_current, current_res, history

        if bracket_width < tol:
            print("Hybrid method converged by bracket width.")
            return scale_current, current_res, history

        dScale = max(abs(scale_current) * 1e-5, 1e-7)

        # make sure derivative test stays inside (0, 1)
        scale_deriv = scale_current + dScale

        if scale_deriv >= 1.0:
            scale_deriv = scale_current - dScale

        if scale_deriv <= 0.0:
            scale_deriv = scale_current + dScale

        deriv_res, deriv_mdotChoke, deriv_results, deriv_ok, deriv_Predicted_stagPressure, deriv_FinalStagPressure = \
            safe_chokedResiduals(scale_deriv, n_solves, Cf)

        use_newton = True

        if (not deriv_ok) or (not np.isfinite(deriv_res)):
            use_newton = False
            dres_dscale = None
        else:
            dres_dscale = (deriv_res - current_res) / (scale_deriv - scale_current)

            if (not np.isfinite(dres_dscale)) or abs(dres_dscale) < 1e-14:
                use_newton = False

        if use_newton:
            scale_newton = scale_current - current_res / dres_dscale

            # keep scale physically valid
            newton_inside_bracket = (scale_low < scale_newton < scale_high)
            newton_inside_physical = (0.0 < scale_newton < 1.0)

            # reject if Newton jump is larger than half the current bracket
            newton_jump = abs(scale_newton - scale_current)
            jump_ok = newton_jump <= 0.5 * bracket_width

            if newton_inside_bracket and newton_inside_physical and jump_ok:
                scale_trial = scale_newton
                method_used = "newton"
            else:
                scale_trial = 0.5 * (scale_low + scale_high)
                method_used = "bisection"
        else:
            scale_trial = 0.5 * (scale_low + scale_high)
            method_used = "bisection"

        trial_res, trial_mdotChoke, trial_results, trial_ok, trial_Predicted_stagPressure, trial_FinalStagPressure = \
            safe_chokedResiduals(scale_trial, n_solves, Cf)

        # If trial solve fails, force bisection
        if (not trial_ok) or (not np.isfinite(trial_res)):
            scale_trial = 0.5 * (scale_low + scale_high)
            method_used = "bisection_after_failed_newton"

            trial_res, trial_mdotChoke, trial_results, trial_ok, trial_Predicted_stagPressure, trial_FinalStagPressure = \
                safe_chokedResiduals(scale_trial, n_solves, Cf)

            if (not trial_ok) or (not np.isfinite(trial_res)):
                raise RuntimeError("Both Newton and bisection trial failed.")

        print("method used:", method_used, "scale_trial:", scale_trial, "trial_res:", trial_res)

        if low_res * trial_res < 0:
            scale_high = scale_trial
            high_res = trial_res
        else:
            scale_low = scale_trial
            low_res = trial_res

        scale_current = scale_trial
        current_res = trial_res

    raise RuntimeError("Hybrid Newton-bisection method did not converge.")

max_scale = 1.0 #because i cant gain stag pressure 
#initial values at scale 1.0
Cf = 0.005 #using the mean value of the prior for Cf for this initial solve.
maxScale_res, maxScale_mdotChoke, maxScale_results, maxScale_ok, maxScale_Predicted_stagPressure, maxScale_FinalStagPressure = safe_chokedResiduals(max_scale, n_solves, Cf)
scales = [max_scale]
residuals = [maxScale_res]


if not maxScale_ok:
    print("Max scale is bad. Pick a safer max_scale.")

else:
    print("Initial scale,", max_scale, "is good.")
    print("mdotChoke at max scale:", maxScale_mdotChoke, "imposed_Mdot: ", imposed_Mdot)

    for i in range(20):

        scale_guess = max_scale * (1  - i/100) #just a quick way to find a scale for which i have an actual brackeet for bisection method
        scale_guess = min(max(scale_guess, 0.01), max_scale) #making sure the guess is between 0.01 and max_scale
        guess_res, guess_mdotChoke, guess_results, guess_ok, guess_Predicted_stagPressure, guess_FinalStagPressure = safe_chokedResiduals(scale_guess, n_solves + 1, Cf) 

        print("guess_mdotChoke: ", guess_mdotChoke, "imposed_Mdot: ", imposed_Mdot)
        print("guess_Predicted_stagPressure: ", guess_Predicted_stagPressure, "FinalStagPressure: ", guess_FinalStagPressure)

        if not guess_ok:
            print("Scale guess is bad.",scale_guess)
        else:
            scales.append(scale_guess)
            residuals.append(guess_res)
            print("Scale guess is good", scale_guess)
            
            if maxScale_res * guess_res < 0:
                scale_low = min(max_scale, scale_guess)
                scale_high = max(max_scale, scale_guess)
                print("Bracket found:", scale_low, scale_high)
                
                scale, residual, history = hybrid_newton_bisection(scale_low, scale_high)
                break
                
            else:
                print("No bracket yet; now do a smaller search around scale_guess")

scales_List = np.array(scales)
residuals_List = np.array(residuals)
plt.plot(scales_List, residuals_List, 'o-')
plt.gca().invert_xaxis()
plt.ticklabel_format(style='plain', axis='x')
plt.axhline(0, linestyle='--')
plt.xlabel("Scale")
plt.ylabel("Residual (imposed mdot - mdotChoke)")
plt.title("Residual vs Scale")
plt.grid()
plt.show()
'''

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
'''
