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

mdot = None
Vinj = None 

numSteps = int(2e5)
dx = L_total / numSteps
xList = np.arange(0, L_total + dx, dx)

r0 = 0.05 #m
D0 = 2 * r0
A0 = np.pi * r0**2
Astar = A0/25
rStar = np.sqrt(Astar/np.pi)
r_exit = r0

x1_conv = 0.5 * L_tochoke #defining range where the area starts to converge 
x2_conv = L_tochoke #where convergence ends and area is minimum

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

fullAreaList = [Area(x) for x in xList]
fullAreaRatioList = [A0/A for A in fullAreaList]
fullArea_List = np.array(fullAreaList)
fullAreaRatio_List = np.array(fullAreaRatioList)
fullDadxList = [dAdx(x) for x in xList]
fullDadx_List = np.array(fullDadxList)


def mNum(v,a): #mach number 
    M = v/a
    return M

def soS(T): #solving for a using variable gamma and Cp 
    a = np.sqrt(gamma(T) * R * T)
    return a


def delMdotdx(mdotn1, mdotn,x1,x): #dmdot/dx function
    return (mdotn1 - mdotn)/(x1-x)


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

# mixing CV inital conditions and mixing CV to preburner inlet functions 


#Defining cv for mixing 
#3 injectors - 2 of the same and one bigger one. 
#pressure, temp, Area, Mach number are the all known for the injectors 
#give somem kinda cv length 
#area will stay the same 
#known for outlet - area preburner, mass flow, 
# need to solve for mach number, temp, pressure and then plug them into my current code 
# start with isentropic for solving from the injector to the breburner. 

#defining initial values for injectors 
#legend
    #1 - small injector 1 
    #2 - small injector 2
    #3 - big injector 3
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

print("uA: ", uA, "uB: ", uB)

TstagA = TA * (1 + (gamma(TA) - 1)/2 * M1**2)
TstagB = TB * (1 + (gamma(TB) - 1)/2 * M3**2)

PstagA = PA * (1 + (gamma(TA) - 1)/2 * M1**2)**(gamma(TA)/(gamma(TA)-1))
PstagB = PB * (1 + (gamma(TB) - 1)/2 * M3**2)**(gamma(TB)/(gamma(TB)-1))

rho1 = P1/(R*TA) #will have to define R for diff species etc - but for now they are all the same
rho2 = P2/(R*TA_2)
rho3 = P3/(R*TB)

mdot1 = 2
mdot2 = 2
mdot3 = 2


mdotA = mdot1 + mdot2 #injector 1 and 2 are the same so can just add them together
mdotB = mdot3 #big injector
mdot_i = mdotA + mdotB

#print("mdotA: ", mdotA, "mdotB: ", mdotB, "mdot_i: ", mdot_i)
#print("uA: ", uA, "uB: ", uB)
A_CV_END = A0 #area at the end of the CV is the same as the area at the start of the preburner inlet.


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
    return rhoA * uA_2 * A_A - mdotA

def E4_InjB_CV(PstagB_2, uB_2, TB_2): #third cv equation check power point for indepth breakdown
    part1 = (PstagB_2/(R * TstagB))
    part2 = (1 + ((gamma(TB_2) - 1)/2) * ((uB_2/soS(TB_2))**2))
    part3 = (1 - 1*(gamma(TB_2)/(gamma(TB_2)-1)))
    rhoB = part1 * part2**part3
    return rhoB * uB_2 * A3 - mdotB

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

#rk4 function
def rk4Step(V,P,Cf, i): 
    #rk4 dvdx is coupled with dhtdx so have to solve both odes - no need to rk4 Ht and etc tho cuz i know dhtdx profile
    # doing rk4 only on V and P. can solve using two options. the easy way which is through mdot and ideal gas law other 
    #other way is to get static enthlapy (integrate dhtdx etc) and set equal to static enthlapy from nasa polynomials, 
    #and then use fsolve to get Temp (this is harder and more "error" due to fsolve also solving intererativaly)
    x = xList[i] 

    #k1
    dHtdx1 = dHtdx(x)
    V1 = V
    P1 = P
    x1 = x
    A1 = Area(x1)
    rho1 = mdot[i]/(A1 * V1)
    T1 = P1/(rho1 * R)
    cp1 = CpNasa(T1)
    a1 = soS(T1)
    M1 = mNum(V1,a1)
    k1V = dVdX(V1,A1,M1,cp1,T1,dAdx(x1),dHtdx1,mdot[i],delMdotdx(mdot[i],mdot[i-1],x1,x1-dx),Dh(x), Cf)
    k1P = dPdX(P1,V1,A1,M1,cp1,T1,dAdx(x1),dHtdx1,mdot[i],delMdotdx(mdot[i],mdot[i-1],x1,x1-dx),Dh(x), Cf)

    #k2
    V2 = V + (k1V * dx/2)
    P2 = P + (k1P * dx/2)
    x2 = x + dx/2
    dHtdx2 = dHtdx(x2)
    rho2 = mdot[i]/(Area(x2) * V2)
    T2 = P2/(rho2 * R)
    cp2 = CpNasa(T2)
    a2 = soS(T2)
    M2 = mNum(V2,a2)
    A2 = Area(x2)
    k2V = dVdX(V2,A2,M2,cp2,T2,dAdx(x2),dHtdx2,mdot[i],delMdotdx(mdot[i],mdot[i-1],x2,x1),Dh(x2), Cf)
    k2P = dPdX(P2,V2,A2,M2,cp2,T2,dAdx(x2),dHtdx2,mdot[i],delMdotdx(mdot[i],mdot[i-1],x2,x1),Dh(x2), Cf)

    #k3
    V3 = V + (k2V * dx/2)
    P3 = P + (k2P * dx/2)
    x3 = x + dx/2
    dHtdx3 = dHtdx(x3)
    rho3 = mdot[i]/(Area(x3) * V3)
    T3 = P3/(rho3 * R)
    cp3 = CpNasa(T3)
    a3 = soS(T3)
    M3 = mNum(V3,a3)
    A3 = Area(x3)
    k3V = dVdX(V3,A3,M3,cp3,T3,dAdx(x3),dHtdx3,mdot[i],delMdotdx(mdot[i],mdot[i-1],x3,x1),Dh(x3), Cf)
    k3P = dPdX(P3,V3,A3,M3,cp3,T3,dAdx(x3),dHtdx3,mdot[i],delMdotdx(mdot[i],mdot[i-1],x3,x1),Dh(x3), Cf)

    #k4
    V4 = V + (k3V * dx)
    P4 = P + (k3P * dx)
    x4 = x + dx
    dHtdx4 = dHtdx(x4)
    rho4 = mdot[i]/(Area(x4) * V4)
    T4 = P4/(rho4 * R)
    cp4 = CpNasa(T4)
    a4 = soS(T4)
    M4 = mNum(V4,a4)
    A4 = Area(x4)
    k4V = dVdX(V4,A4,M4,cp4,T4,dAdx(x4),dHtdx4,mdot[i],delMdotdx(mdot[i],mdot[i-1],x4,x1),Dh(x4), Cf)
    k4P = dPdX(P4,V4,A4,M4,cp4,T4,dAdx(x4),dHtdx4,mdot[i],delMdotdx(mdot[i],mdot[i-1],x4,x1),Dh(x4), Cf)

    Vnext = V + (1/6) * (k1V + 2*k2V + 2*k3V + k4V) * dx
    Pnext = P + (1/6) * (k1P + 2*k2P + 2*k3P + k4P) * dx
    return Vnext, Pnext


# Solving/Defining inital conditons for the preburner inlet - initializing arrays to store values etc

def P_rho_InitialValues(ui,Ti): #finding the rest of the initial values for the preburner inlet
    rho = mdot_i/(A_CV_END * ui)
    return rho

#consecutive solves

def consecutive_solves(pstagA_2,pstagB_2, u2_guess_A, T2_guess_A, u2_guess_B, T2_guess_B, numSolves, Cf):
    global mdot,Vinj #making them global so that i can use them in rk4 and ode functions
    
    u_InjA_2, T_InjA_2 = InjA_Loss_CV(pstagA_2, u2_guess_A, T2_guess_A)
    u_InjB_2, T_InjB_2 = InjB_Loss_CV(pstagB_2, u2_guess_B, T2_guess_B)


    u_preburner_guess = (u_InjA_2 * A_A + u_InjB_2 * A3)/(A_CV_END)
    T_preburner_guess = (T_InjA_2 * mdotA + T_InjB_2 * mdotB)/(mdot_i)

    u_preburner, T_preburner = CV_toPreburner(u_preburner_guess, T_preburner_guess, u_InjA_2, u_InjB_2, T_InjA_2, T_InjB_2)
    rho_preburner = P_rho_InitialValues(u_preburner, T_preburner)
    M_Preburner_Inlet = u_preburner/np.sqrt(gamma(T_preburner) * R * T_preburner) 
    Tstag_Preburner = temperatureStagFunc(T_preburner, M_Preburner_Inlet)

    Pstag_Preburner = 12*pstagA_2


    P_preburner = (1 + (gamma(T_preburner) - 1)/2 * M_Preburner_Inlet**2)**(-gamma(T_preburner)/(gamma(T_preburner)-1)) * Pstag_Preburner
    mdot_preburner = rho_preburner * u_preburner * A_CV_END


    
    #print("Preburner Mach Number:", M_Preburner_Inlet)
    #print("Preburner Pressure:", P_preburner * 1e-6, "MPa")
   # print("Preburner Temperature:", T_preburner)
    #print("Preburner velocity:", u_preburner)
   
 
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

    gas = ct.Solution('gri30.yaml')
    gas.TPX = T_preburner, P_preburner, {'N2': 1.0}
    sInitial = gas.entropy_mass
    entropy = [sInitial]

    mdot = np.full(len(xList), mdot_i)
    mdotReconsturcted = [] #recontruction array to check if calcs are correct 
    mdotReconsturcted.append(mdot_i) 

   # Creating Injector Array and Adding Mass Flow from Injector to global mdot array 
    Vinj = 0 # m/s speed of N2 being injected (alr converted to x direction)
    Dinj = 0.003175 #m Injector diameter
    Ainj = np.pi * (Dinj/2)**2 #m^2 
    injMdot = rho_preburner * Vinj * Ainj #kg/s

    x_injLocation = 0.15 * L_tochoke #m
    injIndex = int(x_injLocation/dx) #index of the center of the injector 
    injIndexRange = int((Dinj/2)/dx) #range is +- so this is only half of total inj diameter
    inj_array = np.zeros(len(xList)) #array to hold injector locations (0 means no injector 1 means injector, 2 means post injector)

    startInj = max(0, int(injIndex - injIndexRange)) #start index of injector
    endInj = min(len(xList)-1, int(injIndex + injIndexRange)) #end index of injector
    inj_array[startInj:endInj+1] = 1 #mark injector location, +1 to include end index
    inj_array[endInj+1:] = 2    #mark post injector locations +1 to start 1 after that end index #going to go over wtf this means 

    mdot[startInj:endInj+1] = mdot_i + injMdot #add injector mass flow to main flow at injector location
    mdot[endInj+1:] = mdot_i + injMdot #post injector mass flow

        #solving flow through Preburner
    for i in range(1, len(xList)): #actual for loop for solving everything. 
        xCurrent = xList[i]      
        mdotlocal = mdot[i]

        localAreaCurrent = Area(xCurrent)
        areaList.append(localAreaCurrent)

        areaRatio.append(Area(0)/localAreaCurrent)

        dAdXCurrent = dAdx(xCurrent)
        dAdxList.append(dAdXCurrent)

        Vbefore = velocities[i-1]
        Pbefore = pressure[i-1]
        VCurrent, PCurrent = rk4Step(Vbefore,Pbefore, Cf,i)

        velocities.append(VCurrent)
        pressure.append(PCurrent)

        rhoCurrent = mdotlocal/(Area(xList[i]) * VCurrent) 
        density.append(rhoCurrent)

        TCurrent = PCurrent/(rhoCurrent * R)
        temp.append(TCurrent)

         #using local/current mdot to get rho to get T and etc
        mdotReconsturcted.append(rhoCurrent * VCurrent * localAreaCurrent)

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

    mdot_List = np.array(mdot[:len(x_used_List)])
    mdotReconsturcted_List = np.array(mdotReconsturcted)
    entropy_List = np.array(entropy)

    #if M_List[-1] < 0.98:
        #raise RuntimeError(f"Flow did not choke. Final Mach = {M_List[-1]}")
    
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


plt.figure()
plt.plot(results["x"],results["mach_number"])
plt.xlabel("X (m)")
plt.ylabel("Mach Num ")
plt.grid()
plt.show()

print("---------------------")
results_Scale = consecutive_solves(PstagA*0.9, PstagB*0.9, uA, TA, uB, TB, n_solves, 0.005)

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
'''

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



