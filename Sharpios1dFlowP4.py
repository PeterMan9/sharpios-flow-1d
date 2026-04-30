
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import fsolve
import cantera as ct

#preburner inlet conditions 
R = 296.8 #J/kgK
L_tochoke = 0.5 #m 
L_total = 0.7   #m
f_darcy = 0.01
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

def dVdX (V,A,M,cp,T,dAdX,localdHtdx,mdot,DMDOTDX,Dh): #first 4 parts of sharpios 1d flow eqn converted to dV/dx
    gammA = gamma(T)
    term1 = ((-V)/(A * (1 - M**2)))* dAdX
    term2 = ((V/((1-M**2) * cp * T)) * localdHtdx)
    term3 = ((gammA *M**2)/(2 * (1 - M**2)))
    term4 = ((((4 * Cf * V)/Dh)) - (2*(Vinj/mdot) * DMDOTDX))
    term5 = (((V*(1 + gammA * M**2))/((1-M**2)*mdot)) * (DMDOTDX))
    return term1 + term2 + (term3*term4) + term5

def dPdX (P,V,A,M,cp,T,DADX,localdHtdx,mdot,DMDOTDX,Dh): #first 4 parts of sharpios 1d flow eqn converted to dP/dx
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
d1 = 0.012 #in
d2 = d1
d3 = 0.03

r1 = d1 / 2
r2 = r1
r3 = d3 / 2

A1 = (r1/39.37)**2 * np.pi #m^2
A2 = (r2/39.37)**2 * np.pi #m^2
A3 = (r3/39.37)**2 * np.pi #m^2
A_A = A1 + A2 #m^2


P1 = 14*1e6 #Pa
P2 = 14*1e6 #Pa
P3 = 14*1e6 #Pa
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

mdot1 = rho1 * uA * A1
mdot2 = rho2 * uA_2 * A2
mdot3 = rho3 * uB * A3

mdotA = mdot1 + mdot2 #injector 1 and 2 are the same so can just add them together
mdotB = mdot3 #big injector
mdot_i = mdotA + mdotB

A_CV_END = A0 #area at the end of the CV is the same as the area at the start of the preburner inlet.


def E1_CV(ui,Ti,uA,uB,TA,TB):
    return ui - (mdotA/mdot_i) * uA - (mdotB/mdot_i) * uB - (mdotA * R * TA)/(mdot_i * uA) - (mdotB * R * TB)/(mdot_i * uB) + (R * Ti)/ui

def E2_CV(ui,Ti,uA,uB,TA,TB):
    hi = hTNasa(Ti)
    hA = hTNasa(TA)
    hB = hTNasa(TB)
    return (hi + ui**2/2) - (mdotA/mdot_i) * (hA + uA**2/2) - (mdotB/mdot_i) * (hB + uB**2/2)

def E3_InjA_CV(PstagA_2, uA_2, TA_2): #third cv equation check power point for indepth breakdown
    part1 = PstagA_2/(R * TstagA)
    part2 = 1 + (gamma(TA_2) - 1)/2 * ((uA_2/soS(TA_2))**2)
    part3 = (1 - 1*(gamma(TA_2)/(gamma(TA_2)-1)))
    return part1 * part2**part3 - uA_2 * A_A - mdotA

def E4_InjB_CV(PstagB_2, uB_2, TB_2): #third cv equation check power point for indepth breakdown
    part1 = PstagB_2/(R * TstagB)
    part2 = 1 + (gamma(TB_2) - 1)/2 * ((uB_2/soS(TB_2))**2)
    part3 = (1 - 1*(gamma(TB_2)/(gamma(TB_2)-1)))
    return part1 * part2**part3 - uB_2 * A3 - mdotB

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

    while(np.linalg.norm(E_vec, 2) >= tol):
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

        #print("velocity = ", u2, " m/s, Temperature = ", T2, " K")
        #print("E1 = ", E1, " E2 = ", E2)   

        numIters += 1 #just counting num of iterations 

    #print("Converged in ", numIters, " iterations")
    #print("________________________________________________\n")
    return u2, T2

def InjA_Loss_CV(Pstag_A2,uA_2, TA_2):
    numIters = 0
    tol = 1e-8


    E3 = E3_InjA_CV(Pstag_A2,uA_2, TA_2)
    E5 = E5_InjA_CV(TA_2,uA_2)
    E_vec = np.array([E5, E3])

    while(np.linalg.norm(E_vec, 2) >= tol):
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
       # print("velocity = ", u2, " m/s, Temperature = ", T2, " K")
        # print("E5 = ", E5, " E3 = ", E3) 

    #print("Converged in ", numIters, " iterations")
    #print("________________________________________________\n")
    return uA_2, TA_2

def InjB_Loss_CV(Pstag_B2,uB_2, TB_2):
    numIters = 0
    tol = 1e-8


    E4 = E4_InjB_CV(Pstag_B2,uB_2, TB_2)
    E6 = E6_InjB_CV(TB_2,uB_2)
    E_vec = np.array([E6, E4])

    while(np.linalg.norm(E_vec, 2) >= tol):
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
       # print("velocity = ", u2, " m/s, Temperature = ", T2, " K")
        # print("E6 = ", E6, " E4 = ", E4) 

    #print("Converged in ", numIters, " iterations")
    #print("________________________________________________\n")
    return uB_2, TB_2

#rk4 function
def rk4Step(V,P, i): 
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
    k1V = dVdX(V1,A1,M1,cp1,T1,dAdx(x1),dHtdx1,mdot[i],delMdotdx(mdot[i],mdot[i-1],x1,x1-dx),Dh(x))
    k1P = dPdX(P1,V1,A1,M1,cp1,T1,dAdx(x1),dHtdx1,mdot[i],delMdotdx(mdot[i],mdot[i-1],x1,x1-dx),Dh(x))

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
    k2V = dVdX(V2,A2,M2,cp2,T2,dAdx(x2),dHtdx2,mdot[i],delMdotdx(mdot[i],mdot[i-1],x2,x1),Dh(x2))
    k2P = dPdX(P2,V2,A2,M2,cp2,T2,dAdx(x2),dHtdx2,mdot[i],delMdotdx(mdot[i],mdot[i-1],x2,x1),Dh(x2))

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
    k3V = dVdX(V3,A3,M3,cp3,T3,dAdx(x3),dHtdx3,mdot[i],delMdotdx(mdot[i],mdot[i-1],x3,x1),Dh(x3))
    k3P = dPdX(P3,V3,A3,M3,cp3,T3,dAdx(x3),dHtdx3,mdot[i],delMdotdx(mdot[i],mdot[i-1],x3,x1),Dh(x3))  

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
    k4V = dVdX(V4,A4,M4,cp4,T4,dAdx(x4),dHtdx4,mdot[i],delMdotdx(mdot[i],mdot[i-1],x4,x1),Dh(x4))
    k4P = dPdX(P4,V4,A4,M4,cp4,T4,dAdx(x4),dHtdx4,mdot[i],delMdotdx(mdot[i],mdot[i-1],x4,x1),Dh(x4))

    Vnext = V + (1/6) * (k1V + 2*k2V + 2*k3V + k4V) * dx
    Pnext = P + (1/6) * (k1P + 2*k2P + 2*k3P + k4P) * dx
    return Vnext, Pnext


# Solving/Defining inital conditons for the preburner inlet - initializing arrays to store values etc

def P_rho_InitialValues(ui,Ti): #finding the rest of the initial values for the preburner inlet
    rho = mdot_i/(A_CV_END * ui)
    P = rho * R * Ti
    return P, rho

#consecutive solves


print("Consecutive Solves:")
print("________________________________________________\n")

def consecutive_solves(pstagA_2,pstagB_2, u2_guess_A, T2_guess_A, u2_guess_B, T2_guess_B, numSolves):
    global mdot,Vinj #making them global so that i can use them in rk4 and ode functions
    
    u_InjA_2, T_InjA_2 = InjA_Loss_CV(pstagA_2, u2_guess_A, T2_guess_A)
    u_InjB_2, T_InjB_2 = InjB_Loss_CV(pstagB_2, u2_guess_B, T2_guess_B)

    u_preburner_guess = (u_InjA_2 * A_A + u_InjB_2 * A3)/(A_CV_END)
    T_preburner_guess = (T_InjA_2 * mdotA + T_InjB_2 * mdotB)/(mdot_i)

    u_preburner, T_preburner = CV_toPreburner(u_preburner_guess, T_preburner_guess, u_InjA_2, u_InjB_2, T_InjA_2, T_InjB_2)
    P_preburner, rho_preburner = P_rho_InitialValues(u_preburner, T_preburner)
    M_Preburner_Inlet = u_preburner/np.sqrt(gamma(T_preburner) * R * T_preburner) 

    #printing I.C. for preburner to inlet 
    
    print("Preburner Inlet Conditions from solve number:", numSolves)
    #print("Velocity = ", u_preburner, " m/s")
    #print("Temperature = ", T_preburner, " K")
    #print("Pressure = ", P_preburner * 1e-3, " kPa")
    #print("Mach Number = ", M_Preburner_Inlet)
    #print("________________________________________________") 
    
    temp = [T_preburner]                # creating fresh arrays in function 
    velocities = [u_preburner]
    pressure = [P_preburner]
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
    Vinj = 250 # m/s speed of N2 being injected (alr converted to x direction)
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

        areaRatio.append(localAreaCurrent/Area(0))

        dAdXCurrent = dAdx(xCurrent)
        dAdxList.append(dAdXCurrent)

        Vbefore = velocities[i-1]
        Pbefore = pressure[i-1]
        VCurrent, PCurrent = rk4Step(Vbefore,Pbefore, i)

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

        gas.TP = TCurrent, PCurrent
        sCurrent = gas.entropy_mass
        entropy.append(sCurrent)
        


        if MCurrent >= 1:
            print("Flow for this solve is choked at x = ", xCurrent)
            break

    #converting to np arrays
    V_List = np.array(velocities)
    P_List = np.array(pressure)
    T_List = np.array(temp)
    rho_List = np.array(density)
    M_List = np.array(machNum)
    AreaRatio_List = np.array(areaRatio)

    x_used_List = np.array(xList[:len(V_List)])
    Area_List = np.array(areaList)
    dAdx_List = np.array(dAdxList)

    mdot_List = np.array(mdot[:len(x_used_List)])
    mdotReconsturcted_List = np.array(mdotReconsturcted)
    entropy_List = np.array(entropy)


    print("Final Mach Number:", M_List[-1])
    print("Area Ratio at Choke Point:", Area_List[-1]/Area_List[0])

    return {
        "velocity": V_List,
        "pressure": P_List,
        "temperature": T_List,
        "density": rho_List,
        "mach_number": M_List,
        "x": x_used_List,
        "area": Area_List,
        "dAdx": dAdx_List,
        "mdot": mdot_List,
        "mdot_reconstructed": mdotReconsturcted_List,
        "entropy": entropy_List,
        "xChoked": xCurrent,
        "Area Ratio": AreaRatio_List    
    }


targetChoke = L_tochoke #m
tol_choke= dx
n_solves = 1 #number of solves. 

results_consecutive_solve = consecutive_solves(PstagA*0.9, PstagB*0.9, uA, TA, uB, TB, n_solves) # initial stag pressure loss guess 
xChoke = results_consecutive_solve["xChoked"] # initial guess/choke
AreaRatio = results_consecutive_solve["Area Ratio"]
xList_consecutive = results_consecutive_solve["x"]

#function to basically sweep through a bunch of diff scales to see if I can find a good bracket for bisection method

def safe_chokedResiduals(scale,numSolves):
    try: #i am using try and except to catch any errors such as cantera errors etc, and then just returning None for those cases so that the code does not crash
        results = consecutive_solves(PstagA*scale, PstagB*scale, uA, TA, uB, TB, numSolves)
        xChoke = results["xChoked"]
        residual = targetChoke - xChoke

        if not np.isfinite(xChoke):
            return None, None, None, False

        if not np.isfinite(residual):
            return None, None, None, False
        return residual, xChoke, results,True
    
    except:
        return None, None, None, False

scale_initial = 1 #starting off at 0% loss or scale 1
dscale = 0.01 #how much i am changing the scale by per iter
max_iters = 100   #max num of iter

scales_plot = [] #def array for plotting error vs scale
errors_plot = []

res_center, x_center, results_center, ok_center = safe_chokedResiduals(scale_initial, n_solves) #initial check

if not ok_center: #checking initial scale guess
    print("Initial scale is bad. Pick a safer scale_initial.")
else: #if initial scale good then creating live graph to plot error vs scale. also sweeping through scale to find good bracket
    scales_plot.append(scale_initial)
    errors_plot.append(res_center)

    bracket_found = False

    plt.ion()  # turn on interactive mode

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o-')
    ax.axhline(0, linestyle='--')

    ax.set_xlabel("Scale")
    ax.set_ylabel("Error (target - xChoke)")
    ax.set_title("Error vs Scale")
    ax.grid()

    for i in range(1, max_iters + 1):

        for direction in [-1]:

            scale_test = scale_initial + direction * i * dscale

            if scale_test <= 0:
                continue

            res_test, x_test, results_test, ok_test = safe_chokedResiduals(scale_test, n_solves + i)

            if not ok_test:
                print("scale bad:", scale_test)
                continue

            scales_plot.append(scale_test)
            errors_plot.append(res_test)
            line.set_xdata(scales_plot)
            line.set_ydata(errors_plot)

            ax.relim()          # recompute limits
            ax.autoscale_view() # rescale axes

            plt.draw()

            print("scale good:", scale_test, "xChoke:", x_test, "res:", res_test)

            if res_center * res_test < 0:
                scale_low = min(scale_initial, scale_test)
                scale_high = max(scale_initial, scale_test)
                bracket_found = True
                print("Bracket found:", scale_low, scale_high)
                break

        if bracket_found:
            break

    if not bracket_found:
        print("No bracket found.")


#setting up braket for bisection method
'''
res_low, x_low, results_low, ok_low = safe_chokedResiduals(scale_low)
res_high, x_high, results_high, ok_high = safe_chokedResiduals(scale_high)

if not ok_low:
    print("scale_low is unsafe")
if not ok_high:
    print("scale_high is unsafe")


else:
    for i in range(max_iters):
        scale_mid = (scale_low + scale_high)/2
        res_mid,x_mid,results_mid,ok_mid = safe_chokedResiduals(scale_mid)
        n_solves +=1

        if not ok_mid:
            print(f"scale_mid = {scale_mid:.3f} is unsafe")
            continue

        if res_low * res_mid < 0:
            scale_high = scale_mid
            res_high = res_mid
        else:
            scale_low = scale_mid
            res_low = res_mid

        if abs(res_mid) <= tol_choke:
            print("Converged")
            converged_results = results_mid
            converged_scale = scale_mid
            xChoke = x_mid
            break

'''    

#plotting results 
'''


plt.figure() #Plotting mdot residuals as a way to debug mdot and make sure my V, rho etc calc correctly
plt.plot(x,mdotReconsturcted_List[:len(x)] - mdot_List[:len(x)])
plt.xlabel('x (m)')
plt.ylabel('Mdot (kg/s)')
plt.title('Plotting Residuals for mDot')
plt.grid()





plt.figure()
plt.plot(x2, pressure_List_2nd_solve)
plt.xlabel('x (m)')
plt.ylabel('Pressure 2nd Solve(Pa)')
plt.title('Plotting Pressure')
plt.grid()


plt.figure()
plt.plot(x,entropy_List, label='1st Solve')
plt.plot(x2, entropy_List_2nd_solve, label='2nd Solve')
plt.xlabel('x (m)')
plt.ylabel('Entropy 2nd Solve(J/kg/K)')
plt.title('Plotting Entropy')
plt.legend()
plt.grid()

plt.figure()
plt.plot(scales_plot, errors_plot, 'o-')
plt.axhline(0, linestyle='--')   # zero line
plt.xlabel("Scale")
plt.ylabel("Error (target - xChoke)")
plt.title("Error vs Scale")
plt.grid()
plt.show()



plt.figure()
plt.plot(xList, fullArea_List, 'o-')
plt.xlabel("x")
plt.ylabel("Area")
plt.title("Area vs x")
plt.grid()

plt.figure()
plt.plot(xList, fullAreaRatio_List, 'o-')
plt.xlabel("x")
plt.ylabel("Area Ratio")
plt.title("Area Ratio vs x")
plt.grid()

plt.figure()
plt.plot(xList, fullDadxList, 'o-')
plt.xlabel("x")
plt.ylabel("dA/dx")
plt.title("dA/dx vs x")
plt.grid()

plt.show()

'''

