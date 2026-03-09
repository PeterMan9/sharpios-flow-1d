
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import fsolve
import cantera as ct





#preburner conditions 
mdotpreburner = 1.2 #kg/s

R = 296.8 #J/kgK
L = 0.5 #m 
f_darcy = 0.01
Cf = f_darcy/4 #skin friction factor
gamma0 = 1.4

numSteps = 100000
dx = L/numSteps
xList = np.arange(0, L + dx, dx)
r0 = 0.0127 #m
D0 = 2 * r0 #m
A0 = np.pi * r0**2 #m^2
Astar = A0/25 #m^2
rStar = np.sqrt(Astar/np.pi) #m

def Area(x):
    if x <= 0.75 * L:
        return A0
    else:
        rX = r0 - ( (r0 - rStar)/L ) * 0.1*x
        A = np.pi * rX**2
        return A

def Dh(x):
    if x <= 0.75 * L:
        return D0
    else:
        rX = r0 - ( (r0 - rStar)/L ) * 0.1*x
        D_h = 2 * rX
    return D_h

def mNum(v,a): #mach number 
    M = v/a
    return M

def soS(T): #solving for a using variable gamma and Cp 
    gammA = (CpNasa(T)/(CpNasa(T) - R))
    a = np.sqrt(gammA * R * T)
    return a

#dhtdx profile
qtotal = 200e3
x_s = 0
def dHtdx(x):#dht/dx profile - quadtratic Ht
    return (qtotal/L) * (x - x_s)

def CpNasa(T): #solving variable Cp with NASA polynomials for N2 
    return (0.02926640*10**2 + 0.14879768E-02 * T - 0.05684760E-05 * T**2 + 0.10097038E-09 * T**3 - 0.06753351E-13 * T**4) * R

def gamma(T):#solving for gamma using 
    SHC = CpNasa(T) / (CpNasa(T) - R)
    return SHC
     
def hTNasa(T): #solving for static enthalpy using NASA polynomials for N2
    return (0.02926640*10**2 + (0.14879768E-02 * T)/2 - (0.05684760E-05 * T**2)/3 + (0.10097038E-09 * T**3)/4 - (0.06753351E-13 * T**4)/5) * R * T

def delAdx (An1,An,x1,x): #dA/dx function
    return (An1 - An)/(x1-x)

def delMdotdx(mdotn1, mdotn,x1,x): #dmdot/dx function
    return (mdotn1 - mdotn)/(x1-x)

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

def pressureStagEntropyFunc(pstag1,entropy1,entropy2): #wrong for now, need to have it an equation that works with changing Tstag etc
    pstag2 = pstag1 * np.exp(-(entropy2 - entropy1)/R)
    return pstag2

################################################

##########################3
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
A1 = 7.917E-6 #m^2
A2 = 7.917E-6 #m^2
A3 = 3 * A1 #m^2
A_A = A1 * 2 #m^2


P1 = 6*1e6 #Pa
P2 = 6*1e6 #Pa
P3 = 2*1e6 #Pa
PA = P1 #Pa. Pa is equal to P1 and P2 because they are the same injector and they are connected to the same plenum.


TA = 300 #K
TA_2 = 300 #K
TB = 300 #K

M1 = 0.5
M2 = 0.5
M3 = 0.44

a1 = soS(TA)
a2 = soS(TA_2)
a3 = soS(TB)

uA = M1 * a1
uA_2 = M2 * a2
uB = M3 * a3



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

def E1_CV(ui,Ti):
    return ui - (mdotA/mdot_i) * uA - (mdotB/mdot_i) * uB - (mdotA * R * TA)/(mdot_i * uA) - (mdotB * R * TB)/(mdot_i * uB) + (R * Ti)/ui
def E2_CV(ui,Ti):
    hi = hTNasa(Ti)
    hA = hTNasa(TA)
    hB = hTNasa(TB)
    return (hi + ui**2/2) - (mdotA/mdot_i) * (hA + uA**2/2) - (mdotB/mdot_i) * (hB + uB**2/2)

def CV_toPreburner(u2,T2): #this is newton raphson for the CV it goes from state 1 (once gasses have mixed) to state 2 (preburner inlet) 
    numIters = 0        #cut down the system of equations to 2 equations and 2 unkowns so just solving till im under tolorence 
    tol = 1e-8

    E1 = E1_CV(u2,T2)
    E2 = E2_CV(u2,T2)
    E_vec = np.array([E1, E2])

    print("_______________________________________________")
    print("CV to Preburner ITERATION VALUES")

    while(np.linalg.norm(E_vec, 2) >= tol):
        #numerical jacobian 


        deltaU = u2/1e8  #the delta or perturbation will be updating as u2 and T2 update to make sure its not too big or too small.
        deltaT = T2/1e8


        dE1du = (E1_CV(u2 + deltaU, T2) - E1)/deltaU
        dE1dT = (E1_CV(u2, T2 + deltaT) - E1)/deltaT
        dE2du = (E2_CV(u2 + deltaU, T2) - E2)/deltaU
        dE2dT = (E2_CV(u2, T2 + deltaT) - E2)/deltaT

        J = np.array([[dE1du, dE1dT], [dE2du, dE2dT]])
        
        deltas = np.linalg.solve(J, -E_vec)
        
        u2 += deltas[0]
        T2 += deltas[1]

        E1 = E1_CV(u2,T2)   #updating E1 and E2 values after updating u2 and T2 to check for convergence and to move
        E2 = E2_CV(u2,T2)   #the method forward
        E_vec = np.array([E1, E2])   

        print("velocity = ", u2, " m/s, Temperature = ", T2, " K")
        print("E1 = ", E1, " E2 = ", E2)   

        numIters += 1 #just counting num of iterations 

    print("Converged in ", numIters, " iterations")
    return u2, T2


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
    k1V = dVdX(V1,A1,M1,cp1,T1,delAdx(Area(x),Area(x-dx),x1,x1-dx),dHtdx1,mdot[i],delMdotdx(mdot[i],mdot[i-1],x1,x1-dx),Dh(x))
    k1P = dPdX(P1,V1,A1,M1,cp1,T1,delAdx(Area(x),Area(x-dx),x1,x1-dx),dHtdx1,mdot[i],delMdotdx(mdot[i],mdot[i-1],x1,x1-dx),Dh(x))

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
    k2V = dVdX(V2,A2,M2,cp2,T2,delAdx(Area(x2),Area(x1),x2,x1),dHtdx2,mdot[i],delMdotdx(mdot[i],mdot[i-1],x2,x1),Dh(x2))
    k2P = dPdX(P2,V2,A2,M2,cp2,T2,delAdx(Area(x2),Area(x1),x2,x1),dHtdx2,mdot[i],delMdotdx(mdot[i],mdot[i-1],x2,x1),Dh(x2))

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
    k3V = dVdX(V3,A3,M3,cp3,T3,delAdx(Area(x3),Area(x1),x3,x1),dHtdx3,mdot[i],delMdotdx(mdot[i],mdot[i-1],x3,x1),Dh(x3))
    k3P = dPdX(P3,V3,A3,M3,cp3,T3,delAdx(Area(x3),Area(x1),x3,x1),dHtdx3,mdot[i],delMdotdx(mdot[i],mdot[i-1],x3,x1),Dh(x3))  

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
    k4V = dVdX(V4,A4,M4,cp4,T4,delAdx(Area(x4),Area(x1),x4,x1),dHtdx4,mdot[i],delMdotdx(mdot[i],mdot[i-1],x4,x1),Dh(x4))
    k4P = dPdX(P4,V4,A4,M4,cp4,T4,delAdx(Area(x4),Area(x1),x4,x1),dHtdx4,mdot[i],delMdotdx(mdot[i],mdot[i-1],x4,x1),Dh(x4))

    Vnext = V + (1/6) * (k1V + 2*k2V + 2*k3V + k4V) * dx
    Pnext = P + (1/6) * (k1P + 2*k2P + 2*k3P + k4P) * dx
    return Vnext, Pnext

#########################################################################################

def P_rho_InitialValues(ui,Ti): #finding the rest of the initial values for the preburner inlet
    rho = mdot_i/(A_CV_END * ui)
    P = rho * R * Ti
    return P, rho

#making an inital guess for newton raphson and then solving for initial values to throw into preburner code
u_Guess_MixingCV = (uA * A_A + uB * A3)/A_CV_END
T_Guess_MixingCV = (TA * A_A + TB * A3)/A_CV_END

#calculating values at preburner inlet
u_Preburner, T_Preburner = CV_toPreburner(u_Guess_MixingCV, T_Guess_MixingCV)
P_preburner, rho_Preburner = P_rho_InitialValues(u_Preburner, T_Preburner)

M_Preburner_Inlet = u_Preburner/np.sqrt(gamma(T_Preburner) * R * T_Preburner) #mach number at state 1 of injector to preburner CV
Pstag_Preburner_Inlet = P_preburner * (1 + (gamma(T_Preburner) - 1)/2 * M_Preburner_Inlet**2)**(gamma(T_Preburner)/(gamma(T_Preburner)-1)) #stagnation pressure at inlet
print("________________________________________________")
print("Preburner Inlet Conditions")
print("Velocity = ", u_Preburner, " m/s")
print("Temperature = ", T_Preburner, " K")
print("Pressure = ", P_preburner * 1e-6, " MPa")
print("Mach Number = ", M_Preburner_Inlet)

#arrays to store values
temp = []
temp.append(T_Preburner)

velocities = []
velocities.append(u_Preburner)

pressure = []
pressure.append(P_preburner)

machNum = []
machNum.append(M_Preburner_Inlet)

density = []
density.append(rho_Preburner)

pressureStag = []
pressureStag.append(Pstag_Preburner_Inlet)

pressureStag_entropy = []
pressureStag_entropy.append(Pstag_Preburner_Inlet)


mdotReconsturcted = []
mdotReconsturcted.append(mdot_i)

entropy = []

gas = ct.Solution('gri30.yaml')
gas.TPX = T_Preburner, P_preburner, {'N2': 1.0}
sInitial = gas.entropy_mass
entropy.append(sInitial)

#########################################################
mdot = np.full(len(xList), mdot_i)

#injector
Vinj = 100 # m/s speed of N2 being injected (alr converted to x direction)
Dinj = 0.01 #m Injector diameter
Ainj = np.pi * (Dinj/2)**2 #m^2 
injMdot = rho_Preburner * Vinj * Ainj #kg/s

x_injLocation = 0.05 #m
injIndex = int(x_injLocation/dx) #index of the center of the injector 
injIndexRange = int((Dinj/2)/dx) #range is +- so this is only half of total inj diameter
inj_array = np.zeros(len(xList)) #array to hold injector locations (0 means no injector 1 means injector, 2 means post injector)

startInj = max(0, int(injIndex - injIndexRange)) #start index of injector
endInj = min(len(xList)-1, int(injIndex + injIndexRange)) #end index of injector
inj_array[startInj:endInj+1] = 1 #mark injector location, +1 to include end index
inj_array[endInj+1:] = 2    #mark post injector locations +1 to start 1 after that end index

mdot[startInj:endInj+1] = mdot_i + injMdot #add injector mass flow to main flow at injector location
mdot[endInj+1:] = mdot_i + injMdot #post injector mass flow



############################################################################################################################

for i in range(1, len(xList)): #actual for loop for solving everything. only updating value vectors (velocity, pressure, rho etc)  
    xCurrent = xList[i]        # after everything in current state is solved to avoid mess ups
    localAreaCurrent = Area(xCurrent)

    Vbefore = velocities[i-1]
    Pbefore = pressure[i-1]
    VCurrent, PCurrent = rk4Step(Vbefore,Pbefore, i)
    mdotlocal = mdot[i] #using local/current mdot to get rho to get T and etc
    rhoCurrent = mdotlocal/(Area(xList[i]) * VCurrent) 
    mdotReconsturcted.append(rhoCurrent * VCurrent * localAreaCurrent)
    velocities.append(VCurrent)
    pressure.append(PCurrent)
    TCurrent = PCurrent/(rhoCurrent * R)
    density.append(rhoCurrent)
    temp.append(TCurrent)
    aCurrent = soS(TCurrent)
    MCurrent = mNum(VCurrent,aCurrent)
    machNum.append(MCurrent)
    pressureStag.append(pressureStagFunc(PCurrent,MCurrent,TCurrent))
    gas.TP = TCurrent, PCurrent
    sCurrent = gas.entropy_mass
    entropy.append(sCurrent)
    pressureStag_entropy.append(pressureStagEntropyFunc(pressureStag_entropy[i-1],entropy[i-1],entropy[i]))

    if machNum[i] >= 1:
        print("Flow is choked at x = ", xCurrent)
        break


#converting list to numpy arrays for plt
V_List = np.array(velocities)
P_List = np.array(pressure)
T_List = np.array(temp)
rho_List = np.array(density)
M_List = np.array(machNum)
pStag_List = np.array(pressureStag)
pStagEntropyFunc_List = np.array(pressureStag_entropy)

x_List = np.array(xList[:len(V_List)])
mdot_List = np.array(mdot)
mdotReconsturcted_List = np.array(mdotReconsturcted)
entropy_List = np.array(entropy)


#plotting results 
'''







plt.figure() #Plotting mdot residuals as a way to debug mdot and make sure my V, rho etc calc correctly
plt.plot(x_List,mdotReconsturcted_List[:len(x_List)] - mdot_List[:len(x_List)])
plt.xlabel('x (m)')
plt.ylabel('Mdot (kg/s)')
plt.title('Plotting Residuals for mDot')
plt.grid()
'''
plt.figure() #plotting entropy. All looks good but seems to drop right at the choking point (mach one)
plt.plot(x_List, entropy_List) #i am assuming this is just because the equations start to explode and results in temp being off and there for lowering entropy
plt.xlabel('x (m)')
plt.ylabel('Entropy (J/(kg * K))')
plt.title('Plotting Entropy with Cantera')
plt.grid()


plt.figure()
plt.plot(x_List, pStag_List)
plt.xlabel('x (m)')
plt.ylabel('Pressure Stagnation (Pa)')
plt.title('Plotting Pressure Stagnation')
plt.grid()


plt.show()




