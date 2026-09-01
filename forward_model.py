import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from scipy import stats
from dataclasses import dataclass
from typing import Literal
from typing import Any
import traceback
import time


@dataclass(frozen=True)
class ModelConfig:
    geometry_type: Literal["wind_tunnel", "constant_area"]
    friction: bool = True
    boundary_layer: bool = True
    combustion: bool = True


@dataclass(frozen=True)
class ConstantAreaGeometry:
    #i am making the tube square
    tube_area: float

    tube_length: float
    @property
    def solver_end_location(self):
        return self.tube_length
    
    x_injLocation: float

    @property
    def x_end(self) -> float:
        return self.tube_length
    
    @property 
    def tube_height(self)-> float:
        return np.sqrt(self.tube_area)
    
    def geometry_regions(self, x: float) -> tuple[str,float,float]:
        if x <= self.solver_end_location:
            local_tol = 1e-6
            h_max = 1e-3
            return "ConstantArea_Tube", local_tol, h_max
    
    #function that allows me to input the % of tube area I will be obstruction and getting BL Layer in Return
    #goal is to give a % amount of area I the BL will take up and then make that the initial height and then apply a growth rate that grows as the flow speeds up
    def bl_height(self,percent_obstruction: float) -> float:
        unObstructed_tube_area = self.tube_area - self.tube_area * percent_obstruction
        unObstructed_tube_height = np.sqrt(unObstructed_tube_area)
        bL_Y = self.tube_height - unObstructed_tube_height
        return bL_Y
    
    #solving for area based on location and boundary layer stuff
    def geom_Area(self,x: float, bl_h: float ,bl_growth: float ) -> float:
        #normalized x
        xi = np.clip((x - 0)/self.x_end, 0, 1)
        bl_multiplier = 1.0 + xi * (bl_growth)
        residual = bl_h - bl_h * bl_multiplier
        return (self.tube_height - bl_h * bl_multiplier)**2
    
    def smallest_eff_area(self, bl_height: float, bl_growth: float) -> float:
        return self.geom_Area(self.tube_length, bl_height, bl_growth)
    
    def inlet_area(self, bl_height: float, bl_growth: float):
        return self.geom_Area(0, bl_height, bl_growth)
    
    #dAdx func. Just using FDM for this - not actually deriving a true dAdx
    def dAdx(self, x: float,bl_h: float,bl_growth: float,tol = 1e-3) -> float:
        xCurrent = x
        xPrev = x - max((x*tol),1e-9)
        dA = self.geom_Area(xCurrent,bl_h,bl_growth) - self.geom_Area(xPrev,bl_h,bl_growth)
        return dA/(xCurrent - xPrev)
    
    #hydraulic diameter 
    def Dh(self, x: float, bl_h: float = 0.0, bl_growth: float = 0.0) -> float:
        return np.sqrt(self.geom_Area(x,bl_h,bl_growth))

    #THIS IS FOR THE TUBE, I WILL JUST BE PUTTING A BLANKET CF ON IT - I WILL NOT EVEN BE SAMPLING CF MOST OF THE TIME
    def cf_location(self,x: float,Cf_sampling: float) -> float:
        return Cf_sampling    
       
@dataclass(frozen=True)
class WindTunnelGeometry:
    preburner_area: float
    @property
    def inlet_area (self):
        return self.preburner_area
     

    preburner_length: float
    nozzle_area_ratio : float
    conv_Nozzle_length: float
    div_Nozzle_length: float
    exit_Area:float

    x_injLocation: float
  
    @property
    def throat_loc(self) -> float:
        return self.preburner_length + self.conv_Nozzle_length
    @property
    def nozzle_exit(self) -> float:
        return self.throat_loc + self.div_Nozzle_length
    
    solver_end_location = nozzle_exit


    @property
    def throat_Area(self) -> float:
        return self.preburner_area/self.nozzle_area_ratio
    
    @property
    def throat_Height(self) -> float:
        return np.sqrt(self.throat_Area)
    @staticmethod
    def smoothstep(xi: float) -> float:
        return 6*xi**5 - 15*xi**4 + 10*xi**3

    #just defining regions
    def geometry_regions(self, x: float) -> tuple[str,float,float]:
                    
        if x <= self.preburner_length:
            local_tol = 1e-2
            h_max = 1e-1
            return "Preburner", local_tol, h_max
        elif self.throat_loc - 0.0005 <= x <= self.throat_loc + 0.0009: 
            local_tol = 1e-8
            h_max = 1e-4
            return "Throat", local_tol, h_max
        elif x <= self.throat_loc:
            local_tol = 1e-6
            h_max =1e-3
            return "Conv Nozzle", local_tol, h_max
        elif x <= self.nozzle_exit:
            local_tol = 1e-6
            h_max =1e-3        
            return "Div Nozzle", local_tol, h_max
        else:
            local_tol = 1e-2
            h_max = 1e-1
            return "Test Section", local_tol, h_max

    #function that allows me to input the % of throat that I will  be obstructing and then getting a boundary layer height inreturn 
    #goal is to use the bL_Y and then just apply it to the whole conv nozzle sectoin 
    def bl_height(self,percent_obstruction: float) -> float:
        unObstructed_throat_area = self.throat_Area - self.throat_Area * percent_obstruction
        unObstructed_throat_height = np.sqrt(unObstructed_throat_area)
        bL_Y = self.throat_Height - unObstructed_throat_height
        return bL_Y

    #solving for area based on location and boundary layer stuff
    #only having the boundary in the converging and diverging parts of the nozzle. 
    #only having the boundary layer growth in the diverging part because that is when the flow goes super sonic
    def geom_Area(self,x: float, bl_h: float ,bl_growth: float) -> float:
    
        if x <= self.preburner_length:
            return self.preburner_area
        
        elif x <= self.throat_loc:
            effective_throat_area = (self.throat_Height - bl_h)**2
            xi = (x - self.preburner_length)/self.conv_Nozzle_length
            return self.preburner_area + self.smoothstep(xi) * (effective_throat_area - self.preburner_area)
        elif x <= self.nozzle_exit:

            effective_throat_height = self.throat_Height - bl_h
            effective_exit_height = np.sqrt(self.exit_Area) - (bl_growth * bl_h)
            
            effective_throat_area = effective_throat_height**2
            effective_exit_area = effective_exit_height**2

            xi = (x - self.throat_loc)/(self.div_Nozzle_length)
            return effective_throat_area + self.smoothstep(xi) * (effective_exit_area - effective_throat_area)
        else:
            effective_exit_height = np.sqrt(self.exit_Area) - (bl_growth * bl_h)
            effective_exit_area = effective_exit_height**2
            return effective_exit_area

    def throat_area(self,bl_h,bl_g):
        return self.geom_Area(self.throat_loc,bl_h,bl_g)
    
    #hydraulic diameter 
    def Dh(self, x: float, bl_h: float = 0.0, bl_growth: float = 0.0) -> float:
        return np.sqrt(self.geom_Area(x,bl_h,bl_growth))

    #dAdx func. Just using FDM for this - not actually deriving a true dAdx
    def dAdx(self, x: float,bl_h: float,bl_growth: float,tol = 1e-3) -> float:
        region = self.geometry_regions(x)

        if region == "Preburner" or region == "Test Section":
            return 0.0
        elif region == "Throat" or region == "Conv Nozzle" or region == "Div Nozzle":
            xCurrent = x
            xPrev = x - max((x*tol),1e-9)
            dA = self.geom_Area(xCurrent,bl_h,bl_growth) - self.geom_Area(xPrev,bl_h,bl_growth)
            return dA/(xCurrent - xPrev)
        
    #just splitting up geometry into sections that have diff Cfs 
    #FOR THIS SPECIFICALLY I AM SAMPLING THE CF IN THE DIV Nozzle AND TEST SECTIONS 
    def cf_location(self,x: float,Cf_sampling: float) -> float:
        region = self.geometry_regions(x)
        Cf_preburner = 0.0025
        Cf_cNz = 0.003
        if region == "Preburner":
            return Cf_preburner
        
        elif region == "Throat" or region == "Conv Nozzle":
            return Cf_cNz
        elif region == "Div Nozzle" or region == "Test Section":
            return Cf_sampling    

#function to define pressure tap locations 
#because of the way my rk45 works right now i basically check if i have crossed the location of a PT and  then use interpolation to get approx values 
#for the pt location and the pressure values at that location
def pressureTap(x_old: float, p_old: float, x_new: float, p_new: float, PT_locations: list[float]) -> tuple[float | None , float| None]:
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

#mach num
def mNum(v:float,a:float) -> float: #mach number 
    M = v/a
    return M
#speed of sound
def soS(T:float, R:float,gamma:float) -> float:#solving for a using variable gamma and Cp 
    a = np.sqrt(gamma * R * T)
    return a

#smarts model
@dataclass(frozen=True)
class SmartsModel:
    
    hpr_h2: float #J/kg
    fst: float
    phi: float
    theta: float
    x_react: float

    #normalized preburner length
    def x_norm(self,x:float,combustion_end:float) -> float:
        X = (x - self.x_react)/(combustion_end - self.x_react)
        return max(0.0, min(X, 1.0))
        return X

    #mixing eff func 
    def eta(self,x:float, eta_total:float, combustion_end:float) -> float:
        X = self.x_norm(x, combustion_end)
        return eta_total * (self.theta * X)/(1 + (self.theta - 1)*X)

    #deriv of phi with respect to dx but that part is sort of like inlucded later
    def dPHI(self,x:float,dx:float,eta_total:float,combustion_end:float) -> float:
        xcurrent = x
        xprev = x - dx
        dPHI = self.phi * (self.eta(xcurrent,eta_total,combustion_end) - self.eta(xprev,eta_total,combustion_end))
        return dPHI
    #Smarts combustion model heat release with respect to dx 
    def dHtdx(self,x:float, dx:float, eta_total:float, combustion_end:float) -> float:
            return (self.dPHI(x, dx, eta_total, combustion_end) * self.hpr_h2 * self.fst) / dx
        

#Tube initial conditions
@dataclass(frozen=True)
class ConstantAreaInletConditions:
    dir_air: float
    d_h2: float

    P_air: float
    P_H2: float
    T_air: float
    T_H2: float

    M1: float
    M2: float
    M3: float

    mdot_Air: float
    mdot_H2: float
    injMdot: float
    Vinj: float

    air_inletGamma: float
    h2_inletGamma: float
    Y_mix: np.ndarray
    R_mix: float
    TstagAir: float

    @property
    def A_airInjs(self):
        return (np.pi * (self.dir_air/2)**2) 
    @property
    def A_H2Injs(self) -> float:
        return np.pi * (self.d_h2 / 2)**2
    
    @property
    def mdot_i(self):
        return self.mdot_Air + self.mdot_H2

    Y_mix: np.ndarray
    R_mix: float
    air_inletGamma: float
    h2_inletGamma: float


#HyperReact initial conditions
@dataclass(frozen=True)
class WindTunnelInletConditions:

    #there are the initial conditions i am using for the code
    #the values are taken from dreyers paper (have a copy in zotero)
    #I am specifically using the case pb-3

    dir_air: float
    d_h2: float

    P_air: float
    P_H2: float
    T_air: float
    T_H2: float

    M1: float
    M2: float
    M3: float

    mdot_Air: float
    mdot_H2: float
    injMdot: float
    Vinj: float

    air_inletGamma: float
    h2_inletGamma: float
    Y_mix: np.ndarray
    R_mix: float
    TstagAir: float

    @property
    def A_airInjs(self):
        return (np.pi * (self.dir_air/2)**2) 
    @property
    def A_H2Injs(self) -> float:
        return np.pi * (self.d_h2 / 2)**2
    
    @property
    def mdot_i(self):
        return self.mdot_Air + self.mdot_H2

    Y_mix: np.ndarray
    R_mix: float
    air_inletGamma: float
    h2_inletGamma: float

def select_case(config,geometry,inlet_conditions):
    
    if config.geometry_type == "wind_tunnel":
        if not isinstance(geometry, WindTunnelGeometry):
            raise TypeError(
                "wind_tunnel requires WindTunnelGeometry")
        
        if not isinstance(inlet_conditions, WindTunnelInletConditions):
            raise TypeError(
                "wind_tunnel requires WindTunnelInletConditions")

    elif config.geometry_type == "constant_area":
        if not isinstance(geometry, ConstantAreaGeometry):
            raise TypeError(
                "constant_area requires ConstantAreaGeometry")

        if not isinstance(inlet_conditions, ConstantAreaInletConditions):
            raise TypeError(
                "constant_area requires ConstantAreaInletConditions")
        
    else:
        raise ValueError(
            f"Unknown geometry type: {config.geometry_type}"
        )

    return geometry, inlet_conditions

class ForwardModel:
    def __init__(self,config,geometry_case,inlet_conditions,combustion_model,mechanism: str):
        self.config = config

        geometry, ICs = select_case(config,geometry_case,inlet_conditions)
        self.geometry = geometry
        self.ICs = ICs
        self.combustion_model = combustion_model
        self.gas = ct.Solution(mechanism)

    def gas_properties(self,T:float, P:float, Y:float):
        self.gas.TPY = T, P, Y
    
        return{
            "cp": self.gas.cp_mass,
            "h": self.gas.enthalpy_mass,
            "gamma": self.gas.cp_mass/self.gas.cv_mass,
            "R_specific": self.gas.cp_mass - self.gas.cv_mass,
            "s": self.gas.entropy_mass,
            "mu": self.gas.viscosity}

    def heat_release(self,x:float, dx:float, eta_total:float, combustion_end:float) -> float:
        if self.config.combustion:
            return self.combustion_model.dHtdx(x,dx,eta_total,combustion_end)
        else:
            return 0.0

    def friction(self, x:float, Cf_sampling: float) -> float:
        if self.config.friction:
            return self.geometry.cf_location(x,Cf_sampling)
        else:
            return 0

    #using this to track mdot in the potential case that stuff is injected in the pb
    def mdotFuncX (self,x: float) -> float:
        if x < self.geometry.x_injLocation: #pre injector mdot
            return self.ICs.mdot_i
        else:   #post injector mdot
            return self.ICs.mdot_i + self.ICs.injMdot 

    #just using FDM for mdot (needed in shapiros eq and stuff)    
    def delMdotdx(self,mdotn1: float, mdotn: float,x1: float,x: float) -> float: #dmdot/dx function
        return (mdotn1 - mdotn)/(x1-x)

    #1st order ODE Functions
    #these are just shapiros 1d flow equations for generalized flow. I believe I am missing like two parts but yeah 
    #they are converted to from dV/V and dP/P to dV/dx and dP/dx
    def dVdX (self,V: float,A: float,M: float,T: float,P: float,mdot: float,dmdotDX: float,
            Cf: float, x: float,dx: float,eta_total: float,combustion_end: float,bl_h: float,bl_growth: float) -> float: #first 4 parts of sharpios 1d flow eqn converted to dV/dx
        gas_Prop = self.gas_properties(T, P, self.ICs.Y_mix)
        cp = gas_Prop["cp"]
        gamma = gas_Prop["gamma"]

        term1 = ((-V)/(A * (1 - M**2)))* self.geometry.dAdx(x,bl_h,bl_growth)
        term2 = ((V/((1-M**2) * cp * T)) * self.heat_release(x,dx,eta_total,combustion_end))
        term3 = ((gamma *M**2)/(2 * (1 - M**2)))
        term4 = ((((4 * Cf * V)/self.geometry.Dh(x,bl_h,bl_growth))) - (2*(self.ICs.Vinj/mdot) * dmdotDX))
        term5 = (((V*(1 + gamma * M**2))/((1-M**2)*mdot)) * (dmdotDX))
        return term1 + term2 + (term3*term4) + term5

    def dPdX (self,V: float,A: float,M: float,T: float,P: float,mdot: float,dmdotDX: float,
            Cf: float, x: float,dx: float,eta_total: float,combustion_end: float,bl_h: float,bl_growth: float) -> float: #first 4 parts of sharpios 1d flow eqn converted to dP/dx
        gas_Prop = self.gas_properties(T, P, self.ICs.Y_mix)
        cp = gas_Prop["cp"]
        gamma = gas_Prop["gamma"]

        term1 = ((gamma * M**2 * P)/(A * (1 - M**2))) * self.geometry.dAdx(x,bl_h,bl_growth)
        term2 = -(((gamma * M**2 * P)/((1-M**2) * cp * T)) * self.heat_release(x,dx,eta_total,combustion_end))
        term3  = -((gamma * M**2 * (1 + (gamma-1) * M**2))/(2 * (1 - M**2)))
        term4 = (((4 * Cf * (P/self.geometry.Dh(x,bl_h,bl_growth)))) - (2 * ((self.ICs.Vinj * P)/(mdot * V)) * (dmdotDX)))
        term5 = -(((2 * gamma * M**2 * (1 + ((gamma-1)/2) *M**2)*P)/((1-M**2)*mdot)) * (dmdotDX))
        return term1 + term2 + (term3 * term4) + term5

    def pressureStagFunc(self,P: float,M: float,gamma: float) -> float:
        Pstag = P * (1 + (gamma - 1)/2 * M**2)**(gamma/(gamma-1))
        return Pstag
    def temperatureStagFunc(self,T: float,M: float,gamma: float) -> float:
        Tstag = T * (1 + (gamma - 1)/2 * M**2)
        return Tstag
    def choked_massFlow(self,Pstag: float,Astar: float,Tstag: float, gamma: float) -> float:
        mdot_choke = (Pstag * Astar/np.sqrt(Tstag)) * np.sqrt(gamma / self.ICs.R_mix) * ((gamma + 1)/2)**(-(gamma + 1)/(2*(gamma-1)))
        return mdot_choke

    def pstag_predicted(self,mdot: float,Astar: float,Tstag: float,gamma: float) -> float:
       
        Pstag_pred = mdot * (np.sqrt(Tstag)/Astar) / (np.sqrt(gamma / self.ICs.R_mix) * ((gamma + 1)/2)**(-(gamma + 1)/(2*(gamma-1))))
        return Pstag_pred

    def stagtostatic(self, Pstag: float,Tstag: float, M: float, gamma: float) -> float:
        middleTerm = (1 + ((gamma -1)/2) * M**2)
        P = Pstag * middleTerm **(- gamma/(gamma - 1))
        T = Tstag * middleTerm ** (-1)
        return P,T

    #residual for temp. Basically making sure that my resolved temperature in stuff like rk45 is constrained by my smarts heat release 
    #was havintg the issue that when I was solving for temp in rk45 using ideal gas law i would get massive temps because the variables were
    #just not propely constrained 
    def residualT(self,T_new: float,T_old: float,xOld: float,uOld: float,uNew: float,dx: float,eta_total: float,P: float,combustion_end: float) -> float:
        T_gasProperties_old = self.gas_properties(T_old, P, self.ICs.Y_mix)
        T_gasProperties_new = self.gas_properties(T_new, P, self.ICs.Y_mix)
        ht_old = T_gasProperties_old["h"]
        ht_new = T_gasProperties_new["h"]
        term1 = (ht_new - ht_old)
        term2 = (uNew**2 - uOld**2)/2
        term3 = self.heat_release(xOld,dx,eta_total,combustion_end) * dx
        return term1 + term2 - term3

    def newtonRaphson_T(self,T_Guess: float, T_old: float, xOld: float, uOld: float,
                        uNew: float, dx: float,eta_total: float,P: float,combustion_end: float) -> float:
        numIters = 0
        tol = 1e-8
        E = self.residualT(T_Guess, T_old, xOld, uOld, uNew, dx,eta_total,P,combustion_end)

        while abs(E) >= tol and numIters <= 100:
            deltaT = max(abs(T_Guess)*1e-6, 1e-6)
            dEdT = (self.residualT(T_Guess + deltaT, T_old, xOld, uOld, uNew, dx,eta_total,P,combustion_end) - E)/deltaT

            if not np.isfinite(dEdT) or abs(dEdT) < 1e-14:
                raise RuntimeError("Bad temperature Newton derivative")

            lamda = 1.0
            accepted = False

            while lamda > 1e-3:
                T_new = T_Guess - lamda * E/dEdT

                if T_new <= 0 or not np.isfinite(T_new) or T_new > 3*T_old:
                    lamda *= 0.5
                    continue

                E_new = self.residualT(T_new, T_old, xOld, uOld, uNew, dx,eta_total,P,combustion_end)

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
    #the point of this residual is to find a static pressure that is consistent with my inlet stagnation pressure 
    def pressureResidual(self,Pstag: float,P_guess: float,T: float,gamma: float, bl_h, bl_g) -> float:
        #mdot * R * T / ( P * A) = u
        u = (self.ICs.mdot_i * self.ICs.R_mix * T)/(P_guess * self.geometry.inlet_area(bl_h,bl_g))

        M = mNum(u, soS(T, self.ICs.R_mix, gamma))
        PstaticfromPstag = Pstag / (1 + 0.5 * (gamma - 1) * M**2)**(gamma/(gamma-1))

        return PstaticfromPstag - P_guess

    def newtonRaphson_P(self,P_guess: float, Pstag: float, T: float, gamma: float, bl_h, bl_g) -> float:
        numIters = 0
        tol = 1e-8
        E = self.pressureResidual(Pstag, P_guess, T, gamma, bl_h, bl_g)

        P_vals = np.linspace(0.01 * Pstag, 0.999999 * Pstag, 500)

        E_vals = [self.pressureResidual(Pstag, P, T, gamma, bl_h, bl_g)for P in P_vals]

       
        while abs(E) >= tol and numIters <= 100:
            deltaP = max(abs(P_guess)*1e-6, 1e-6)

            dEdP = (self.pressureResidual(Pstag, P_guess + deltaP, T, gamma, bl_h, bl_g) - E)/deltaP

            if not np.isfinite(dEdP) or abs(dEdP) < 1e-14:
                raise RuntimeError("Bad pressure Newton derivative")

            lamda = 1.0
            accepted = False
            while lamda > 1e-7:
                P_new = P_guess - lamda * E/dEdP
                if P_new <= 0 or not np.isfinite(P_new) or P_new > 3*Pstag:
                    lamda *= 0.5
                    continue
                E_new = self.pressureResidual(Pstag, P_new, T, gamma, bl_h, bl_g)
                #print("P_new", P_new,"lamda", lamda,"E_new", E_new)

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
    def rk45Step(self,V: float,P: float,Cf_sampling: float, h: float, x: float, T_preburner: float
                ,eta_total: float,combustion_end: float,bl_h: float,bl_growth: float) -> tuple[float,float,float,float,float,str]: #add stages for each mdot 3
        accepted = False 
        location,local_tol,h_max = self.geometry.geometry_regions(x)
        #print("V",V,"P",P,"x",x,"T_preburner",T_preburner,"eta_total",eta_total,"combustion_end",combustion_end,"bl_h",bl_h,"bl_growth",bl_growth)
    
        h = min(h,h_max)
        attempts = 0

        while accepted != True:
            attempts += 1

            if attempts > 200:
                raise RuntimeError(
                    f"rk45Step failed to accept step: "
                    f"x={x}, h={h}, V={V}, P={P}, T={T_preburner}, "
                    f"Cf_sampling={Cf_sampling}, eta_total={eta_total}, "
                    f"combustion_end={combustion_end}")
            
            if h < 1e-14:
                raise RuntimeError(f"RK45 step size got too small at x = {x:.4f}")
                
            x1 = x
            Cf1 = self.friction(x1,Cf_sampling)
            A1 = self.geometry.geom_Area(x1,bl_h,bl_growth)
            V1 = V
            P1 =  P

            try:
                T1 = T_preburner
                a1 = soS(T1,self.ICs.R_mix,self.gas_properties(T1, P1, self.ICs.Y_mix)["gamma"])
            except:
                h *= 0.5
                continue

            mdot_1Cur = self.mdotFuncX(x1)
            mdot_1Prev = self.mdotFuncX(x-h)  
            M1 = mNum(V1,a1)
            k1V = h * self.dVdX(V1,A1,M1,T1,P1,mdot_1Cur,self.delMdotdx(mdot_1Cur,mdot_1Prev,x1,x1-h),Cf1,x1,h,eta_total,combustion_end,bl_h,bl_growth)
            k1P = h * self.dPdX(V1,A1,M1,T1,P1,mdot_1Cur,self.delMdotdx(mdot_1Cur,mdot_1Prev,x1,x1-h),Cf1,x1,h,eta_total,combustion_end,bl_h,bl_growth)

            x2 = x1 + 1/5 * h
            Cf2 = self.friction(x2,Cf_sampling)
            A2 = self.geometry.geom_Area(x2,bl_h,bl_growth)
            V2 = V + 1/5 * k1V 
            P2 = P + 1/5 * k1P
            try:
                T2 = self.newtonRaphson_T(T1, T1, x1, V1, V2, 1/5 * h,eta_total,P2,combustion_end)
                a2 = soS(T2,self.ICs.R_mix,self.gas_properties(T2, P2, self.ICs.Y_mix)["gamma"])
            except:
                h *= 0.5
                continue
            M2 = mNum(V2,a2)
            mdot_2Cur = self.mdotFuncX(x2)
            mdot_2Prev = self.mdotFuncX(x1) 
            k2V = h * self.dVdX(V2,A2,M2,T2,P2,mdot_2Cur,self.delMdotdx(mdot_2Cur,mdot_2Prev,x2,x1),Cf2,x2,1/5 * h,eta_total,combustion_end,bl_h,bl_growth)
            k2P = h * self.dPdX(V2,A2,M2,T2,P2,mdot_2Cur,self.delMdotdx(mdot_2Cur,mdot_2Prev,x2,x1),Cf2,x2,1/5 * h,eta_total,combustion_end,bl_h,bl_growth)

            x3 = x1 + 3/10 * h
            Cf3 = self.friction(x3,Cf_sampling)

            A3 = self.geometry.geom_Area(x3,bl_h,bl_growth)
            V3 = V + 3/40 * k1V + 9/40 * k2V
            P3 = P + 3/40 * k1P + 9/40 * k2P
            try:
                T3 = self.newtonRaphson_T(T1, T1, x1, V1, V3, 3/10 * h,eta_total,P3,combustion_end) 
                a3 = soS(T3,self.ICs.R_mix,self.gas_properties(T3, P3, self.ICs.Y_mix)["gamma"])
            except:
                h *= 0.5
                continue
            M3 = mNum(V3,a3)
            mdot_3Cur = self.mdotFuncX(x3)
            mdot_3Prev = self.mdotFuncX(x1) 
            k3V = h * self.dVdX(V3,A3,M3,T3,P3,mdot_3Cur,self.delMdotdx(mdot_3Cur,mdot_3Prev,x3,x2),Cf3,x3,3/10 * h,eta_total,combustion_end,bl_h,bl_growth)
            k3P = h *self. dPdX(V3,A3,M3,T3,P3,mdot_3Cur,self.delMdotdx(mdot_3Cur,mdot_3Prev,x3,x2),Cf3,x3,3/10 * h,eta_total,combustion_end,bl_h,bl_growth)

            x4 = x1 + 4/5 * h
            Cf4 = self.friction(x4,Cf_sampling)
            A4 = self.geometry.geom_Area(x4,bl_h,bl_growth)

            V4 = V + 44/45 * k1V - 56/15 * k2V + 32/9 * k3V
            P4 = P + 44/45 * k1P - 56/15 * k2P + 32/9 * k3P
            try:
                T4 = self.newtonRaphson_T(T1, T1, x1, V1, V4, 4/5 * h,eta_total,P4,combustion_end) 
                a4 = soS(T4,self.ICs.R_mix,self.gas_properties(T4, P4, self.ICs.Y_mix)["gamma"])
            except:
                h *= 0.5
                continue
            M4 = mNum(V4,a4)
            mdot_4Cur = self.mdotFuncX(x4)
            mdot_4Prev = self.mdotFuncX(x1) 
            k4V = h * self.dVdX(V4,A4,M4,T4,P4,mdot_4Cur,self.delMdotdx(mdot_4Cur,mdot_4Prev,x4,x3),Cf4,x4,4/5 * h,eta_total,combustion_end,bl_h,bl_growth)
            k4P = h * self.dPdX(V4,A4,M4,T4,P4,mdot_4Cur,self.delMdotdx(mdot_4Cur,mdot_4Prev,x4,x3),Cf4,x4,4/5 * h,eta_total,combustion_end,bl_h,bl_growth)

            x5 = x1 + 8/9 * h
            Cf5 = self.friction(x5,Cf_sampling)
            A5 = self.geometry.geom_Area(x5,bl_h,bl_growth)

            V5 = V + 19372/6561 * k1V - 25360/2187 * k2V + 64448/6561 * k3V - 212/729 * k4V
            P5 = P + 19372/6561 * k1P - 25360/2187 * k2P + 64448/6561 * k3P - 212/729 * k4P
            try:
                T5 = self.newtonRaphson_T(T1, T1, x1, V1, V5, 8/9 * h,eta_total,P5,combustion_end)
                a5 = soS(T5,self.ICs.R_mix,self.gas_properties(T5, P5, self.ICs.Y_mix)["gamma"])

            except:
                h *= 0.5
                continue

            M5 = mNum(V5,a5)
            mdot_5Cur = self.mdotFuncX(x5)
            mdot_5Prev = self.mdotFuncX(x1) 
            k5V = h * self.dVdX(V5,A5,M5,T5,P5,mdot_5Cur,self.delMdotdx(mdot_5Cur,mdot_5Prev,x5,x1),Cf5,x5,8/9 * h,eta_total,combustion_end,bl_h,bl_growth)
            k5P = h * self.dPdX(V5,A5,M5,T5,P5,mdot_5Cur,self.delMdotdx(mdot_5Cur,mdot_5Prev,x5,x1),Cf5,x5,8/9 * h,eta_total,combustion_end,bl_h,bl_growth)

            x6 = x1 + h
            Cf6 = self.friction(x6,Cf_sampling)
            A6 = self.geometry.geom_Area(x6,bl_h,bl_growth)

            V6 = V + 9017/3168 * k1V - 355/33 * k2V + 46732/5247 * k3V + 49/176 * k4V - 5103/18656 * k5V
            P6 = P + 9017/3168 * k1P - 355/33 * k2P + 46732/5247 * k3P + 49/176 * k4P - 5103/18656 * k5P
            try:
                T6 = self.newtonRaphson_T(T1, T1, x1, V1, V6, 1 * h,eta_total,P6,combustion_end)
                a6 = soS(T6,self.ICs.R_mix,self.gas_properties(T6, P6, self.ICs.Y_mix)["gamma"])
            except:
                h *= 0.5
                continue

            M6 = mNum(V6,a6)
            mdot_6Cur = self.mdotFuncX(x6)
            mdot_6Prev = self.mdotFuncX(x1) 
            k6V = h * self.dVdX(V6,A6,M6,T6,P6,mdot_6Cur,self.delMdotdx(mdot_6Cur,mdot_6Prev,x6,x1),Cf6,x6,h,eta_total,combustion_end,bl_h,bl_growth)
            k6P = h * self.dPdX(V6,A6,M6,T6,P6,mdot_6Cur,self.delMdotdx(mdot_6Cur,mdot_6Prev,x6,x1),Cf6,x6,h,eta_total,combustion_end,bl_h,bl_growth)

            #5th order solution 
            v_5Order = V + 35/384 * k1V + 500/1113 * k3V + 125/192 * k4V - 2187/6784 * k5V + 11/84 * k6V
            p_5Order = P + 35/384 * k1P + 500/1113 * k3P + 125/192 * k4P - 2187/6784 * k5P + 11/84 * k6P

            x7 = x1 + h
            Cf7 = self.friction(x7,Cf_sampling)
            A7 = self.geometry.geom_Area(x7,bl_h,bl_growth)
            V7 = v_5Order
            P7 = p_5Order
            try:
                T7 = self.newtonRaphson_T(T1, T1, x1, V1, V7, 1 * h,eta_total,P7,combustion_end)
                a7 = soS(T7,self.ICs.R_mix,self.gas_properties(T7, P7, self.ICs.Y_mix)["gamma"])
            except:
                h *= 0.5
                continue       
            M7 = mNum(V7,a7)
            mdot_7Cur = self.mdotFuncX(x7)
            mdot_7Prev = self.mdotFuncX(x1) 
            k7V = h * self.dVdX(V7,A7,M7,T7,P7,mdot_7Cur,self.delMdotdx(mdot_7Cur,mdot_7Prev,x7,x1),Cf7,x7,h,eta_total,combustion_end,bl_h,bl_growth)
            k7P = h * self.dPdX(V7,A7,M7,T7,P7,mdot_7Cur,self.delMdotdx(mdot_7Cur,mdot_7Prev,x7,x1),Cf7,x7,h,eta_total,combustion_end,bl_h,bl_growth)

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
                Tnext = self.newtonRaphson_T(T1, T1, x1, V1, Vnext, 1 * h,eta_total,Pnext,combustion_end)

                s = 1.2 if errorRatio == 0 else min(2, 0.9 * errorRatio**(-1/5))

                h_next = min(s * h, h_max)
                break
        
        return xNext, Vnext, Pnext, Tnext,h_next, location

    #Full Solver
    def solver(self,Preburner_TStag: float,Cf_sampling: float,eta_total: float,combustion_end: float,bl_h: float,
            bl_growth: float,scale: float, acceptedScale: bool, supersonicSolve: bool) -> dict[str, Any]:
        inlet_T = Preburner_TStag #k

        if self.config.geometry_type == "wind_tunnel":
            Preburner_predictedPStag = self.pstag_predicted(self.ICs.mdot_i, self.geometry.throat_area(bl_h,bl_growth), Preburner_TStag, self.gas_properties(Preburner_TStag, 101325, self.ICs.Y_mix)["gamma"])
            og_Preburner_P = self.newtonRaphson_P(Preburner_predictedPStag,Preburner_predictedPStag, inlet_T, self.gas_properties(inlet_T, 101325, self.ICs.Y_mix)["gamma"],bl_h, bl_growth)

            if acceptedScale == False:
                if scale ==1:
                    Preburner_P = og_Preburner_P
                else:
                    Preburner_P = og_Preburner_P * scale

                Preburner_U = self.ICs.mdot_i/(Preburner_P * self.geometry.inlet_area(bl_h, bl_growth) / (self.ICs.R_mix * inlet_T))
                Preburner_gasProperties = self.gas_properties(inlet_T, Preburner_P, self.ICs.Y_mix)

                M_Preburner_Inlet = Preburner_U/soS(inlet_T,self.ICs.R_mix,Preburner_gasProperties["gamma"])
                rho_preburner = self.ICs.mdot_i/(self.geometry.inlet_area(bl_h, bl_growth) * Preburner_U)
                Preburner_Pstag = Preburner_predictedPStag

                inlet_U = Preburner_U
                inlet_P = Preburner_P
                inlet_Tstag = Preburner_TStag
                inlet_Pstag = Preburner_Pstag
                inlet_rho = rho_preburner
                inlet_M = M_Preburner_Inlet

            elif acceptedScale == True:
                Preburner_P = og_Preburner_P * scale
                Preburner_gasProperties = self.gas_properties(inlet_T, Preburner_P, self.ICs.Y_mix)
                Preburner_U = self.ICs.mdot_i/(Preburner_P * self.geometry.inlet_area(bl_h, bl_growth) / (self.ICs.R_mix * inlet_T))
                M_Preburner_Inlet = Preburner_U/soS(inlet_T,self.ICs.R_mix,Preburner_gasProperties["gamma"])
                rho_preburner = self.ICs.mdot_i/(self.geometry.inlet_area(bl_h, bl_growth) * Preburner_U)
                Preburner_Pstag = self.pressureStagFunc(Preburner_P,M_Preburner_Inlet,Preburner_gasProperties["gamma"])

                inlet_U = Preburner_U
                inlet_P = Preburner_P
                inlet_Tstag = Preburner_TStag
                inlet_Pstag = Preburner_Pstag
                inlet_rho = rho_preburner
                inlet_M = M_Preburner_Inlet

            PT_locations = [0.1,0.3,0.4,0.495,0.5,0.505,0.51,0.55,0.6,0.64]

        elif self.config.geometry_type == "constant_area":
            inlet_Tstag = Preburner_TStag
            inlet_Pstag = self.pstag_predicted(self.ICs.mdot_i, self.geometry.smallest_eff_area(bl_h,bl_growth), 
                                               inlet_Tstag, self.gas_properties(inlet_Tstag, 101325, self.ICs.Y_mix)["gamma"])
            #inlet_P = self.newtonRaphson_P(inlet_Pstag,inlet_Pstag, inlet_T, 
                                #           self.gas_properties(inlet_T, 101325, self.ICs.Y_mix)["gamma"],bl_h, bl_growth)
            inlet_P = 0.99* inlet_Pstag
            inlet_gasProperties = self.gas_properties(inlet_T, inlet_P, self.ICs.Y_mix)
            
            inlet_U = self.ICs.mdot_i/(inlet_P * self.geometry.inlet_area(bl_h, bl_growth) / (self.ICs.R_mix * inlet_T))
            inlet_M = inlet_U/soS(inlet_T,self.ICs.R_mix,inlet_gasProperties["gamma"])
            inlet_rho = self.ICs.mdot_i/(self.geometry.inlet_area(bl_h, bl_growth) * inlet_U)
            sonicSolveCount = 0
            dataGatherCounter = 0
            PT_locations = [0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.9]

            
        temp = [inlet_T]                # creating fresh arrays in function 
        velocities = [inlet_U]
        pressure = [inlet_P]
        pStag = [inlet_Pstag]
        tStag = [inlet_Tstag]
        machNum = [inlet_M]
        density = [inlet_rho]
        areaList = [self.geometry.geom_Area(0,bl_h,bl_growth)]
        dAdxList = [0.0]
        areaRatio = [1.0]
        xList = [0.0] #this list starts at the preburner 
        stepList = [1e-1]
        mdotList = [self.ICs.mdot_i]
        
        pt_location = []
        pt_pressures = []
        


        sInitial = self.gas_properties(inlet_T, inlet_P,self.ICs.Y_mix)["s"]
        entropy = [sInitial]

        mdotReconstructed = [self.ICs.mdot_i] #recontruction array to check if calcs are correct 
        throatP = 0
        pb_count = 0
        throat_count = 0
        conv_count = 0
        div_count = 0
        max_solver_steps = 20000
        solver_steps = 0

        while (xList[-1] < self.geometry.solver_end_location):
            solver_steps += 1

            if solver_steps > max_solver_steps:
                print("solver failed, exceeded max_solver_steps")
                print("M = ",machNum[-1], "M2 = ",machNum[-2])
                print("x = ",xList[-1], "x2 = ",xList[-2])
                
                print(f"dMdx={(machNum[-1] - machNum[-2]) / (xList[-1] - xList[-2]):.6e}")
                print(f"V={velocities[-1]}, P={pressure[-1]}, T={temp[-1]}")
                break
                '''
                raise RuntimeError(
                    f"solver exceeded max_solver_steps:\n "
                    f"x={xList[-1]}, solver_end_location={self.geometry.solver_end_location},\n"
                    f"h={stepList[-1]},\n "
                    f"M={machNum[-1]},\n"
                    f"dMdx={(machNum[-1] - machNum[-2]) / (xList[-1] - xList[-2]):.6e},\n"                    
                    f"V={velocities[-1]},\n"
                    f"P={pressure[-1]},\n"
                    f"T={temp[-1]},\n"
                    f"Area={self.geometry.geom_Area(xList[-1],bl_h,bl_growth)},\n"
                    f"dAdx={self.geometry.dAdx(xList[-1],bl_h,bl_growth)},\n"
                    f"location={self.geometry.geometry_regions(xList[-1])},\n"
                    f"acceptedScale={acceptedScale},\n"
                    f"supersonicSolve={supersonicSolve},\n"
                    f"Cf_sampling={Cf_sampling},\n"
                    f"eta_total={eta_total},\n"
                    f"combustion_end={combustion_end},\n"
                    f"bl_h={bl_h},\n"
                    f"bl_growth={bl_growth}\n")'''
                    

            xPrev = xList[-1]     
            hPrev = stepList[-1] #from step 0 to step 1 and then step 1 to step 2 etc 
            
            Vbefore = velocities[-1]
            Pbefore = pressure[-1]
            Tbefore = temp [-1]
            
            xNext, VCurrent, PCurrent, TCurrent, hNext, location = self.rk45Step(Vbefore,Pbefore,Cf_sampling,hPrev, 
                                                                                 xPrev,Tbefore,eta_total,combustion_end,bl_h,bl_growth)

            if location == "Preburner":
                pb_count +=1
            elif location == "Throat":
                throat_count +=1 
            elif location == "Conv Nozzle":
                conv_count+=1
            elif location == "Div Nozzle":
                div_count +=1 

            currentMix_properties = self.gas_properties(TCurrent, PCurrent,self.ICs.Y_mix)
            currentMix_gamma = currentMix_properties["gamma"]

            xList.append(xNext)
            xCurrent = xList[-1]

            areaList.append(self.geometry.geom_Area(xCurrent,bl_h,bl_growth))
            dAdxList.append(self.geometry.dAdx(xCurrent,bl_h,bl_growth))

            mdotlocal = self.mdotFuncX(xCurrent)
            stepList.append(hNext)

            velocities.append(VCurrent)
            pressure.append(PCurrent)

            rhoCurrent = mdotlocal/(self.geometry.geom_Area(xCurrent,bl_h,bl_growth) * VCurrent) 
            density.append(rhoCurrent)

            temp.append(TCurrent)

            mdotReconstructed.append(rhoCurrent * VCurrent * self.geometry.geom_Area(xCurrent,bl_h,bl_growth))
            mdotList.append(self.mdotFuncX(xCurrent))

            aCurrent = soS(TCurrent,self.ICs.R_mix,currentMix_gamma)
            MCurrent = mNum(VCurrent,aCurrent)
            machNum.append(MCurrent)

            Pstag_current = self.pressureStagFunc(PCurrent, MCurrent, currentMix_gamma)
            pStag.append(Pstag_current)

            Tstag_current = self.temperatureStagFunc(TCurrent, MCurrent, currentMix_gamma)
            tStag.append(Tstag_current)

            sCurrent = currentMix_properties["s"]
            entropy.append(sCurrent)
            
            if self.config.geometry_type == "wind_tunnel":
                if supersonicSolve == True:
                    pt_P, pt_x= pressureTap(xList[-2],pressure[-2],xCurrent, PCurrent,PT_locations)

                    if pt_P is not None:
                        pt_location.append(pt_x)
                        pt_pressures.append(pt_P)

                if MCurrent > 0.99 and supersonicSolve == False:
                    break

                elif acceptedScale == True and supersonicSolve == True and (self.geometry.geometry_regions(xCurrent) == "Throat" or (MCurrent >= 0.99 and xCurrent  < self.geometry.throat_loc)):
                    throatP = pressure[-1]
                    throatT = temp[-1]
                    throatPstag = pStag[-1]
                    throatTstag = tStag[-1]
                    MachN = 1.005
                    throatMix_properties = self.gas_properties(throatT, throatP,self.ICs.Y_mix)
                    throatMix_gamma = throatMix_properties["gamma"]
                    entropy_throat = throatMix_properties["s"]

                    P_new, T_New = self.stagtostatic(throatPstag,throatTstag,MachN,throatMix_gamma)
                    V_new = MachN * soS(T_New,self.ICs.R_mix,throatMix_gamma)
                    xEndofThroat = self.geometry.throat_loc + 0.005
                    rho_New = mdotlocal/(self.geometry.geom_Area(xEndofThroat,bl_h,bl_growth) * V_new)

                    mdotReconstructed.append(rho_New * V_new * self.geometry.geom_Area(xEndofThroat,bl_h,bl_growth))
                    mdotList.append(self.mdotFuncX(xEndofThroat))
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
                    areaList.append(self.geometry.geom_Area(xEndofThroat,bl_h,bl_growth))
                    dAdxList.append(self.geometry.dAdx(xEndofThroat,bl_h,bl_growth))

                    pt_P, pt_x = pressureTap(xList[-2],pressure[-2],xEndofThroat, P_new,PT_locations)
                    if pt_P is not None:

                        pt_location.append(pt_x)
                        pt_pressures.append(pt_P)
                else:
                    continue

            if self.config.geometry_type == "constant_area":

                pt_P, pt_x= pressureTap(xList[-2],pressure[-2],xCurrent, PCurrent,PT_locations)

                if pt_P is not None:
                    pt_location.append(pt_x)
                    pt_pressures.append(pt_P)

                if len(pt_location) > 3 and dataGatherCounter == 0:
                    dynamic_viscosity = self.gas_properties(temp[-1],pressure[-1],self.ICs.Y_mix)["mu"]
                    d_tube = np.sqrt(self.geometry.tube_area)
                    Re = density[-1] * velocities[-1] * d_tube / dynamic_viscosity
                    data_vector = np.array([Re, machNum[-1],temp[-1],xList[-1]])
                    dataGatherCounter += 1


                #1st mach num crossing 
                if 0.995 <  MCurrent < 1.00 and (sonicSolveCount == 0 or sonicSolveCount > 1): 
                    sonicSolveCount = 1
                    pre_chokeP = pressure[-1]
                    pre_chokeT = temp[-1]
                    pre_chokePstag = pStag[-1]
                    pre_chokeTstag = tStag[-1]
                    pre_chokeX = xList[-1]
                    MachN = 1.0001
                    choke_Mix_properties = self.gas_properties(pre_chokeT, pre_chokeP,self.ICs.Y_mix)
                    choke_Mix_gamma = choke_Mix_properties["gamma"]
                    choke_entropy = choke_Mix_properties["s"]

                    P_new, T_New = self.stagtostatic(pre_chokePstag,pre_chokeTstag,MachN,choke_Mix_gamma)
                    V_new = MachN * soS(T_New,self.ICs.R_mix,choke_Mix_gamma)
                    post_chokeX = pre_chokeX + (self.geometry.solver_end_location)/100
                    rho_New = mdotlocal/(self.geometry.geom_Area(post_chokeX,bl_h,bl_growth) * V_new)

                    mdotReconstructed.append(rho_New * V_new * self.geometry.geom_Area(post_chokeX,bl_h,bl_growth))
                    mdotList.append(self.mdotFuncX(post_chokeX))
                    stepList.append(0.001)
                    xList.append(post_chokeX)
                    pressure.append(P_new)
                    velocities.append(V_new)
                    temp.append(T_New)
                    density.append(rho_New)
                    machNum.append(MachN)
                    pStag.append(pre_chokePstag)
                    tStag.append(pre_chokeTstag)
                    entropy.append(choke_entropy)
                    areaList.append(self.geometry.geom_Area(post_chokeX,bl_h,bl_growth))
                    dAdxList.append(self.geometry.dAdx(post_chokeX,bl_h,bl_growth))

                    pt_P, pt_x = pressureTap(xList[-2],pressure[-2],post_chokeX, P_new,PT_locations)

                    if pt_P is not None:
                        pt_location.append(pt_x)
                        pt_pressures.append(pt_P)

                elif 1.00 < MCurrent < 1.005 and (sonicSolveCount == 1 or sonicSolveCount > 1):
                    sonicSolveCount = 2
                    post_chokeP = pressure[-1]
                    post_chokeT = temp[-1]
                    post_chokePstag = pStag[-1]
                    post_chokeTstag = tStag[-1]
                    post_chokeX = xList[-1]
                    MachN = 0.9999
                    choke_Mix_properties = self.gas_properties(post_chokeT, post_chokeP,self.ICs.Y_mix)
                    choke_Mix_gamma = choke_Mix_properties["gamma"]
                    choke_entropy = choke_Mix_properties["s"]

                    P_new, T_New = self.stagtostatic(post_chokePstag,post_chokeTstag,MachN,choke_Mix_gamma)
                    V_new = MachN * soS(T_New,self.ICs.R_mix,choke_Mix_gamma)
                    pre_chokeX = post_chokeX + (self.geometry.solver_end_location)/100
                    rho_New = mdotlocal/(self.geometry.geom_Area(pre_chokeX,bl_h,bl_growth) * V_new)

                    mdotReconstructed.append(rho_New * V_new * self.geometry.geom_Area(pre_chokeX,bl_h,bl_growth))
                    mdotList.append(self.mdotFuncX(pre_chokeX))
                    stepList.append(0.001)
                    xList.append(pre_chokeX)
                    pressure.append(P_new)
                    velocities.append(V_new)
                    temp.append(T_New)
                    density.append(rho_New)
                    machNum.append(MachN)
                    pStag.append(post_chokePstag)
                    tStag.append(post_chokeTstag)
                    entropy.append(choke_entropy)
                    areaList.append(self.geometry.geom_Area(pre_chokeX,bl_h,bl_growth))
                    dAdxList.append(self.geometry.dAdx(pre_chokeX,bl_h,bl_growth))

                    pt_P, pt_x = pressureTap(xList[-2],pressure[-2],pre_chokeX, P_new,PT_locations)

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
            "Area_Ratio": AreaRatio_List, 
            "Choked_Area_Ratio": Area_List[0]/Area_List[-1], 
            "Initial_Pstag" : pStag_List[0],
            "Initial_Tstag" : tStag_List[0],
            "Preburner_Count": pb_count,
            "Conv_Count": conv_count,
            "throat_Count": throat_count,
            "Throat_Pressure": throatP,
            "PT_P": pt_PressureList,
            "PT_X": pt_locationList,
            "data_vector": data_vector if self.config.geometry_type == "constant_area" else None}
    
    #Sweeping
    def chokedLocationResiduals(self,scale: float, Cf_sampling: float,eta_total: float,combustion_end: float,bl_h: float,bl_growth: float) -> float:
        results = self.solver(self.ICs.TstagAir,Cf_sampling,eta_total,combustion_end,bl_h,bl_growth,scale,False,False)
        x_Choke = results["x"][-1]
        residual = self.geometry.throat_loc - x_Choke  #we want this to be zero
        return residual
        
    def eval_scale(self,scale: float,Cf_sampling: float,eta_total: float,combustion_end: float,bl_h: float,bl_growth: float) -> tuple[float,float]:
        try:
            res = self.chokedLocationResiduals(scale, Cf_sampling,eta_total,combustion_end,bl_h,bl_growth)
            return scale, res
        except Exception as eS:
            print("Scale Failed ", scale, eS)
            traceback.print_exc()
            return scale, np.nan
        
    def scaling_InletPressure_NOTPar(self,Cf_sampling: float, eta_total: float, combustion_end: float, bl_h: float, bl_growth: float) -> tuple[float,float,float,float]:

        max_scale = 1.0
        max_res = self.chokedLocationResiduals(
            max_scale, Cf_sampling, eta_total, combustion_end, bl_h, bl_growth
        )

        if max_res > 0:
            direction = 1
        elif max_res < 0:
            direction = -1
        else:
            return max_scale, max_scale, max_res, max_res

        prev_scale = max_scale
        prev_res = max_res

        for i in range(1, 100):
            try:
                cur_scale = max_scale + direction * (i / 10)
                cur_scale, cur_res = self.eval_scale(
                    cur_scale, Cf_sampling, eta_total, combustion_end, bl_h, bl_growth
                )
            except Exception as eS:
                print(f"Failed because of: {eS}")
                traceback.print_exc()
                continue

            if not np.isfinite(cur_res):
                continue

            if cur_res * prev_res < 0:
                scale_low = cur_scale
                scale_high = prev_scale
                res_low = cur_res
                res_high = prev_res

                if scale_low > scale_high:
                    scale_low, scale_high = scale_high, scale_low
                    res_low, res_high = res_high, res_low

                return scale_high, scale_low, res_high, res_low

            prev_scale = cur_scale
            prev_res = cur_res

        raise RuntimeError(
            f"No bracket found for inlet pressure scale. "
            f"Last scale={prev_scale}, last residual={prev_res}, "
            f"direction={direction}, "
        )


    def scale_HybridNewBisec(self,scale_low: float, scale_high: float,res_low: float, res_high: float,
                            Cf_sampling: float,eta_total: float,combustion_end: float,bl_h: float,bl_growth: float) -> tuple[float,float]:

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

            res_candidate = self.chokedLocationResiduals(scale_candidate, Cf_sampling,eta_total,combustion_end,bl_h,bl_growth)

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
    
    #run forward model
    def run(self,percent_obstruction: float,Cf_sampling: float,eta_total: float,combustion_end: float,bl_growth: float,) -> dict[str, Any]:

        if self.config.boundary_layer:
            bl_h = self.geometry.bl_height(percent_obstruction)
            effective_bl_growth = bl_growth
        else:
            bl_h = 0.0
            effective_bl_growth = 0.0

        if self.config.geometry_type == "wind_tunnel":
                
            start_time_total =  time.perf_counter()
            start_time_scaling = time.perf_counter()

            scale_low, scale_high, res_low, res_high = (self.scaling_InletPressure_NOTPar(Cf_sampling,eta_total,combustion_end,bl_h,effective_bl_growth,))
            end_time_scaling = time.perf_counter()

            if res_low * res_high > 0:
                raise RuntimeError(f"Scale bracket does not contain a root: "f"scale_low={scale_low}, res_low={res_low}, "f"scale_high={scale_high}, res_high={res_high}")
            
            start_time_H = time.perf_counter()
            final_scale, final_res = self.scale_HybridNewBisec(scale_low,scale_high,res_low,res_high,Cf_sampling,
                eta_total,combustion_end,bl_h,effective_bl_growth,)
            end_time_H = time.perf_counter()

            start_time_solve = time.perf_counter()

            results = self.solver(Preburner_TStag=self.ICs.TstagAir,Cf_sampling=Cf_sampling,eta_total=eta_total,combustion_end=combustion_end,bl_h=bl_h,
                                bl_growth=effective_bl_growth,scale=final_scale,acceptedScale=True,supersonicSolve=True,)
            end_time_solve = time.perf_counter()

            end_time_total =  time.perf_counter()
            '''
            print("scaling time", end_time_scaling - start_time_scaling)
            print("Hybrid time", end_time_H - start_time_H)
            print("Solver time", end_time_solve - start_time_solve)
            print("Total time", end_time_total - start_time_total)
            '''
            return results

        elif self.config.geometry_type == "constant_area":
            start_time_total =  time.perf_counter()
           
            results = self.solver(Preburner_TStag=self.ICs.TstagAir,Cf_sampling=Cf_sampling,eta_total=eta_total,combustion_end=combustion_end,bl_h=bl_h,
                                bl_growth=effective_bl_growth,scale=0,acceptedScale=True,supersonicSolve=True,)

            end_time_total =  time.perf_counter()
            
            '''
            print("scaling time", end_time_scaling - start_time_scaling)
            print("Hybrid time", end_time_H - start_time_H)
            print("Solver time", end_time_solve - start_time_solve)
            print("Total time", end_time_total - start_time_total)
            '''
            return results