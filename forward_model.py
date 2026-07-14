import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from scipy import stats
from dataclasses import dataclass
from typing import Literal
gas = ct.Solution('h2_air.yaml')


class ModelConfig:
    geometry_type: Literal["wind_tunnel", "constant_area"] = "wind_tunnel"
    friction: bool = True
    boundary_layer: bool = True
    combustion: bool = True


@dataclass(frozen=True)
class ConstantAreaGeometry:
    area: float
    length: float
    
    @property
    def x_end(self) -> float:
        return self.length
    def geom_Area(self, x: float) -> float:
        return self.area
    
    def dAdx(self, x: float) -> float:
        return 0.0
    def Dh(self, x: float) -> float:
        return np.sqrt(self.geom_Area(x))
    
    def geometry_regions(self, x: float) -> float:
        return "Tube"
    

@dataclass(frozen=True)
class WindTunnelGeometry:
    preburner_area: float
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
    def geometry_regions(self, x: float) -> str:
        if x <= self.preburner_length:
            return "Preburner"
        elif self.throat_loc - 0.0005 <= x <= self.throat_loc + 0.0009: 
            return "Throat"
        elif x <= self.throat_loc:
            return "Conv Nozzle"
        elif x <= self.nozzle_exit:
            return "Div Nozzle"
        else:
            return "Test Section"

    #function that allows me to input the % of throat that I will  be obstructing and then getting a boundary layer height inreturn 
    #goal is to use the bL_Y and then just apply it to the whole conv nozzle sectoin 
    def bl_height(self,throatOb: float) -> float:
        unObstructed_throat_area = self.throat_Area - self.throat_Area * throatOb
        unObstructed_throat_height = np.sqrt(unObstructed_throat_area)
        bL_Y = self.throat_Height - unObstructed_throat_height
        return bL_Y

    #solving for area based on location and boundary layer stuff
    #only having the boundary in the converging and diverging parts of the nozzle. 
    #only having the boundary layer growth in the diverging part because that is when the flow goes super sonic
    def geom_Area(self,x: float, bl_h: float = 0.0 ,bl_growth: float = 0.0) -> float:
    
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
    def cf_location(self,x: float,Cf_dnz: float) -> float:
        region = self.geometry_regions(x)
        Cf_preburner = 0.0025
        Cf_cNz = 0.003
        if region == "Preburner":
            return Cf_preburner
        
        elif region == "Throat" or region == "Conv Nozzle":
            return Cf_cNz
        elif region == "Div Nozzle" or region == "Test Section":
            return Cf_dnz    

def select_geometry(config: ModelConfig):

    if config.geometry_type == "constant_area":
        print("using constant area geometry")
        return ConstantAreaGeometry

    elif config.geometry_type == "wind_tunnel":
        print("using wind tunnel geometry")

        return WindTunnelGeometry

    raise ValueError(f"Unknown geometry type: {config.geometry_type}")

geometry = select_geometry(config=ModelConfig)


def get_boundary_layer_inputs(config: ModelConfig, geometry: WindTunnelGeometry, throat_obstruction: float, bl_growth: float) -> tuple[float, float]:
    if not config.boundary_layer:
        return 0.0, 0.0

    bl_height = geometry.bl_height(throat_obstruction)
    return bl_height, bl_growth

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



#just getting all the properties for a given T,P,Y
def gas_properties(T:float, P:float, Y:float) -> float:
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

def get_Y(self, comp_string: str) -> float:
    gas.TPX = 300, 101325, comp_string
    return gas.Y.copy()

#solving for the mass fraction of the mixed gasses 
@property
def Ymix(YA:float,mdotAir:float, YB:float, mdotH2:float) -> float:
    Ymix = (mdotAir * YA + mdotH2 * YB)/(mdotAir + mdotH2)
    return Ymix

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
        if ModelConfig.combustion:
            return (self.dPHI(x, dx, eta_total, combustion_end) * self.hpr_h2 * self.fst) / dx
        else:
            return 0.0

#Tube initial conditions
@dataclass(frozen=True)
class TubeInletConditions:

#HyperReact initial conditions
@dataclass(frozen=True)
class WindTunnelInletConditions:

    #there are the initial conditions i am using for the code
    #the values are taken from dreyers paper (have a copy in zotero)
    #I am specifically using the case pb-3

    dir_air:float #meters
    d_h2 :float #meters
    A_airInjs = (np.pi * (dir_air/2)**2) 
    A_H2Injs = (np.pi * (d_h2/2)**2)

    P1 :float #Pa
    P2 :float #Pa
    P3 :float #Pa

    P_air = P1 #Pa. Pa is equal to P1 and P2 because they are the same injector and they are connected to the same plenum.
    P_H2 = P3 #Pa

    T_air :float #K
    T_air2:float #K
    T_H2 :float #K

    M1 :float
    M2 :float
    M3 :float

    mdot1_Air :float
    mdot2_Air :float
    mdot3_H2 :float

    mdotAir = mdot1_Air + mdot2_Air #injector 1 and 2 are the same so can just add them together
    mdotH2 = mdot3_H2 #big injector
    mdot_i = mdotAir + mdotH2

    injMdot: float

    Y_air = get_Y("O2:0.21, N2:0.79")
    Y_H2 = get_Y("H2:1.0")
    Y_mix = Ymix(Y_air, mdotAir, Y_H2, mdotH2)

    R_air = gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["R_specific"]
    R_H2 = gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["R_specific"]
    R_mix = gas_properties(300, 101325, Y_mix)["R_specific"]

    a1 = soS(T_air, R_air, gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"])
    a2 = soS(T_air2, R_air, gas_properties(T_air2, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"])
    a3 = soS(T_H2, R_H2, gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"])

    uA = M1 * a1
    uA_2 = M2 * a2
    uB = M3 * a3

    TstagA = T_air * (1 + ((gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"] - 1)/2) * M1**2)
    TstagB = T_H2 * (1 + ((gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"] - 1)/2) * M3**2)

    Pstag_Air = P_air * (1 + (gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"] - 1)/2 * M1**2)**(gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"]/(gas_properties(T_air, P_air, get_Y("O2:0.21, N2:0.79"))["gamma"]-1))
    Pstag_H2 = P_H2 * (1 + (gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"] - 1)/2 * M3**2)**(gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"]/(gas_properties(T_H2, P_H2, get_Y("H2:1.0"))["gamma"]-1))

   

    #using this to track mdot in the potential case that stuff is injected in the pb
    def mdotFuncX (self, x: float) -> float:
        if x < WindTunnelGeometry.x_injLocation: #pre injector mdot
            return self.mdot_i
        else:   #post injector mdot
            return self.mdot_i + self.injMdot 

#just using FDM for mdot (needed in shapiros eq and stuff)    
def delMdotdx(mdotn1: float, mdotn: float,x1: float,x: float) -> float: #dmdot/dx function
    return (mdotn1 - mdotn)/(x1-x)

#residual for temp. Basically making sure that my resolved temperature in stuff like rk45 is constrained by my smarts heat release 
#was havintg the issue that when I was solving for temp in rk45 using ideal gas law i would get massive temps because the variables were
#just not propely constrained 
def residualT(T_new: float,T_old: float,xOld: float,uOld: float,uNew: float,dx: float,eta_total: float,P: float,combustion_end: float) -> float:
    T_gasProperties_old = gas_properties(T_old, P, Ymix)
    T_gasProperties_new = gas_properties(T_new, P, Ymix)
    ht_old = T_gasProperties_old["h"]
    ht_new = T_gasProperties_new["h"]
    term1 = (ht_new - ht_old)
    term2 = (uNew**2 - uOld**2)/2
    term3 = SmartsModel.dHtdx(xOld,dx,eta_total,combustion_end) * dx
    return term1 + term2 - term3


#1st order ODE Functions
#these are just shapiros 1d flow equations for generalized flow. I believe I am missing like two parts but yeah 
#they are converted to from dV/V and dP/P to dV/dx and dP/dx
def dVdX (V,A,M,T,P,mdot,dmdotDX, Cf, x,dx,eta_total,combustion_end,bl_h,bl_growth): #first 4 parts of sharpios 1d flow eqn converted to dV/dx

    gas_Prop = gas_properties(T, P, Ymix)
    cp = gas_Prop["cp"]
    gamma = gas_Prop["gamma"]

    term1 = ((-V)/(A * (1 - M**2)))* geometry.dAdx(x,bl_h,bl_growth)
    term2 = ((V/((1-M**2) * cp * T)) * dHtdx(x,dx,eta_total,combustion_end))
    term3 = ((gamma *M**2)/(2 * (1 - M**2)))
    term4 = ((((4 * Cf * V)/Dh(x,bl_h,bl_growth))) - (2*(Vinj/mdot) * dmdotDX))
    term5 = (((V*(1 + gamma * M**2))/((1-M**2)*mdot)) * (dmdotDX))
    return term1 + term2 + (term3*term4) + term5

def dPdX (V,A,M,T,P,mdot,dmdotDX, Cf, x,dx,eta_total,combustion_end,bl_h,bl_growth): #first 4 parts of sharpios 1d flow eqn converted to dP/dx
    gas_Prop = gas_properties(T, P, Ymix)
    cp = gas_Prop["cp"]
    gamma = gas_Prop["gamma"]


    term1 = ((gamma * M**2 * P)/(A * (1 - M**2))) * dAdx(x,bl_h,bl_growth)
    term2 = -(((gamma * M**2 * P)/((1-M**2) * cp * T)) * dHtdx(x,dx,eta_total,combustion_end))
    term3  = -((gamma * M**2 * (1 + (gamma-1) * M**2))/(2 * (1 - M**2)))
    term4 = (((4 * Cf * (P/Dh(x,bl_h,bl_growth)))) - (2 * ((Vinj * P)/(mdot * V)) * (dmdotDX)))
    term5 = -(((2 * gamma * M**2 * (1 + ((gamma-1)/2) *M**2)*P)/((1-M**2)*mdot)) * (dmdotDX))
    return term1 + term2 + (term3 * term4) + term5



