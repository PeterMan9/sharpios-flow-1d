import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

#Forward Model Set/Configs 
from forward_model import (
    ForwardModel,
    ModelConfig,
    ConstantAreaGeometry,
    ConstantAreaInletConditions,
    SmartsModel,
)

config = ModelConfig(
    geometry_type="constant_area",
    friction=True,
    boundary_layer=True,
    combustion=False,
)

geometry = ConstantAreaGeometry(
    tube_area=1 *(45e-3 * 45e-3),
    tube_length=1,
    x_injLocation=1 * 0.15,
)

setup_gas = ct.Solution("h2_air.yaml")
#just getting all the properties for a given T,P,Y
def gas_properties(T:float, P:float, Y:float) -> float:
    setup_gas.TPY = T, P, Y
    cp = setup_gas.cp_mass
    h = setup_gas.enthalpy_mass
    gamma = setup_gas.cp_mass/setup_gas.cv_mass
    R_specific = setup_gas.cp_mass - setup_gas.cv_mass
    s = setup_gas.entropy_mass
    return{
        "cp": cp,
        "h": h,
        "gamma": gamma,
        "R_specific": R_specific,
        "s": s
    }

def get_Y(comp_string: str) -> float:
    setup_gas.TPX = 300, 101325, comp_string
    return setup_gas.Y.copy()

#solving for the mass fraction of the mixed gasses 
def Ymix(YA:float,mdotAir:float, YB:float, mdotH2:float) -> float:
    Ymix = (mdotAir * YA + mdotH2 * YB)/(mdotAir + mdotH2)
    return Ymix

T_air = 300.0
T_H2 = 300.0

P_air = 7.708339e6
P_H2 = 8.315077e6

M1 = 0.999
M2 = 0.999
M3 = 0.999

mdot_Air = 0.4430
mdot_H2 = 0.003

Y_air = get_Y("O2:0.21, N2:0.79")
Y_H2 = get_Y("H2:1.0")

Y_mix = Ymix(Y_air,mdot_Air,Y_H2,mdot_H2,)

air_properties = gas_properties(T_air,P_air,Y_air,)
h2_properties = gas_properties(T_H2,P_H2,Y_H2,)

mix_properties = gas_properties(300.0,ct.one_atm,Y_mix,)

air_inlet_gamma = air_properties["gamma"]
h2_inlet_gamma = h2_properties["gamma"]
R_mix = mix_properties["R_specific"]
Tstag_air = T_air * (1.0+ (air_inlet_gamma - 1.0) / 2.0 * M1**2)

inlet = ConstantAreaInletConditions(
    dir_air=0.229 / 39.37,
    d_h2=0.034 / 39.37,
    P_air=P_air,
    P_H2=P_H2,
    T_air=T_air,
    T_H2=T_H2,
    M1=M1,
    M2=M2,
    M3=M3,
    mdot_Air=mdot_Air,
    mdot_H2=mdot_H2,
    injMdot=0.0,
    Vinj=0.0,
    air_inletGamma=air_inlet_gamma,
    h2_inletGamma=h2_inlet_gamma,
    Y_mix=Y_mix,
    R_mix=R_mix,
    TstagAir=Tstag_air,)

combustion = SmartsModel(
    hpr_h2=120e6, #J/kg
    fst=0.029,
    phi=0.2306,
    theta=1.2,
    x_react=0.0,
)

forward_model = ForwardModel(
    config=config,
    geometry_case=geometry,
    inlet_conditions=inlet,
    combustion_model=combustion,
    mechanism="h2_air.yaml"
)


True_Cf_dNz = 0.006
True_eta_Total = 0.8
True_combustion_end = geometry.tube_length*0.6
True_precent_obstruction = 0.01
True_bl_growth = 1.2

results = forward_model.run(True_precent_obstruction,True_Cf_dNz,True_eta_Total,True_combustion_end,True_bl_growth)


plt.plot(results["x"], results["Area"])
plt.xlabel("X")
plt.ylabel("Area (m^2)")
plt.savefig("x vs area for tube")
plt.close()

plt.plot(results["x"], results["Mach"])
plt.xlabel("X")
plt.ylabel("Mach")
plt.savefig("x vs Mach for tube")
plt.close()

plt.plot(results["x"], results["entropy"])
plt.xlabel("X")
plt.ylabel("entropy")
plt.savefig("x vs entropy for tube")
plt.close()

