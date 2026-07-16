import numpy as np
import matplotlib.pyplot as plt

from forward_model import (
    ForwardModel,
    ModelConfig,
    WindTunnelGeometry,
    WindTunnelInletConditions,
    SmartsModel,
)

config = ModelConfig(
    geometry_type="wind_tunnel",
    friction=True,
    boundary_layer=True,
    combustion=True,
)

geometry = WindTunnelGeometry(
    preburner_area=45e-3 * 45e-3,
    preburner_length=0.42,
    nozzle_area_ratio=25,
    conv_Nozzle_length=0.08,
    div_Nozzle_length=0.140,
    exit_Area=45e-3 * 45e-3,
    x_injLocation=0.42 * 0.15,
)

inlet = WindTunnelInletConditions(
    dir_air=0.229/39.37, #meters
    d_h2=0.034/39.37,
    P_air=7.708339*1e6, #Pa
    P_H2=8.315077*1e6,
    T_air=300,
    T_H2=300,
    M1=0.999,
    M2=0.999,
    M3=0.999,
    mdot_Air=0.4430,
    mdot_H2=0.003,
    injMdot=0,
    Vinj=0,
)

combustion = SmartsModel(
    hpr_h2=120e6, #J/kg
    fst=0.029,
    phi=0.2306,
    theta=1.2,
    x_react=0.0,
)

model = ForwardModel(
    config=config,
    geometry_case=geometry,
    inlet_conditions=inlet,
    combustion_model=combustion,
)

True_Cf_dNz = 0.006
True_eta_Total = 0.8
True_combustion_end = geometry.preburner_length
True_throat_obstruction = 0.2
True_bl_growth = 10

results = model.run(True_throat_obstruction,True_Cf_dNz,True_eta_Total,True_combustion_end,True_bl_growth)

plt.plot(results["x"], results["entropy"])
plt.xlabel("X (m)")
plt.ylabel("entropy Number")
plt.title("entropy Number vs X")
plt.grid()
plt.savefig("entropy_vs_X.png")
plt.close()

