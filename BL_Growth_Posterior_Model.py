import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from scipy import stats

import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

import sys
import warnings
from datetime import datetime
from pathlib import Path
import traceback
#Forward Model Set/Configs 

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
    tube_area=7 *(45e-3 * 45e-3),
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


def bl_growthFunc(A,B,C,Re,M,T):
    return A * Re + B * M + C * T
#MCMC functions
#using black box approach 

# wrapper for Op var. basically tells log_likelihood func that i am just putting in a double prec scalar and will
# output a double precision scalar 
# this is just so that i can easily pas my prior into my function 
@as_op(itypes=[pt.dscalar,pt.dscalar,pt.dscalar,pt.dvector,pt.dvector,pt.dvector],otypes=[pt.dscalar]) 
def log_likelihood(A,B,C,true_PTPressure,true_param_values,data_vector):    
    A = float(A)
    B = float(B)
    C = float(C)
    #true_param_values consists of these params in this order[cf, eta_total, combustion_end, throat_obstruction, bl_growth]
    #data_vector is the Re,M,T generated from the true run of the forward model 

    try:
        #the order of inputs for this func is: throat_obstruction,Cf_dnz,eta_total,combustion_end,bl_growth
        #i am building my own bl_growth based off of A,B,C, but the others will stay the same 
        bl_growth = bl_growthFunc(A,B,C,data_vector[0],data_vector[1],data_vector[2])
        results = forward_model.run(true_param_values[3],true_param_values[0],true_param_values[1],true_param_values[2],bl_growth)

        Predicted_PTPressure = results["PT_P"]

        if Predicted_PTPressure.shape != true_PTPressure.shape:
            raise RuntimeError(
                f"PT shape mismatch: pred={Predicted_PTPressure.shape}, "
                f"true={true_PTPressure.shape}, PT_X={results.get('PT_X')}"
            )
        
        predicted_error = Predicted_PTPressure - true_PTPressure
        percent_uncertainty = 0.01 
        sigma_i = np.sqrt((percent_uncertainty * true_PTPressure)**2) 

        log_prob = np.sum(stats.norm.logpdf(predicted_error,loc = 0.0,scale = sigma_i))

        return np.array(log_prob, dtype=np.float64)
    
    except Exception as e:
            print(f"Failed because of: {e}")
            traceback.print_exc()
            return np.array(-np.inf, dtype=np.float64)
    

def generatingTrueValues(True_Cf_dnz,True_eta_total,True_combustion_end,True_throat_obstruction,True_bl_growth):
    try:
        results = forward_model.run(True_throat_obstruction,True_Cf_dnz,True_eta_total,True_combustion_end,True_bl_growth)
        data_vector = results["data_vector"]
        rng = np.random.default_rng(42)
        true_PTPressure = results["PT_P"]
        pt_noise = rng.normal(0, 0.01 * true_PTPressure,true_PTPressure.shape)  
        true_noisy_PTPressure = true_PTPressure #+ pt_noise

        return true_noisy_PTPressure,data_vector
    except Exception as e:
        print(f"Failed to Gen True Values because of {e}")
        raise

#MCMC Model

def run_MCMC_case(case, parameters, true_values):

    param_names = ["A", "B", "C"]


    #setting true values for my 5 uncertain params so that I can mimic and create a true/real pt data set 
    set_True_Cf_dnz = true_values[0]
    set_True_eta_Total = true_values[1]
    set_True_combustion_end = true_values[2]
    set_True_throat_obstruction = true_values[3]
    set_True_bl_growth = true_values[4]
    true_param_values = true_values

    #setting up prior mean, sgima and then respective scale and scaling for MCMC Sampler 
    set_A_Prior_mu = parameters["A"]["prior_mu"]
    set_A_Prior_sigma = parameters["A"]["prior_sigma"]
    set_A_scale = parameters["A"]["scale"]
    set_A_scaling = parameters["A"]["scaling"]

    set_B_Prior_mu = parameters["B"]["prior_mu"]
    set_B_Prior_sigma = parameters["B"]["prior_sigma"]
    set_B_scale = parameters["B"]["scale"]
    set_B_scaling = parameters["B"]["scaling"]

    set_C_Prior_mu = parameters["C"]["prior_mu"]
    set_C_Prior_sigma = parameters["C"]["prior_sigma"]
    set_C_scale = parameters["C"]["scale"]
    set_C_scaling = parameters["C"]["scaling"]

    set_draws = case["Draws"]
    set_tune = case["Tune"]
    set_chains = case["Chains"]
    set_cores = case["Cores"]

    #initializing pymc model
    with pm.Model() as model:

            #getting timestap for unique run label and creating folders to save results 
            timestamp = datetime.now().strftime("%Y-%m_%d-%H-%M")

            run_label = (
                f"{timestamp}_"
                f"{case['Case_Name']}_"
                f"{param_names}"
            )
            nameofCase = case["Case_Name"]

            results_root = Path("MCMC Results")
            results_root.mkdir(exist_ok = True)

            case_folder = results_root / run_label
            case_folder.mkdir(exist_ok = True)
            
            priors_folder = case_folder / "Priors"
            priors_folder.mkdir(exist_ok = True)

            diagnostics_folder = case_folder / "Diagnostics"
            diagnostics_folder.mkdir(exist_ok = True)

            with open(case_folder/f"{nameofCase}_config.txt", "w") as f:
                for key,value in case.items():
                    f.write(f"{key}: {value}\n")

            #generating my TRUE pressure data. also getting my data vector for my emperical bl growth function
            true_PTPressure,data_vector = generatingTrueValues(set_True_Cf_dnz,set_True_eta_Total,set_True_combustion_end,set_True_throat_obstruction,set_True_bl_growth)

            # Setting up prior distributions for the three coeff A, B, and C.
            # PyMC will propose/sample values of these random variables during MCMC.
            rv_PriorA = pm.Normal("A", mu=set_A_Prior_mu,  sigma=set_A_Prior_sigma,
                                              initval=set_A_Prior_mu,default_transform=None)
            rv_PriorB = pm.Normal("B", mu=set_B_Prior_mu, sigma = set_B_Prior_sigma, 
                                                 initval=set_B_Prior_mu, default_transform=None)
            rv_PriorC = pm.Normal("C", mu=set_C_Prior_mu, sigma = set_C_Prior_sigma, 
                                                      initval=set_C_Prior_mu, default_transform=None)
            
            #getting the log likelihood of my model given the random scalors taken from the priors vs the true data 
            log_like = log_likelihood(rv_PriorA,rv_PriorB,rv_PriorC,
                                      pt.as_tensor_variable(true_PTPressure, dtype="float64"),
                                      pt.as_tensor_variable(true_param_values, dtype="float64"),
                                      pt.as_tensor_variable(data_vector, dtype="float64"),)
            #adds the custom likelihood to PyMC's model log-probability. 
            #So this is what pymc will use to evaluate the posterior distribution of the parameters A, B, and C.
            pm.Potential("likelihood", log_like)

            #just drawing some samples from the priors that i can 
            prior_A_samples = pm.draw(rv_PriorA, 10000, random_seed=42)
            prior_B_samples = pm.draw(rv_PriorB, 10000, random_seed=42)
            prior_C_samples = pm.draw(rv_PriorC, 10000, random_seed=42)

            plt.hist(prior_A_samples, bins=50, density=True)
            plt.xlabel("Coefficient A")
            plt.ylabel("Density")
            plt.savefig(priors_folder / f"Prior_Coefficient_A_{nameofCase}.png", dpi=200)
            plt.close()

            plt.hist(prior_B_samples, bins=50, density=True)
            plt.xlabel("Coefficient B")
            plt.ylabel("Density")
            plt.savefig(priors_folder / f"Prior_Coefficient_B_{nameofCase}.png", dpi=200)
            plt.close()

            plt.hist(prior_C_samples, bins=50, density=True)
            plt.xlabel("Coefficient C")
            plt.ylabel("Density")
            plt.savefig(priors_folder / f"Prior_Coefficient_C_{nameofCase}.png", dpi=200)
            plt.close()

          #this is the sampler that will be used to sample from the posterior distribution of the parameters A, B, and C.
            step = pm.DEMetropolisZ(
                vars = [rv_PriorA,rv_PriorB,rv_PriorC], #random variables to sample from
                S= np.array([set_A_scale,set_B_scale,set_C_scale]), #initial scale factor 
                scaling = np.array([set_A_scaling,set_B_scaling,set_C_scaling]),  #Initial scale factor for how aggressive the sampler noise moves around 
                tune="scaling",
                tune_interval=100,
                tune_drop_fraction=0.9
            )
            #this is where the actual sampling from the posterior distribution happens.
            #the posterior dist is established by the 1st instance of the model and the log likelihood function that was defined earlier. 
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

    #writing a txt file with all the base settings for the mcmc run and some arviz summary stats and acceptance rate 
    with open(case_folder/ f"{nameofCase}_MCMC_Report.txt", "w") as f:
        f.write(f"Parameters Included in Model = {param_names}\n")

        f.write(f"{case['Case_Name']} Values \n")

        f.write(f"True Values for uncertain parameters used to mimic real gathered data \n")
        f.write(f"True Diverging Nozzle Cf = {set_True_Cf_dnz}\n")
        f.write(f"True Eta Total = {set_True_eta_Total}\n")
        f.write(f"True Combustion End = {set_True_combustion_end}\n")
        f.write(f"True Throat Obstruction Pecentage = {set_True_throat_obstruction}\n")
        f.write(f"True Boundary Layer Growth = {set_True_bl_growth}\n")

        f.write(f"\nCoefficent A Prior Mean = {set_A_Prior_mu}\n")
        f.write(f"Coefficent A Prior Sigma = {set_A_Prior_sigma}\n")
        f.write(f"Coefficent A Scale = {set_A_scale}\n")
        f.write(f"Coefficent A Scaling = {set_A_scaling}\n")

        f.write(f"\nCoefficent B Prior Mean = {set_B_Prior_mu}\n")
        f.write(f"Coefficent B Prior Sigma = {set_B_Prior_sigma}\n")
        f.write(f"Coefficent B Scale = {set_B_scale}\n")
        f.write(f"Coefficent B Scaling = {set_B_scaling}\n")

        f.write(f"\nCoefficent C Prior Mean = {set_C_Prior_mu}\n")
        f.write(f"Coefficent C Prior Sigma = {set_C_Prior_sigma}\n")
        f.write(f"Coefficent C Scale = {set_C_scale}\n")
        f.write(f"Coefficent C Scaling = {set_C_scaling}\n")
        
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

    #getting the posterior samples for A, B, and C from the trace object.
    #and then flattening them into 1D arrays for further analysis or plotting. so this means that if it is 4 chains and 1000 draws,
    #then the flattened array will have 4000 samples for each parameter.
    A_samples = trace.posterior["A"].values.flatten()
    B_samples = trace.posterior["B"].values.flatten()
    C_samples = trace.posterior["C"].values.flatten()
    #calculating the boundary layer growth ussing the posterior samples
    bl_growth_samples = bl_growthFunc(A_samples,B_samples,C_samples,data_vector[0],data_vector[1],data_vector[2])

  
    
    true_values = {
            "Cf_dnz": set_True_Cf_dnz,
            "eta_Total": set_True_eta_Total,
            "combustion_end" : set_True_combustion_end,
            "throat_obstruction" : set_True_throat_obstruction,
            "bl_growth" : set_True_bl_growth
        }

    #labels for parameters to be used in plots - its just there so that i dont have to manually make new plots 
    param_labels = {
            "A": "Coefficient A",
            "B": "Coefficient B",
            "C": "Coefficient C",}
    #for loop for plotting
    for param in summary.index:
        label = param_labels.get(param, param)
        true_val = true_values.get(param)

        #plotting the trace - this is the sampled values of the parameter over the course of the MCMC sampling iterations.
        az.plot_trace(trace, var_names=[param])
        plt.suptitle(f"Trace Plot \u2014 {label} | {nameofCase}", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_Trace.png", dpi=200)
        plt.close()

        #Plotting the posterior dists 
        az.plot_dist(trace, var_names=[param])
        plt.xlabel(label, fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.title(f"Posterior Distribution \u2014 {label} | {nameofCase}", fontsize=11)
        plt.tight_layout()
        plt.savefig(case_folder / f"{param}_{nameofCase}_Posterior.png", dpi=200)
        plt.close()

        # plotting the autocorrelation of the sampled values of the parameter. 
        # This shows me how correlated (dependant) the samples are with each other.
        az.plot_autocorr(trace, var_names=[param])
        plt.suptitle(f"Autocorrelation \u2014 {label} | {nameofCase}", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_Autocorr.png", dpi=200)
        plt.close()

        # plotting the effective sample size (ESS) of the sampled values of the parameter. 
        # This shows me how many independent samples I have effectively obtained from the MCMC sampling process.
        az.plot_ess(trace, var_names=[param])
        plt.suptitle(f"ESS Plot \u2014 {label} | {nameofCase}", fontsize=11)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_Ess.png", dpi=200)
        plt.close()

        #plotting the rank of the sampled values of the parameter.
        # this just shows how well the sampler explored the parameter space 
        az.plot_rank(trace, var_names=[param])
        plt.suptitle(f"rank Plot \u2014 {label} | {nameofCase}", fontsize=11)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_rank.png", dpi=200)
        plt.close()

    pair_vars = ["A","B", "C"]

    pair_samples = {
        var: trace.posterior[var].values.flatten()
        for var in pair_vars}

    n = len(pair_vars)
    fig, axs = plt.subplots(n, n, figsize=(11, 11))

    for i, yvar in enumerate(pair_vars):
        for j, xvar in enumerate(pair_vars):
            ax = axs[i, j]

            x = pair_samples[xvar]
            y = pair_samples[yvar]

            if i == j:
                ax.hist(x, bins=45, color="steelblue", alpha=0.85, density=True)
            elif i > j:
                ax.hexbin(x,y,gridsize=35,cmap="viridis",mincnt=1,linewidths=0.0,)
            else:
                ax.axis("off")

            if i == n - 1:
                ax.set_xlabel(xvar, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(yvar, fontsize=8)
            elif j != 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(False)

    fig.suptitle(f"Joint Posterior Density | {nameofCase}", fontsize=13, y=0.995)
    fig.tight_layout()
    plt.savefig(case_folder / f"PairPlot_{nameofCase}.png",dpi=220,bbox_inches="tight",)
    plt.close(fig)

#cases and running model 
if __name__ == "__main__":
    

    parameters = {
        
        "A":{
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "scale": 0.1,
            "scaling": 0.001},
        "B":{
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "scale": 0.1,
                "scaling": 0.001},
        "C":{
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "scale": 0.1,
                "scaling": 0.001},

    }
    
    case = {
        "Case_Name": "Boundary_Layer_Growth_submodel_Test",
        "Draws": 10,
        "Tune": 10,
        "Chains": 1,
        "Cores": 1,
    }
    

    set_True_Cf_dnz =  0.006
    set_True_eta_Total = 0.8
    set_True_combustion_end = geometry.tube_length*0.6
    set_True_throat_obstruction = 0.01
    set_True_bl_growth = 1.2
    true_values = np.array([set_True_Cf_dnz,set_True_eta_Total,set_True_combustion_end,set_True_throat_obstruction,set_True_bl_growth],dtype=np.float64)
    set_A_Prior_mu = parameters["A"]["prior_mu"]
    set_A_Prior_sigma = parameters["A"]["prior_sigma"]
    set_A_scale = parameters["A"]["scale"]
    set_A_scaling = parameters["A"]["scaling"]
   
    # B prior / sampler settings
    set_B_Prior_mu = parameters["B"]["prior_mu"]
    set_B_Prior_sigma = parameters["B"]["prior_sigma"]
    set_B_scale = parameters["B"]["scale"]
    set_B_scaling = parameters["B"]["scaling"]

    # C prior / sampler settings
    set_C_Prior_mu = parameters["C"]["prior_mu"]
    set_C_Prior_sigma = parameters["C"]["prior_sigma"]
    set_C_scale = parameters["C"]["scale"]
    set_C_scaling = parameters["C"]["scaling"]

    run_MCMC_case(case,parameters,true_values)
