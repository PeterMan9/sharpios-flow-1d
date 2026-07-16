import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from scipy import stats

import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

import sys
from datetime import datetime
from pathlib import Path
import traceback

#Forward Model Set/Configs 
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
#MCMC functions
#using black box approach 

# wrapper for Op var. basically tells log_likelihood func that i am just putting in a double prec scalar and will
# output a double precision scalar 
# this is just so that i can easily pas my prior into my function 


@as_op(itypes=[pt.dscalar,pt.dscalar,pt.dscalar,pt.dscalar,pt.dscalar,pt.dvector],otypes=[pt.dscalar]) 
def log_likelihood(Cf_dnz,eta_total,combustion_end,throat_obstruction,bl_growth,true_PTPressure):    
    Cf_dnz = float(Cf_dnz)
    eta_total = float(eta_total)
    combustion_end = float(combustion_end)
    throat_obstruction = float(throat_obstruction)
    bl_growth = float(bl_growth)

    try:
        results = model.run(throat_obstruction,Cf_dnz,eta_total,combustion_end,bl_growth)

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
        results = model.run(True_throat_obstruction,True_Cf_dnz,True_eta_total,True_combustion_end,True_bl_growth)
        '''
        plt.plot(resultsAtCorrectScale["x"], resultsAtCorrectScale["Area"])
        plt.xlabel("X (m)")
        plt.ylabel("Area (m^2)")
        plt.title("Area vs X")
        plt.grid()
        plt.savefig("Area_vs_X.png")  # Saves to the remote workspace
        plt.close()

        plt.plot(resultsAtCorrectScale["x"], resultsAtCorrectScale["Mach"])
        plt.xlabel("X (m)")
        plt.ylabel("Mach Number")
        plt.title("Mach Number vs X")
        plt.grid()
        plt.savefig("Mach_vs_X.png")  # Saves to the remote workspace
        plt.close()
        '''
        rng = np.random.default_rng(42)


        true_PTPressure = results["PT_P"]
        pt_noise = rng.normal(0, 0.01 * true_PTPressure,true_PTPressure.shape)  
        true_noisy_PTPressure = true_PTPressure + pt_noise

        plt.plot(results["x"], results["pressure"] * 1e-6,'k-', label='FM Pressure (MPa)')
        plt.plot(results["PT_X"], true_noisy_PTPressure * 1e-6, 'ro', label='Noisy PT Pressure (MPa)')
        plt.plot(results["PT_X"], true_PTPressure * 1e-6, 'bo', label='True PT Pressure (MPa)')
        plt.xlabel("X (m)")
        plt.ylabel("Pressure (MPa)")
        plt.title("Pressure vs X")  
        plt.legend()
        plt.grid()
        plt.savefig("1per_PTPressure_noisy_notNoisy_vs_X.png")  # Saves to the remote workspace
        plt.close()
        return true_noisy_PTPressure
    except Exception as e:
        print(f"Failed to Gen True Values because of {e}")
        raise

#sensitivity testing for params 
def likelihoodPlotting(Cf_dnz, eta_total, combustion_end,throat_obstruction,bl_growth):

    logplist = []
    count = 0 
    
    MovingVar = combustion_end

    movingVar_grid = np.linspace(MovingVar - MovingVar * 0.1, MovingVar + MovingVar * 0.1, 100)

    True_Cf_dnz = Cf_dnz
    True_eta_Total = eta_total
    True_combustion_end = combustion_end
    True_throat_obst = throat_obstruction
    True_bl_growth = bl_growth
    true_PTPressure = generatingTrueValues(True_Cf_dnz, True_eta_Total,True_combustion_end,True_throat_obst,True_bl_growth)

    count = 0

    for movingvar in movingVar_grid:
        count += 1
        combustion_end = movingvar
        try:
            results = model.run(throat_obstruction,Cf_dnz,eta_total,combustion_end,bl_growth)


            Predicted_PTPressure = results["PT_P"]

            if Predicted_PTPressure.shape != true_PTPressure.shape:
                raise RuntimeError(f"PT shape mismatch at {movingvar}: "f"pred={Predicted_PTPressure.shape}, true={true_PTPressure.shape}")

            predicted_error = Predicted_PTPressure - true_PTPressure
            percent_uncertainty = 0.01
            sigma_i = percent_uncertainty * np.abs(true_PTPressure)

            log_prob = np.sum(stats.norm.logpdf(predicted_error,loc=0.0,scale=sigma_i))

            print(count, movingvar, log_prob)
            logplist.append(log_prob)

        except Exception as e:
            print(f"Failed at {movingvar} because of: {e}")
            traceback.print_exc()
            logplist.append(-np.inf)
        


    logp_array = np.array(logplist)
    movingVar_list = movingVar_grid[:len(logp_array)]
    
    plt.figure()
    plt.plot(movingVar_list,logp_array)
    plt.xlabel("Combustion End Values")
    plt.ylabel("Log Likelihood")
    plt.grid()
    plt.savefig("combustion_end.png")  # Saves to the remote workspace
    plt.close()
'''
if __name__ == "__main__":
    Cf_dnz = 0.006
    eta_total= 0.8
    combustion_end = preburner_length
    throat_obstruction = 0.30
    bl_growth = 10
    results = likelihoodPlotting(Cf_dnz, eta_total, combustion_end,throat_obstruction,bl_growth)
'''

#MCMC Model

def run_MCMC_case(caseConfig):

    param_names = caseConfig["Parameters"]
    
    set_True_Cf_dnz = caseConfig["True_Cf_dnz"]
    set_Cf_dnz_Prior_mu = caseConfig["Cf_dnz_Prior_mu"]
    set_Cf_dnz_Prior_sigma = caseConfig["Cf_dnz_Prior_sigma"]
    set_Cf_dnz_scale = caseConfig["Cf_dnz_Scale"]
    set_Cf_dnz_scaling = caseConfig["Cf_dnz_Scaling"]

    set_True_eta_Total = caseConfig["True_eta_Total"]
    set_eta_Total_Prior_mu = caseConfig["eta_Total_Prior_mu"]
    set_eta_Total_Prior_sigma = caseConfig["eta_Total_Prior_sigma"]
    set_eta_Total_scale = caseConfig["eta_Total_Scale"]
    set_eta_Total_scaling = caseConfig["eta_Total_Scaling"]

    set_True_combustion_end = caseConfig["True_combustion_end"]
    set_combustion_end_Prior_mu = caseConfig["combustion_end_Prior_mu"]
    set_combustion_end_Prior_sigma = caseConfig["combustion_end_Prior_sigma"]
    set_combustion_end_scale = caseConfig["combustion_end_Scale"]
    set_combustion_end_scaling = caseConfig["combustion_end_Scaling"]

    set_True_throat_obstruction = caseConfig["True_throat_obstruction"]
    set_throat_obstruction_Prior_mu = caseConfig["throat_obstruction_Prior_mu"]
    set_throat_obstruction_Prior_sigma = caseConfig["throat_obstruction_Prior_sigma"]
    set_throat_obstruction_scale = caseConfig["throat_obstruction_Scale"]
    set_throat_obstruction_scaling = caseConfig["throat_obstruction_Scaling"]

    set_True_bl_growth = caseConfig["True_bl_growth"]
    set_bl_growth_Prior_mu = caseConfig["bl_growth_Prior_mu"]
    set_bl_growth_Prior_sigma = caseConfig["bl_growth_Prior_sigma"]
    set_bl_growth_scale = caseConfig["bl_growth_Scale"]
    set_bl_growth_scaling = caseConfig["bl_growth_Scaling"]

    set_draws = caseConfig["Draws"]
    set_tune = caseConfig["Tune"]
    set_chains = caseConfig["Chains"]
    set_cores = caseConfig["Cores"]

    with pm.Model() as model:
      
            timestamp = datetime.now().strftime("%Y-%m_%d-%H-%M")

            run_label = (
                f"{timestamp}_"
                f"{caseConfig['Case_Name']}_"
                f"{param_names}"
            )
            nameofCase = caseConfig["Case_Name"]

            results_root = Path("MCMC Results")
            results_root.mkdir(exist_ok = True)

            case_folder = results_root / run_label
            case_folder.mkdir(exist_ok = True)
            
            priors_folder = case_folder / "Priors"
            priors_folder.mkdir(exist_ok = True)

            diagnostics_folder = case_folder / "Diagnostics"
            diagnostics_folder.mkdir(exist_ok = True)

            with open(case_folder/f"{nameofCase}_config.txt", "w") as f:
                for key,value in caseConfig.items():
                    f.write(f"{key}: {value}\n")


            true_PTPressure = generatingTrueValues(set_True_Cf_dnz,set_True_eta_Total,set_True_combustion_end,set_True_throat_obstruction,set_True_bl_growth)

            prior_Cf_dnz = pm.TruncatedNormal("Cf_dnz", mu=set_Cf_dnz_Prior_mu,  sigma=set_Cf_dnz_Prior_sigma,
                                              lower = 0,upper = 0.009,initval=set_Cf_dnz_Prior_mu,default_transform=None)
            prior_eta_Total = pm.TruncatedNormal("eta_Total", mu=set_eta_Total_Prior_mu, sigma = set_eta_Total_Prior_sigma, 
                                                 lower = 0, upper = 1,initval=set_eta_Total_Prior_mu, default_transform=None)
            prior_combustion_end = pm.TruncatedNormal("combustion_end", mu=set_combustion_end_Prior_mu, sigma = set_combustion_end_Prior_sigma, 
                                                      lower = combustion.x_react,upper = 0.625,initval=set_combustion_end_Prior_mu, default_transform=None)
            prior_throat_obstruction = pm.TruncatedNormal("throat_obstruction", mu=set_throat_obstruction_Prior_mu, sigma = set_throat_obstruction_Prior_sigma, 
                                                          lower = 0.05, upper = 0.25,initval=set_throat_obstruction_Prior_mu, default_transform=None)
            prior_bl_growth = pm.TruncatedNormal("bl_growth", mu=set_bl_growth_Prior_mu, sigma = set_bl_growth_Prior_sigma,
                                                  lower = 5, upper = 15, initval=set_bl_growth_Prior_mu, default_transform=None)

            log_like = log_likelihood(prior_Cf_dnz,prior_eta_Total,prior_combustion_end,prior_throat_obstruction,prior_bl_growth,
                                      pt.as_tensor_variable(true_PTPressure, dtype="float64"))

            prior_Cfdnz_samples = pm.draw(prior_Cf_dnz, 10000, random_seed=42)
            prior_eta_Total_samples = pm.draw(prior_eta_Total, 10000, random_seed=42)
            prior_combustion_end_samples = pm.draw(prior_combustion_end, 10000, random_seed=42)
            prior_throat_obstruction_samples = pm.draw(prior_throat_obstruction, 10000, random_seed=42)
            prior_bl_growth_samples = pm.draw(prior_bl_growth, 10000, random_seed=42)

            plt.hist(prior_Cfdnz_samples, bins=50, density=True)
            plt.xlabel("Cf Diverging Nozzle")
            plt.ylabel("Density")
            plt.savefig(priors_folder / f"Prior_Cf_dnz_{nameofCase}.png", dpi=200)
            plt.close()

            plt.hist(prior_eta_Total_samples, bins=50, density=True)
            plt.xlabel("Eta Total")
            plt.ylabel("Density")
            plt.savefig(priors_folder / f"Prior_eta_Total_{nameofCase}.png", dpi=200)
            plt.close()

            plt.hist(prior_combustion_end_samples, bins=50, density=True)
            plt.xlabel("Combustion End")
            plt.ylabel("Density")
            plt.savefig(priors_folder / f"Prior_combustion_end_{nameofCase}.png", dpi=200)
            plt.close()

            plt.hist(prior_throat_obstruction_samples, bins=50, density=True)
            plt.xlabel("Throat Obstruction")
            plt.ylabel("Density")
            plt.savefig(priors_folder / f"Prior_throat_obstruction_{nameofCase}.png", dpi=200)
            plt.close()

            plt.hist(prior_bl_growth_samples, bins=50, density=True)
            plt.xlabel("Bl Growth")
            plt.ylabel("Density")
            plt.savefig(priors_folder / f"Prior_bl_growth_{nameofCase}.png", dpi=200)
            plt.close()

            pm.Potential("Error Likelihood",log_like)

            step = pm.DEMetropolisZ(
                vars = [prior_Cf_dnz,prior_eta_Total,prior_combustion_end,prior_throat_obstruction,prior_bl_growth],
                S= np.array([set_Cf_dnz_scale,set_eta_Total_scale,set_combustion_end_scale,set_throat_obstruction_scale,set_bl_growth_scale]), 
                scaling = np.array([set_Cf_dnz_scaling,set_eta_Total_scaling,set_combustion_end_scaling,set_throat_obstruction_scaling,set_bl_growth_scaling]),  #Initial scale factor for how aggressive the sampler noise moves around 
                tune="lambda",
                tune_interval=100,
                tune_drop_fraction=0.9
            )

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

    cf_dNz_samples = trace.posterior["Cf_dnz"].values.flatten()
    eta_Total_samples = trace.posterior["eta_Total"].values.flatten()
    combustion_end_samples = trace.posterior["combustion_end"].values.flatten()
    throat_obstruction_samples = trace.posterior["throat_obstruction"].values.flatten()
    bl_growth_samples = trace.posterior["bl_growth"].values.flatten()

    with open(case_folder/ f"{nameofCase}_MCMC_Report.txt", "w") as f:
        f.write(f"Parameters Included in Model = {param_names}\n")

        f.write(f"{caseConfig['Case_Name']} Values \n")

        f.write(f"\nTrue Diverging Nozzle Cf = {set_True_Cf_dnz}\n")
        f.write(f"Diverging Nozzle Cf Prior Mean = {set_Cf_dnz_Prior_mu}\n")
        f.write(f"Diverging Nozzle Cf Prior Sigma = {set_Cf_dnz_Prior_sigma}\n")
        f.write(f"Diverging Nozzle Cf Scale = {set_Cf_dnz_scale}\n")
        f.write(f"Diverging Nozzle Cf Scaling = {set_Cf_dnz_scaling}\n")

        f.write(f"\nTrue Eta Total = {set_True_eta_Total}\n")
        f.write(f"Eta Total Prior Mean = {set_eta_Total_Prior_mu}\n")
        f.write(f"Eta Total Prior Sigma = {set_eta_Total_Prior_sigma}\n")
        f.write(f"Eta Total Scale = {set_eta_Total_scale}\n")
        f.write(f"Eta Total Scaling = {set_eta_Total_scaling}\n")

        f.write(f"\nTrue Combustion End = {set_True_combustion_end}\n")
        f.write(f"Combustion End Prior Mean = {set_combustion_end_Prior_mu}\n")
        f.write(f"Combustion End Prior Sigma = {set_combustion_end_Prior_sigma}\n")
        f.write(f"Combustion End Scale = {set_combustion_end_scale}\n")
        f.write(f"Combustion End Scaling = {set_combustion_end_scaling}\n")

        f.write(f"\nTrue Throat Obstruction Pecentage = {set_True_throat_obstruction}\n")
        f.write(f"Throat Obstruction Pecentage Prior Mean = {set_throat_obstruction_Prior_mu}\n")
        f.write(f"Throat Obstruction Pecentage Prior Sigma = {set_throat_obstruction_Prior_sigma}\n")
        f.write(f"Throat Obstruction Pecentage Scale = {set_throat_obstruction_scale}\n")
        f.write(f"Throat Obstruction Pecentage Scaling = {set_throat_obstruction_scaling}\n")

        f.write(f"\nTrue Boundary Layer Growth = {set_True_bl_growth}\n")
        f.write(f"Boundary Layer Growth Prior Mean = {set_bl_growth_Prior_mu}\n")
        f.write(f"Boundary Layer Growth Prior Sigma = {set_bl_growth_Prior_sigma}\n")
        f.write(f"Boundary Layer Growth Scale = {set_bl_growth_scale}\n")
        f.write(f"Boundary Layer Growth Scaling = {set_bl_growth_scaling}\n")
        
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

    param_labels = {
            "Cf_dnz": "Diverging Nozzle Friction Coefficient (Cf)",
            "eta_Total": "Combustion Efficiency (\u03b7_Total)",
            "combustion_end": "Combustion End Location",
            "throat_obstruction": "Throat Obstruction Pecentage",
            "bl_growth": "Boundary Layer Growth Scaling"

        }
    
    true_values = {
            "Cf_dnz": set_True_Cf_dnz,
            "eta_Total": set_True_eta_Total,
            "combustion_end" : set_True_combustion_end,
            "throat_obstruction" : set_True_throat_obstruction,
            "bl_growth" : set_True_bl_growth
        }

    

    for param in summary.index:
        label = param_labels.get(param, param)
        true_val = true_values.get(param)

        az.plot_trace(trace, var_names=[param])
        plt.suptitle(f"Trace Plot \u2014 {label} | {nameofCase}", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_Trace.png", dpi=200)
        plt.close()

        az.plot_dist(trace, var_names=[param])
        if true_val is not None:
            plt.axvline(true_val, color="red", linestyle="--", linewidth=1.5, label=f"True = {true_val}")
            plt.legend(fontsize=9)

        plt.xlabel(label, fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.title(f"Posterior Distribution \u2014 {label} | {nameofCase}", fontsize=11)
        plt.tight_layout()
        plt.savefig(case_folder / f"{param}_{nameofCase}_Posterior.png", dpi=200)
        plt.close()

        az.plot_autocorr(trace, var_names=[param])
        plt.suptitle(f"Autocorrelation \u2014 {label} | {nameofCase}", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_Autocorr.png", dpi=200)
        plt.close()

        az.plot_ess(trace, var_names=[param])
        plt.suptitle(f"ESS Plot \u2014 {label} | {nameofCase}", fontsize=11)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_Ess.png", dpi=200)
        plt.close()

        az.plot_rank(trace, var_names=[param])
        plt.suptitle(f"rank Plot \u2014 {label} | {nameofCase}", fontsize=11)
        plt.tight_layout()
        plt.savefig(diagnostics_folder / f"{param}_{nameofCase}_rank.png", dpi=200)
        plt.close()


    running_mean_cf_dNz = np.cumsum(cf_dNz_samples) / np.arange(1, len(cf_dNz_samples) + 1)
    running_mean_eta_Total = np.cumsum(eta_Total_samples) / np.arange(1, len(eta_Total_samples) + 1)
    running_mean_combustion_end = np.cumsum(combustion_end_samples) / np.arange(1, len(combustion_end_samples) + 1)
    running_mean_throat_obstruction = np.cumsum(throat_obstruction_samples) / np.arange(1, len(throat_obstruction_samples) + 1)
    running_mean_bl_growth = np.cumsum(bl_growth_samples) / np.arange(1, len(bl_growth_samples) + 1)

    fig, axs = plt.subplots(1, 5,figsize=(17, 4),constrained_layout=True)

    plots = [
        (running_mean_cf_dNz, set_True_Cf_dnz, "Cf_dnz", "Friction Coefficient", "steelblue"),
        (running_mean_throat_obstruction, set_True_throat_obstruction, "Throat Obstruction", "Throat Obstruction", "steelblue"),
        (running_mean_eta_Total, set_True_eta_Total, "η_Total", "Combustion Efficiency", "darkorange"),
        (running_mean_combustion_end, set_True_combustion_end, "combustion_end", "Combustion End Location", "darkorange"),
        (running_mean_bl_growth, set_True_bl_growth, "bl_growth", "Boundary Layer Growth", "darkorange"),]
    for ax, (running_mean, true_val, ylabel, title, color) in zip(axs, plots):
        ax.plot(running_mean, color=color, linewidth=1.4, label="Running mean")
        ax.axhline(true_val,color="red",linestyle="--",linewidth=1.2,label=f"True = {true_val:g}")

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Sample", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, frameon=True, loc="best")

    fig.suptitle(f"Running Means | {nameofCase}", fontsize=11)
    plt.savefig(case_folder / f"Combined_Running_mean_{nameofCase}.png",dpi=220,bbox_inches="tight")
    plt.close(fig)

    pair_vars = [
    "Cf_dnz",
    "eta_Total",
    "combustion_end",
    "throat_obstruction",
    "bl_growth",
    ]

    pair_samples = {
        var: trace.posterior[var].values.flatten()
        for var in pair_vars
    }

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
    
    case_name = sys.argv[1]

    parameters = {
        "Cf_dnz": {
            "true": 0.006,
            "prior_mu": 0.006 * 0.95,
            "prior_sigma": 0.006 * 0.05,
            "scale": 0.001,
            "scaling": 0.05,
        },
        "eta_Total": {
            "true": 0.8,
            "prior_mu": 0.8 * 0.95,
            "prior_sigma": 0.8 * 0.05,
            "scale": 0.1,
            "scaling": 0.5,
            },
        "combustion_end"  :{
            "true": 0.42,
            "prior_mu": 0.55,
            "prior_sigma": 0.55 * 0.175,
            "scale": 0.1,
            "scaling": 0.1,
        },
        "throat_obstruction" : {
            "true": 0.20,
            "prior_mu": 0.20 * 0.95,
            "prior_sigma": 0.20 * 0.05,
            "scale": 0.01,
            "scaling": 0.01,
        },
        "bl_growth":{
            "true": 10,
            "prior_mu": 10 * 0.95,
            "prior_sigma": 10 * 0.05,
            "scale": 1,
            "scaling": 0.01,
        }
    }
    
    all_cases = {    
        "5p_nData_highOT": {
            "Case_Name": "5p_nData_highOT",
            "Parameters": ["Cf_dnz", "eta_Total","combustion_end","throat_obstruction","bl_growth"],

            "Draws" : 750,
            "Tune" : 250,
            "Chains" : 12 ,
            "Cores" : 12
            }, 
    }

    for case_name, case_data in all_cases.items():
        for param in case_data["Parameters"]:
            if param in parameters:
                # Grabs the specific nested keys and writes them to your flat setup
                case_data[f"True_{param}"] = parameters[param]["true"]
                case_data[f"{param}_Prior_sigma"] = parameters[param]["prior_sigma"]
                case_data[f"{param}_Prior_mu"] = parameters[param]["prior_mu"]
                case_data[f"{param}_Scale"] = parameters[param]["scale"]
                case_data[f"{param}_Scaling"] = parameters[param]["scaling"]

        case = all_cases[case_name]

    run_MCMC_case(case)
