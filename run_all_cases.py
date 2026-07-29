from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys

cases = [
    "5p_nData_Long",
]

def run_case(CaseName):
    run_cmd = [
        sys.executable,
        "HyperReact1D_MCMC_Model.py",
        CaseName
    ]

    subprocess.run(run_cmd,check = True)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(run_case,cases))

'''
Remove Cf pb because it does not do much if anything at all 
remove combustion start

parameters to add:

add a boundary layer thing 
    so in the conv section of the nozzle add a constant 1mm growth and until the throat so that you have like an throat ubstruction of sort and thats what i can add as a parameter 
        so maybe like a target % of the throat ubstructed 
add combustion end 
    i will have combustion end at the preburner but it has the ability to end after or before the preburner end.
    The limit will be the throat 

add friction post the throat 

look into post throat boundary layer growth and add that as a parameter 



    
'''