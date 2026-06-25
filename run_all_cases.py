from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys

cases = [
    "4_Param_Base_Case",
]

def run_case(CaseName):
    run_cmd = [
        sys.executable,
        "HyperReact1D_Multiparam.py",
        CaseName
    ]

    subprocess.run(run_cmd,check = True)

if __name__ == "__main__":

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_case,cases))