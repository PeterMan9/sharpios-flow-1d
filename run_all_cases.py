from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys

cases = [
    "xReact_High_Scaling",
    "xReact_Mid_Scaling",
    "xReact_Low_Scaling"
]

def run_case(CaseName):
    run_cmd = [
        sys.executable,
        "HyperReact1D_2param.py",
        CaseName
    ]

    subprocess.run(run_cmd,check = True)

if __name__ == "__main__":

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_case,cases))