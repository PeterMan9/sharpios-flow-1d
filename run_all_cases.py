from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys

cases = [
    "Case_1",
    "Case_2",
    "Case_3"
]

def run_case(CaseName):
    run_cmd = [
        sys.executable,
        "HyperReact1D_2param.py",
        CaseName
    ]

    subprocess.run(run_cmd,check = True)

if __name__ == "__main__":

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_case,cases))