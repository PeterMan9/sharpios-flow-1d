from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys

cases = [
    "Case_1",
    "Case_2",
    "Case_3",
    "Case_4"
]

def run_case(CaseName):
    run_cmd = [
        sys.executable,
        "Sharpios1dFlowP4.py",
        CaseName
    ]

    subprocess.run(run_cmd,check = True)

if __name__ == "__main__":

    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(run_case,cases)