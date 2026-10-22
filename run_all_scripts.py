
# subprocess allows Python to launch and manage other programs/scripts.
# In this case, we use it to run each MCMC script as a separate Python process.
import subprocess

# sys provides information about the current Python environment.
# We use sys.executable to ensure each script runs using the same Python
# interpreter (your CFD conda environment) as this master script.
import sys

# Path provides tools for working with file and folder paths.
# It is generally easier and more reliable than manually combining strings.
from pathlib import Path

# datetime allows us to record when each MCMC script starts and finishes.
from datetime import datetime


# ============================================================
# 1. FIND THE MAIN PROJECT DIRECTORY
# ============================================================

# __file__ refers to the location of THIS Python script (run_all_MCMC.py).
#
# Path(__file__) converts that location into a Path object.
# .resolve() converts it into an absolute path.
# .parent gets the folder containing this Python script.
#
# For example, if run_all_MCMC.py is located at:
# /home/pe452646/HyperReact_UndefinabilityModel/HyperReact_Framework/
#
# root_dir will point to that directory.
#
# This assumes run_all_MCMC.py is saved inside HyperReact_Framework.
root_dir = Path(__file__).resolve().parent


# ============================================================
# 2. DEFINE WHICH SCRIPTS TO RUN
# ============================================================

# This is a Python list containing the relative paths to the
# five MCMC scripts.
#
# Scripts will execute in the order listed here.
#
# IMPORTANT:
# These scripts run SEQUENTIALLY, not simultaneously.
# The runner waits until one finishes or crashes before starting
# the next script.
#
# Each individual script can still run multiple PyMC chains in
# parallel, such as 12 chains using 12 CPU cores.
scripts = [
    "SubModels/BL_Growth_Posterior_Model.py",
    "SubModels/CombustionEndLocation_Posterior_Model.py",
    "SubModels/PercentObstruction_Model.py",
    "SubModels/SkinFriction_Model.py",
    "SubModels/TotalMixing_Efficiency_Model.py",
]


# ============================================================
# 3. CREATE A FOLDER TO STORE TERMINAL OUTPUT AND ERRORS
# ============================================================

# The / operator joins paths when used with Path objects.
#
# This constructs:
# HyperReact_Framework/MCMC Results/Run_Logs/
#
# This folder will contain a separate .log file for each run.
# A log file is a text file containing the script's printed output
# and any error traceback messages.
log_dir = root_dir / "MCMC Results" / "Run_Logs"

# Create the log directory if it does not already exist.
#
# parents=True:
#   Also creates any missing parent folders.
#
# exist_ok=True:
#   Does not raise an error if the directory already exists.
log_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. LOOP THROUGH THE FIVE SCRIPTS
# ============================================================

# The for loop takes each entry from the scripts list.
#
# First iteration:
# script = "SubModels/BL_Growth_Posterior_Model.py"
#
# Second iteration:
# script = "SubModels/CombustionEndLocation_Posterior_Model.py"
#
# And so on until all five scripts have been attempted.
for script in scripts:

    # Construct the absolute path to the current MCMC script.
    #
    # Example:
    # /home/.../HyperReact_Framework/SubModels/BL_Growth_Posterior_Model.py
    script_path = root_dir / script

    # Record the current date and time.
    # We will use this to calculate the execution duration.
    start = datetime.now()

    # Print a separator containing 60 "=" characters.
    #
    # "\n" adds a blank line before the separator.
    # "="*60 repeats the "=" character 60 times.
    #
    # flush=True forces Python to immediately display the output
    # instead of potentially buffering it.
    print(f"\n{'='*60}", flush=True)

    # Display which script is starting.
    print(f"STARTING: {script}", flush=True)

    # Display its start time.
    print(f"TIME: {start}", flush=True)


    # ========================================================
    # 5. CREATE A UNIQUE LOG FILE FOR THIS SCRIPT
    # ========================================================

    # script_path.stem returns the filename without its extension.
    #
    # Example:
    # "BL_Growth_Posterior_Model.py"
    # becomes:
    # "BL_Growth_Posterior_Model"
    #
    # {start:%Y%m%d_%H%M%S} formats the start time:
    #
    # %Y = four-digit year
    # %m = month
    # %d = day
    # %H = hour (24-hour format)
    # %M = minute
    # %S = second
    #
    # Example log filename:
    # BL_Growth_Posterior_Model_20261020_233000.log
    #
    # The timestamp reduces the chance of overwriting logs from
    # earlier runs.
    log_file = log_dir / f"{script_path.stem}_{start:%Y%m%d_%H%M%S}.log"


    # ========================================================
    # 6. EXECUTE THE CURRENT MCMC SCRIPT
    # ========================================================

    # try/except is Python's error-handling mechanism.
    #
    # Python attempts to execute the code inside "try".
    #
    # If a Python exception occurs in this runner (for example,
    # the script file cannot be opened), execution jumps to
    # the "except" block instead of terminating the runner.
    try:

        # Open the log file for writing.
        #
        # "w" means write mode:
        #   Creates the file if it does not exist.
        #   Overwrites it if it already exists.
        #
        # "as log" gives us a variable representing the open file.
        #
        # The "with" statement automatically closes the file
        # when execution leaves this block, even if an exception
        # occurs.
        with open(log_file, "w") as log:

            # subprocess.run() launches an external program
            # and WAITS until that program finishes.
            #
            # Here, the external program is another Python script.
            #
            # The process running that script is separate from
            # the process running this master script.
            #
            # This separation is useful because a Python exception
            # or normal process-level crash in a child script
            # generally does not crash the master runner.
            result = subprocess.run(

                # Command to execute:
                #
                # sys.executable:
                #   Path to the Python interpreter currently running
                #   this master script.
                #
                #   For your setup, it should point to something like:
                #   /home/pe452646/anaconda3/envs/CFD/bin/python
                #
                # "-u":
                #   Runs Python in unbuffered mode.
                #   Printed output is written to the log promptly.
                #
                # str(script_path):
                #   Converts the Path object to a string containing
                #   the absolute location of the MCMC script.
                #
                # Equivalent terminal command:
                # python -u SubModels/BL_Growth_Posterior_Model.py
                [sys.executable, "-u", str(script_path)],

                # Set the working directory of the child process.
                #
                # This matters because your MCMC scripts use
                # relative paths like:
                #
                # Path("MCMC Results")
                #
                # By setting cwd=root_dir, those paths refer to:
                # HyperReact_Framework/MCMC Results/
                #
                # Without cwd, relative paths could point somewhere
                # different depending on where the runner was launched.
                cwd=root_dir,

                # Redirect the script's standard output to the log file.
                #
                # Standard output (stdout) includes output from:
                # print()
                # Progress messages
                # General program information
                #
                # These messages go into the log instead of
                # appearing in the runner's terminal.
                stdout=log,

                # stderr means standard error.
                #
                # It normally contains Python exception tracebacks,
                # warnings, and error messages.
                #
                # subprocess.STDOUT redirects stderr to the same
                # destination as stdout.
                #
                # Consequently, both normal output and errors
                # are written to the SAME log file.
                stderr=subprocess.STDOUT,

                # check=False means subprocess.run() does not raise
                # an exception just because the child script exits
                # with a nonzero exit code.
                #
                # This is important for your goal:
                # Even if an MCMC script fails, the runner can move
                # on to the next script.
                #
                # If check=True, a nonzero exit code would instead
                # raise a CalledProcessError.
                check=False,
            )

        # subprocess.run() returns a CompletedProcess object.
        #
        # result.returncode tells us how the child process ended.
        #
        # 0 means the script exited successfully.
        # Nonzero generally indicates an error or abnormal exit.
        #
        # IMPORTANT:
        # A return code of 0 does not guarantee that the MCMC
        # results are scientifically valid. It only means the
        # script exited without reporting a process-level failure.
        if result.returncode == 0:

            print(f"COMPLETED: {script}", flush=True)

        else:

            # The child script failed or exited abnormally.
            #
            # The master runner does not stop.
            # It records the failure and continues to the next script.
            print(f"FAILED: {script}", flush=True)

            # Display the exit code for debugging.
            #
            # Examples:
            # 1 = common exit code for an unhandled Python exception
            # -9 = process terminated by SIGKILL on Linux
            #
            # A negative return code on Linux generally indicates
            # termination by the corresponding signal.
            print(f"Exit code: {result.returncode}", flush=True)


    # ========================================================
    # 7. HANDLE ERRORS IN THE MASTER RUNNER
    # ========================================================

    # This block catches Python exceptions raised while the
    # master runner is trying to launch or manage the subprocess.
    #
    # Examples include:
    # - Missing script file
    # - Permission errors
    # - Failure to create a subprocess
    #
    # Note:
    # Python exceptions occurring INSIDE an MCMC child script
    # are normally handled as a nonzero subprocess return code,
    # not by this except block.
    except Exception as e:

        # Display the error message.
        #
        # The runner then continues beyond this block and
        # eventually attempts the next script.
        print(f"ERROR running {script}: {e}", flush=True)


    # ========================================================
    # 8. RECORD THE EXECUTION TIME
    # ========================================================

    # Record the time after the script has exited or failed.
    end = datetime.now()

    # Subtracting two datetime objects produces a timedelta.
    #
    # Example:
    # 3:25:17.123456
    #
    # Meaning:
    # 3 hours, 25 minutes, and approximately 17 seconds.
    #
    # This includes script startup, MCMC sampling, result saving,
    # plotting, and any other operations performed by the script.
    print(f"Duration: {end - start}", flush=True)

    # Display the location of the log file.
    #
    # If the MCMC script crashed, open this file to inspect
    # the traceback and determine what caused the failure.
    print(f"Log saved: {log_file}", flush=True)


# ============================================================
# 9. ALL SCRIPTS HAVE BEEN ATTEMPTED
# ============================================================

# This statement executes after the for loop completes.
#
# It does not necessarily mean all five scripts succeeded.
# It means the runner has attempted each script in the list.
print("\nAll five MCMC scripts attempted.", flush=True)
