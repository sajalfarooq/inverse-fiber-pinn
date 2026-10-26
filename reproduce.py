
"""Reproduce this standalone study; previous output directories are archived."""
from pathlib import Path
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from optics_portfolio.profiles import PROFILES
from optics_portfolio.output import result_dir,save_json
import run
import study

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",choices=list(PROFILES),default="standard")
    p.add_argument("--run-tag")
    a=p.parse_args()
    os.environ["OPTICS_PROFILE"]=a.profile
    if a.run_tag: os.environ["OPTICS_RUN_TAG"]=a.run_tag
    result_dir(ROOT)
    run.main(PROFILES[a.profile]["pinn_steps"])
    study.main(a.profile)
    save_json(result_dir(ROOT)/"run_config.json",{"profile":a.profile,"budget":PROFILES[a.profile],"run_tag":a.run_tag,"data":"synthetic"})
