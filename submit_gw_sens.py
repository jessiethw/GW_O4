import pycondor
import numpy as np
import pandas as pd
import argparse
import pwd
import os

parser = argparse.ArgumentParser(
    description='Submit script')
parser.add_argument(
    '--output', type=str,
    default='/data/user/jthwaites/gw_o4/sens_trials/',
    help="Where to store output"
)

args = parser.parse_args()
username = pwd.getpwuid(os.getuid())[0]
if not os.path.exists(f'/scratch/{username}/'):
    os.mkdir(f'/scratch/{username}/')
if not os.path.exists(f'/scratch/{username}/gw/'):
    os.mkdir(f'/scratch/{username}/gw/')
if not os.path.exists(f'/scratch/{username}/gw/condor/'):
    os.mkdir(f'/scratch/{username}/gw/condor')

error = f'/scratch/{username}/gw/condor/error'
output = f'/scratch/{username}/gw/condor/output'
log = f'/scratch/{username}/gw/condor/log'
submit = f'/scratch/{username}/gw/condor/submit'

### Create Dagman to submit jobs to cluster    
job = pycondor.Job(
    'sensitivity_gw',
    './ps_sensitivity.py',
    error=error,
    output=output,
    log=log,
    submit=submit,
    getenv=True,
    universe='vanilla',
    verbose=2,
    request_memory=6000,
    extra_lines=[
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT']
    )


#gw_list = [150914,151012,151226,170104,170608,170809,170814,170818,170823]

#scan over decs
decs=[-67.5,-45.,-22.5, 0., 22.5, 45., 67.5]
for dec in decs:
    for i in range(200):
        job.add_arg('--dec %s --pid %s --output %s' % (dec, i, args.output))
    #job.add_arg('--dec %s --output %s' % (dec, args.output))

dagman = pycondor.Dagman(
    'gw_dagman_sens',
    submit=submit, verbose=2)
    
dagman.add_job(job)
dagman.build_submit()