import pycondor
import numpy as np
import pandas as pd
import argparse
import pwd
import os
import lxml.etree
from astropy.time import Time
import wget

parser = argparse.ArgumentParser(
    description='Submit script')
parser.add_argument(
    '--output', type=str,
    default='/data/user/jthwaites/gw_o4/sens_trials/',
    help="Where to store output")
parser.add_argument(
    '--skymap_path', type=str,
    default=None,
    help='skymap path, if using a prior')
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
    'gw_sensitivity_using_prior',
    #'./ps_sensitivity.py',
    './prior_sensitivity.py',
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

#scan over decs (for point source)
"""
decs=[-67.5,-45.,-22.5, 0., 22.5, 45., 67.5]
for dec in decs:
    for i in range(200):
        job.add_arg('--dec %s --pid %s --output %s' % (dec, i, args.output))
"""

# for spatial prior map
payload=open(args.skymap_path, 'rb').read()
root = lxml.etree.fromstring(payload) 
eventtime = root.find('.//ISOTime').text
event_mjd = Time(eventtime, format='isot').mjd

params = {elem.attrib['name']:
          elem.attrib['value']
          for elem in root.iterfind('.//Param')}
skymap = params['skymap_fits']

name = root.attrib['ivorn'].split('#')[1]
name = name.split('-')[0]

if not os.path.exists(args.output+name):
    os.mkdir(args.output+name)
skymap_path=wget.download(skymap, out=f'{args.output}{name}/{name}.fits.gz')

for i in range(200):
    #job.add_arg('--skymap %s --time %s --pid %s --output %s --name %s'
    #            %(f'{args.output}{name}/{name}.fits.gz',event_mjd,i, args.output, name))
    job.add_arg('--skymap %s --pid %s --output %s --name %s'
                %(f'{args.output}{name}/{name}.fits.gz',i, args.output, name))

dagman = pycondor.Dagman(
    'gw_dagman_sens',
    submit=submit, verbose=2)
    
dagman.add_job(job)
dagman.build_submit()