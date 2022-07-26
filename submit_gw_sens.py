import pycondor
import numpy as np
import pandas as pd
import argparse
import pwd
import os
import lxml.etree
from astropy.time import Time
import wget
import glob

parser = argparse.ArgumentParser(
    description='Submit script')
parser.add_argument(
    '--output', type=str,
    default='/data/user/jthwaites/gw_o4/sens_trials/',
    help="Where to store output")
parser.add_argument(
    '--skymap', type=str,
    default=None,
    help='skymap path to xml file, if using a prior')
parser.add_argument(
    '--version', type=str,
    default='v001p02',
    help='GFUOnline version and patch (default=v001p02)')
parser.add_argument(
    '--tw', type=float, default=1000., #[-1, +14]day: 1382400
    help='time window to use (in sec)')
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

if args.skymap is None:
    print('Running trials for point sources')
    code_file='/data/user/jthwaites/gw_o4/ps_sensitivity.py'
    mem=6000
else: 
    print('Running trials for given skymap')
    code_file='/data/user/jthwaites/gw_o4/prior_sensitivity.py'
    mem=8000

### Create Dagman to submit jobs to cluster    
job = pycondor.Job(
    'gw_sensitivity_jobs_prior',
    code_file,
    error=error,
    output=output,
    log=log,
    submit=submit,
    getenv=True,
    universe='vanilla',
    verbose=2,
    request_cpus=5,
    request_memory=mem,
    extra_lines=[
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT']
    )

#scan over decs (for point source)
if args.skymap is None:
    decs= np.linspace(-85,85,35)
    #decs=[-67.5, -45., -22.5, 0., 22.5, 45., 67.5]

    if int(args.tw) !=1000: out_folder = args.output+'point_source_2week/'
    else: out_folder = args.output+'point_source/'
    for dec in decs:
        sens_trials=glob.glob(f'{out_folder}ps_sens_{str(dec)}_trials_*.pkl')
        for i in range(200):
            if f'{out_folder}ps_sens_{str(dec)}_trials_{i}_2week.pkl' in sens_trials:
                continue
            #else:
            #    print(f'{out_folder}/ps_sens_{str(dec)}_trials_{i}_2week.pkl')
            job.add_arg('--dec %s --pid %s --output %s --tw %s'%(dec, i, out_folder, args.tw))

# for spatial prior map
else: 
    payload=open(args.skymap, 'rb').read()
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

    if int(args.tw) !=1000: out_folder = args.output+f'{name}_2week/'
    else: out_folder = f'{args.output}{name}/'

    for i in range(50):
        for j in range(10):
            job.add_arg('--skymap %s --pid %i --output %s --name %s --version %s --time %f --tw %f --seed %i --ntrials %i'
                        %(f'{args.output}{name}/{name}.fits.gz', i, out_folder, name, args.version, 
                        event_mjd, args.tw, j, 200))

if args.skymap is None:
    dag_name=f'gw_sens_ps_v{args.version[-1]}'
else: 
    dag_name=f'gw_sens_prior_{args.version[-1]}'

dagman = pycondor.Dagman(
    dag_name,
    submit=submit, verbose=2)
    
dagman.add_job(job)
dagman.build_submit()