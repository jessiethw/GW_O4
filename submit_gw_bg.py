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
    default='/data/user/jthwaites/gw_o4/bg_trials/',
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

### Create Dagman to submit jobs to cluster    
job = pycondor.Job(
    'gw_sensitivity_jobs_prior',
    './run_bg_trials.py',
    error=error,
    output=output,
    log=log,
    submit=submit,
    getenv=True,
    universe='vanilla',
    verbose=2,
    request_cpus=5,
    request_memory=6000,
    extra_lines=[
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT']
    )

#scan over decs (for point source)
if args.skymap is None:
    decs= np.linspace(-85,85,35)
    #decs=[-67.5, -45., -22.5, 0., 22.5, 45., 67.5]

    for dec in decs:
        #if not os.path.exists(args.output+f'ps_map_dec_{dec}.fits'):
        #    import healpy as hp
        #    ps_map = np.full_like([0]*hp.nside2npix(256), 1e-20, dtype=np.float32)
        #    ps_map[hp.pixelfunc.ang2pix(256, 0.5 * np.pi - np.deg2rad(dec), np.deg2rad(0), nest=True)] = 0.999999
        #    hp.fitsfunc.write_map(args.output+f'ps_map_dec_{dec}_nested.fits', ps_map, overwrite=True, nest=True)

        if int(args.tw) !=1000: 
            out_folder = args.output+'point_source_2week/'
            #for i in range(100):
            #    if os.path.exists(out_folder+f'/v001p02_bg_trials_dec_{dec}_{i}_2week.pkl'):
            #        continue
            #    job.add_arg('--skymap %s --output %s --name %s --tw %f --dec %f --pid %i --ntrials 100'
            #        %(args.output+f'ps_map_dec_{dec}_nested.fits', out_folder, 'ps',
            #          args.tw, dec, i))
        else: 
            out_folder = args.output+'point_source/'
            #job.add_arg('--skymap %s --output %s --name %s --tw %f --dec %f'
            #        %(args.output+f'ps_map_dec_{dec}_nested.fits', out_folder, 'ps',
            #          args.tw, dec))
        for i in range(10):
            job.add_arg('--output %s --name %s --tw %f --dec %f --pid %i'
                    %(out_folder, 'ps', args.tw, dec, i+1))

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

    if int(args.tw) !=1000: 
        out_folder = args.output+f'{name}_2week/'
        for i in range(1, 31):
            for j in range(10):
                job.add_arg('--skymap %s --pid %i --output %s --name %s --time %f --tw %f --seed %i --ntrials 100'
                        %(f'{args.output}{name}/{name}.fits.gz', i, out_folder, name, 
                        event_mjd, args.tw, j))
    else: 
        out_folder = f'{args.output}{name}/'
        job.add_arg('--skymap %s --output %s --name %s --time %f --tw %f'
                    %(f'{args.output}{name}/{name}.fits.gz', out_folder, name, 
                    event_mjd, args.tw))

if args.skymap is None:
    dag_name=f'gw_bg_ps_v{args.version[-1]}'
else: 
    dag_name=f'gw_bg_prior_v{args.version[-1]}'

dagman = pycondor.Dagman(
    dag_name,
    submit=submit, verbose=2)
    
dagman.add_job(job)
dagman.build_submit()