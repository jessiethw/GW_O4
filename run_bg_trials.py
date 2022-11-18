#!/usr/bin/env python

"""
This script runs background trials for a GW.
Written by Jessie Thwaites
Nov 2022
"""

import numpy  as np
import argparse
import healpy as hp
from fast_response.GWFollowup import GWFollowup
from skylab.priors          import SpatialPrior
from fast_response.FastResponseAnalysis import PointSourceFollowup
from astropy.time import Time

import sys
if '/data/user/jthwaites/gw_o4' not in sys.path:
    sys.path.append('/data/user/jthwaites/gw_o4')
#from config_GW            import config
import pickle


##################### CONFIGURE ARGUMENTS ########################
p = argparse.ArgumentParser(description="Calculates Sensitivity and Discovery"
                            " Potential Fluxes for Background Gravitational wave/Neutrino Coincidence study",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--pid", default=0, type=int,
                help="Process ID number for jobs running on cluster (Default=0)")
p.add_argument("--ntrials", default=1000, type=int,
                help="Number of trials to run per injected flux (Default=1000)")
p.add_argument("--skymap", default=None, type=str,
                help="path/url for skymap, if using a skymap")
p.add_argument("--dec", default=0.0, type=float,
                help='dec to run trials at, if using point source')
p.add_argument("--time", default=57982.52852350, type=float,
                help="Time of GW merger event (Default=time of GW170817)")
p.add_argument("--output", default='./',type=str,
                help="path to save output")
p.add_argument("--name", default='ps test',type=str,
                help="name of gw event")
p.add_argument("--ncpu", default=5, type=int,
                help='Number of cpus to request')
p.add_argument("--version", default="v001p02", type=str,
                help='GFU version and patch number (default = v001p02)')
p.add_argument("--nside", default=256, type=int,
                help='nside of map to use')
p.add_argument("--tw", default=1000., type=float,
                help='Time window to use, in sec (default 500)')
p.add_argument('--seed', default=0, type=int,
                help='ensure unique seed for trials')
args = p.parse_args()
###################################################################

############## CONFIGURE LLH AND INJECTOR ################
GW_time = args.time

delta_t = args.tw
time_window = delta_t/2./3600./24. #500 seconds in days
gw_time = Time(args.time, format='mjd')
start_time = gw_time - (delta_t / 86400. / 2.)
stop_time = gw_time + (delta_t / 86400. / 2.)
start = start_time.iso
stop = stop_time.iso

name = args.name
name = name.replace('_', ' ')

if args.skymap is not None:
    f = GWFollowup(args.name, args.skymap, start, stop, save=False)
    f._allow_neg = False
    spatial_prior = SpatialPrior(f.skymap, containment = f._containment, allow_neg=f._allow_neg)
else:
    f= PointSourceFollowup(args.name, 0.0, args.dec, start, stop, save=False)
llh=f.llh
nside=f.nside

### Set range of for loop to guarantee unique seeds 
### for every job on the cluster
stop = args.ntrials * (args.pid+1+args.seed)
start = stop-(args.ntrials*args.seed)

##need to unify naming, make sure saving correct things
TS_list=[]
ns_fit=[]
gamma_fit=[]

print('starting trials')
for j in range(start,stop):
    if args.skymap is not None:
        val = llh.scan(0.0,0.0, scramble = True, seed = j, spatial_prior=spatial_prior,
                       time_mask=[time_window,GW_time], pixel_scan=[nside,3.])
        key='TS_spatial_prior_0'
    else: 
        val = llh.scan(0.0, args.dec, scramble = True, seed = j,
                       time_mask=[time_window,GW_time], pixel_scan=[nside,3.])
        key='TS'
    try:
        maxLoc = np.argmax(val[key])  #pick out max of all likelihood ratios at diff pixels
    except ValueError:
        continue

    TS_list.append(val[key].max())
    ns_fit.append(val['nsignal'][maxLoc])
    gamma_fit.append(val['gamma'][maxLoc])

print('done with trials')

results={
    'TS_List':TS_list,
    'ns_fit':ns_fit,
    'gamma_fit':gamma_fit
}
print('saving everything')
if int(args.tw) !=1000.: suffix='_2week'
else: suffix=''

if args.name is not None:
    with open(args.output+f'/{args.version}_{args.name}_prior_bg_trials_{args.pid}{suffix}.pkl', 'wb') as f:
        pickle.dump(results, f)
else: 
    with open(args.output+f'/{args.version}_prior_bg_trials_{args.pid}{suffix}.pkl', 'wb') as f:
        pickle.dump(results, f)
