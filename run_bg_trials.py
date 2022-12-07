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

import sys
if '/data/user/jthwaites/gw_o4' not in sys.path:
    sys.path.append('/data/user/jthwaites/gw_o4')
from config_GW            import config
import pickle


##################### CONFIGURE ARGUMENTS ########################
p = argparse.ArgumentParser(description="Calculates bg trials for GW",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--pid", default=0, type=int,
                help="Process ID number for jobs running on cluster (Default=0)")
p.add_argument("--ntrials", default=1000, type=int,
                help="Number of trials to run per injected flux (Default=1000)")
#GWFollowup must have a skymap: use a map with all probability in one pixel if point source-like
p.add_argument("--skymap", default=None, type=str,
                help="path/url for skymap")
p.add_argument("--dec", default=None, type=float,
                help='dec if using point source')
p.add_argument("--time", default=57982.52852350, type=float,
                help="Time of GW merger event (Default=time of GW170817)")
p.add_argument("--output", default='./',type=str,
                help="path to save output")
p.add_argument("--name", default='ps',type=str,
                help="name of gw event")
#p.add_argument("--ncpu", default=5, type=int,
#                help='Number of cpus to request')
#p.add_argument("--version", default="v001p02", type=str,
#                help='GFU version and patch number (default = v001p02)')
p.add_argument("--nside", default=256, type=int,
                help='nside of map to use')
p.add_argument("--tw", default=1000., type=float,
                help='Time window to use, in sec (default 500)')
p.add_argument('--seed', default=1, type=int,
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
start_iso = start_time.iso
stop_iso = stop_time.iso

name = args.name
name = name.replace('_', ' ')

### Set range of for loop to guarantee unique seeds 
### for every job on the cluster
stop = args.ntrials * (args.pid+1+args.seed)
start = stop-(args.ntrials*(args.seed))
if args.dec is not None:
    seasons = ['GFUOnline_v001p02','IC86, 2011-2018']
    erange  = [0,10]
    index = 2.
    GW_time = 57982.52852350
    llh = config(seasons,gamma=index,ncpu=2,seed=args.pid+1, days=5,
            time_mask=[time_window,GW_time], poisson=True)
    TS_list=[]

    for j in range(start,stop):
        val = llh.scan(0.,np.deg2rad(args.dec),scramble=True,seed=j,
                   time_mask=[time_window,GW_time])
        TS_list.append(val['TS'])
    suffix=''

elif int(args.tw) != 1000:
    f = GWFollowup(args.name+'_test', args.skymap, start_iso, stop_iso, save=False)
    f._allow_neg = False
    spatial_prior = SpatialPrior(f.skymap, containment = f._containment, allow_neg=f._allow_neg)
    llh=f.llh
    nside=f.nside

    TS_list=[]
    for j in range(start,stop):
        if (j+1)%10==0: print(j/args.ntrials)
        val = llh.scan(0.0,0.0, scramble = True, seed = j, spatial_prior=spatial_prior,
                   time_mask=[time_window,GW_time], pixel_scan=[nside,3.])
        try:
            TS_list.append(val['TS_spatial_prior_0'].max())
        except ValueError:
            TS_list.append(0.)
    #f.run_background_trials(ntrials=args.ntrials)
    #TS_List = f.tsd
    suffix=f'_2week'
    
else: 
    f = GWFollowup(args.name+'_test', args.skymap, start_iso, stop_iso, save=False)
    f._allow_neg = False
    spatial_prior = SpatialPrior(f.skymap, containment = f._containment, allow_neg=f._allow_neg)
    llh=f.llh
    nside=f.nside

    TS_list = f.reload_background_trials()
    suffix=''
results={'TS_List':TS_list}

if args.dec is None:
    suffix+=f'_{args.seed}'

if args.dec is not None:
    with open(args.output+f'/v001p02_bg_trials_dec_{args.dec}_{args.pid}{suffix}.pkl', 'wb') as f:
        pickle.dump(results, f)
elif args.name is not None:
    with open(args.output+f'/v001p02_{args.name}_bg_trials_{args.pid}{suffix}.pkl', 'wb') as f:
        pickle.dump(results, f)
else: 
    with open(args.output+f'/v001p02_bg_trials_{args.pid}{suffix}.pkl', 'wb') as f:
        pickle.dump(results, f)
