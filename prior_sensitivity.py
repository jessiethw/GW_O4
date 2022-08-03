#!/usr/bin/env python

"""
This script runs trials to calculate sensitivity flux for a given GW map.
This script will inject one value of inject flux and record passing
fraction. Meant to be run in parallel with more jobs at different values
of injected flux.
Written by Jessie Thwaites
August 2022
"""

import numpy  as np
import argparse

from skylab.priors        import SpatialPrior
from skylab.ps_injector   import PriorInjector, PointSourceInjector

import sys
if '/data/user/jthwaites/gw_o4' not in sys.path:
    sys.path.append('/data/user/jthwaites/gw_o4')
from config_GW            import config
from scipy.optimize       import curve_fit
from scipy.stats          import chi2
import pickle


##################### CONFIGURE ARGUMENTS ########################
p = argparse.ArgumentParser(description="Calculates Sensitivity and Discovery"
                            " Potential Fluxes for Background Gravitational wave/Neutrino Coincidence study",
                            formatter_class=argparse.RawTextHelpFormatter)

p.add_argument("--nsMin", default=0.0, type=float,
                help="Minimum flux to inject (default=0.0)")
p.add_argument("--nsMax", default=10, type=float,
                help="Maximum signal events to inject (default=10)")
p.add_argument("--pid", default=0, type=int,
                help="Process ID number for jobs running on cluster (Default=0)")
p.add_argument("--ntrials", default=1000, type=int,
                help="Number of trials to run per injected flux (Default=1000)")
p.add_argument("--nstep", default=16, type=int,
                help="Step size to increment injected flux (Default=16)")
p.add_argument("--skymap", required=True, type=str,
                help="path/url for skymap to use (required)")
p.add_argument("--time", default=57982.52852350, type=float,
                help="Time of GW merger event (Default=time of GW170817)")
p.add_argument("--output", default='./',type=str,
                help="path to save output")
p.add_argument("--name", default=None,type=str,
                help="name of gw event")
args = p.parse_args()
###################################################################

############## CONFIGURE LLH AND INJECTOR ################
seasons = ['GFUOnline_v001p03','IC86, 2011-2018']
erange  = [0,10]
index = 2.
GW_time = args.time

spatial_prior = SpatialPrior(args.skymap, allow_neg=False)

time_window = 500./3600./24. #500 seconds in days

llh, inj = config(seasons,gamma=index,ncpu=2,seed=args.pid+1, days=5,
            spatial_prior=spatial_prior,
            time_mask=[time_window,GW_time], poisson=True)

###########################################################

######### CALCULATE DISCOVERY POTENTIAL ###############
### step through values of injected flux 
ns_min = args.nsMin
ns_max = args.nsMax
nstep  = args.nstep
ntrials = args.ntrials
delta = (ns_max - ns_min)/nstep

### Set range of for loop to guarantee unique seeds 
### for every job on the cluster
stop = ntrials * (args.pid+1)
start = stop-ntrials

### flux to be injected
ns = ns_min + delta*args.pid
flux = inj.mu2flux(ns)
fluxList = [flux]

ndisc = 0
TS_list=[]
ns_list=[]

for j in range(start,stop):

    ni, sample = inj.sample(spatial_prior,ns,poisson=True)

    val = llh.scan(0.0, 0.0, scramble = True, seed=j, 
            spatial_prior=spatial_prior,inject=sample,
            time_mask = [time_window,GW_time])
    
    TS_list.append(val['TS'])
    ns_list.append(val['nsignal'])

    if val['TS']>0.0:
        ndisc+=1


P = float(ndisc)/ntrials
if P==0.:
    P=0.0001
if P==1.:
    P=0.9999

results={
    'passFrac':[P],
    'fluxList':fluxList,
    'TS_List':TS_list,
    'ns_fit':ns_list,
    'ns_inj':ns
}

if args.name is not None:
    with open(args.output+'/%s_prior_sens_trials_%s.pkl'%(args.name, args.pid), 'wb') as f:
        pickle.dump(results, f)
else: 
    with open(args.output+'/prior_sens_trials_%s.pkl'%(args.pid), 'wb') as f:
        pickle.dump(results, f)
