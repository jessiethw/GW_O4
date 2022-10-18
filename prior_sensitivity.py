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
import healpy as hp

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
p.add_argument("--ncpu", default=5, type=int,
                help='Number of cpus to request')
p.add_argument("--version", default="v001p02", type=str,
                help='GFU version and patch number (default = v001p02)')
p.add_argument("--nside", default=256, type=int,
                help='nside of map to use')
args = p.parse_args()
###################################################################

############## CONFIGURE LLH AND INJECTOR ################
seasons = [f'GFUOnline_{args.version}','IC86, 2011-2019']
erange  = [0,10]
index = 2.
GW_time = args.time
skymap, skymap_header = hp.read_map(args.skymap, h=True, verbose=False)
nside=hp.pixelfunc.get_nside(skymap)
skymap = hp.pixelfunc.ud_grade(skymap,nside_out=args.nside,power=-2)
nside=args.nside

spatial_prior = SpatialPrior(skymap, allow_neg=False)

time_window = 500./3600./24. #500 seconds in days

print('going into config')
llh, inj = config(seasons,gamma=index,seed=args.pid+1, days=5,
            spatial_prior=spatial_prior, ncpu=args.ncpu,
            time_mask=[time_window,GW_time], poisson=True)
print('done with config llh/inj')
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

##need to unify naming, make sure saving correct things
ndisc = 0
TS_list=[]
ns_fit=[]
ns_inj=[]
gamma_fit=[]
flux_inj=[]
flux_fit=[]
print('starting trials')
for j in range(start,stop):
    ni, sample = inj.sample(mean_signal=ns,poisson=False)

    if sample is not None:
        sample['time']=GW_time
        ns_inj.append(sample.size)
        flux_inj.append(inj.mu2flux(sample.size))
    else: 
        ns_inj.append(0)
        flux_inj.append(0.)

    val = llh.scan(0.0,0.0, scramble = True, seed = j,spatial_prior=spatial_prior, 
                   inject = sample,time_mask=[time_window,GW_time], pixel_scan=[nside,3.])
                   
    maxLoc = np.argmax(val['TS_spatial_prior_0'])###pick out max of all likelihood ratios at diff pixels
    if val['TS_spatial_prior_0'].max() > 0.:
        ndisc+=1
    
    ns_fit.append(val['nsignal'][maxLoc])##get corresponding fitted number of signals
    flux_fit.append(inj.mu2flux(val['nsignal'][maxLoc]))
    gamma_fit.append(val['gamma'][0])

print('done with trials')
P = float(ndisc)/ntrials
if P==0.:
    P=0.0001
if P==1.:
    P=0.9999

results={
    'passFrac':[P],
    'TS_List':TS_list,
    'ns_fit':ns_fit,
    'ns_inj':ns_inj,
    'gamma_fit':gamma_fit,
    'flux_inj':flux_inj,
    'flux_fit':flux_fit
}
print('saving everything')
if args.name is not None:
    with open(args.output+f'/{args.version}_{args.name}_prior_sens_trials_{args.pid}.pkl', 'wb') as f:
        pickle.dump(results, f)
else: 
    with open(args.output+f'/{args.version}_prior_sens_trials_{args.pid}.pkl', 'wb') as f:
        pickle.dump(results, f)
