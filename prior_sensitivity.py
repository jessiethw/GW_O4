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
from fast_response.GWFollowup import GWFollowup
from astropy.time import Time

#from skylab.priors        import SpatialPrior
#from skylab.ps_injector   import PriorInjector, PointSourceInjector

import sys
if '/data/user/jthwaites/gw_o4' not in sys.path:
    sys.path.append('/data/user/jthwaites/gw_o4')
#from config_GW            import config
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
GW_time = args.time

delta_t = 1000.
time_window = delta_t/2./3600./24. #500 seconds in days
gw_time = Time(args.time, format='mjd')
start_time = gw_time - (delta_t / 86400. / 2.)
stop_time = gw_time + (delta_t / 86400. / 2.)
start = start_time.iso
stop = stop_time.iso

name = args.name
name = name.replace('_', ' ')

f = GWFollowup(args.name, args.skymap, start, stop)
f._allow_neg = args.allow_neg_ts
spatial_prior = f.skymap
llh=f.llh
nside=f.nside

f.initialize_injector()
inj=f.inj

print('Initialized LLH/injector')
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
    
    TS_list.append(val['TS_spatial_prior_0']).max()
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
