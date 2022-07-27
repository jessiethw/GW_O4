#!/usr/bin/env python

"""
This script runs trials to calculate point source sensitivity flux.
This script will inject one value of inject flux and record passing
fraction. Meant to be run in parallel with more jobs at different values
of injected flux.
Written by Raamis, updated by Jessie Thwaites
Updated July 2022
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
#p.add_argument("--nsMax", default=2e-15, type=float,
#                help="Maximum flux to inject (default=2e-15)")
p.add_argument("--nsMax", default=10, type=float,
                help="Maximum signal events to inject (default=10)")
p.add_argument("--pid", default=0, type=int,
                help="Process ID number for jobs running on cluster (Default=0)")
p.add_argument("--ntrials", default=1000, type=int,
                help="Number of trials to run per injected flux (Default=1000)")
p.add_argument("--nstep", default=16, type=int,
                help="Step size to increment injected flux (Default=16)")
p.add_argument("--dec", default=0., type=float,
                help="Declination of point source to test (Default=0.0)")
p.add_argument("--ra", default=0., type=float,
                help="RA of point source to test (Default=0.0)")
p.add_argument("--output", default='./',type=str,
                help="path to save output")
args = p.parse_args()
###################################################################

############## CONFIGURE LLH AND INJECTOR ################
seasons = ['GFUOnline_v001p03','IC86, 2011-2018']
erange  = [0,10]
index = 2.
GW_time = 57982.52852350
#Location of host galaxy of GW170817: Ra: 197.4458, dec: -23.3844
src_ra = np.radians(args.ra)
src_dec = np.radians(args.dec)
time_window = 500./3600./24. #500 seconds in days

llh = config(seasons,gamma=index,ncpu=2,seed=args.pid+1, days=5,
              time_mask=[time_window,GW_time], poisson=True)

inj = PointSourceInjector(E0=1000.)
inj.fill(src_dec,llh.exp,llh.mc,llh.livetime,
         temporal_model=llh.temporal_model)
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

    ni, sample = inj.sample(src_ra,ns,poisson=True)
    val = llh.scan(src_ra,src_dec,scramble=True,seed=j,inject=sample,
                   time_mask=[time_window,GW_time])
    
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

with open(args.output+'/ps_sens_%s_trials_%s.pkl'%(str(args.dec), args.pid), 'wb') as f:
    pickle.dump(results, f)
