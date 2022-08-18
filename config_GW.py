r"""
Config File which sets up likelihood object and PriorInjector object.
Modified version of Josh Wood's multi-year config file which can be found 
here: skylab/doc/analyses/icecube-170922A_wGFU/config.py 
"""


import os
import numpy as np 

from   skylab.ps_llh            import PointSourceLLH, MultiPointSourceLLH
from   skylab.ps_injector       import PriorInjector
from   skylab.temporal_models   import TemporalModel,BoxProfile
from   skylab.llh_models        import EnergyLLH
from   skylab.datasets          import Datasets
from   skylab.utils             import times
from   skylab.test_statistics   import TestStatisticNegativeSlope

###############################################################################

###########
# GLOBALS #
###########

# energy units
GeV = 1
TeV = 1000*GeV

###############################################################################

def config(seasons, seed = 1, scramble = True, e_range=(0,np.inf), g_range=[1.,4.],
           gamma = 2.0, E0 = 1*TeV, dec = None, remove = False,ncpu=20,
           spatial_prior = None, time_mask=None, poisson=True,days=None,timescramble=True):
  r""" Configure point source likelihood and injector. 

  Parameters
  ----------
  season : single season name

  seed : int
    Seed for random number generator

  Returns
  -------
  llh : PointSourceLLH
    Point source likelihood object
  inj : PriorInjector
     Point source injector object
  """

  #print("Scramble is %s" % str(scramble))

  # setup likelihoods
  sample = seasons[0]
  name = seasons[1]
  exp, mc, livetime = Datasets[sample].season(name)
  grl = Datasets[sample].grl(name)
  
  #exp_18, mc_18, livetime_18 = Datasets['GFUOnline'].season('IC86, 2018')
  #grl_18 = Datasets['GFUOnline'].grl('IC86, 2018')

  #exp = np.concatenate((exp,exp_18))
  #livetime+=livetime_18
  #grl = np.concatenate((grl,grl_18))
  
  sinDec_bins = Datasets[sample].sinDec_bins(name)
  energy_bins = Datasets[sample].energy_bins(name)

  # Add floor to ang. uncertainty
  floor = np.deg2rad(0.2)
  exp['sigma'][exp['sigma']<floor] = floor
  mc['sigma'][mc['sigma']<floor] = floor

  llh_model = EnergyLLH(twodim_bins = [energy_bins, sinDec_bins], allow_empty=True,
                        bounds = g_range,seed = 2.0,ncpu=ncpu)

  tstart = time_mask[1]-time_mask[0]; tend = time_mask[1]+time_mask[0]
  box = TemporalModel(grl=grl,poisson_llh=poisson,
                      signal=BoxProfile(start=tstart,stop=tend),days=days) 
  
  llh = PointSourceLLH(exp,mc,livetime,scramble=scramble,timescramble=timescramble,
                       llh_model=llh_model,temporal_model=box,ncpu=ncpu,
                       nsource_bounds=(0., 1e4))
  llh._warn_nsignal_max=False 
  # save a little RAM by removing items copied into LLHs
  del exp, mc

  #######
  # LLH #
  #######

  # return only llh if no spatial prior specified
  if spatial_prior is None: return llh

  #############################################################################

  ############
  # INJECTOR #
  ############
  
  inj = PriorInjector(spatial_prior, gamma=gamma, E0=1000., seed=seed,e_range=(e_range[0],e_range[1]))
  inj.fill(llh.exp, llh.mc, llh.livetime, temporal_model=llh.temporal_model) 
  
  ############
  # INJECTOR #
  ############

  #############################################################################

  return llh, inj

###############################################################################

