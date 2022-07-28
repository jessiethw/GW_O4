"""
Reads in previously calculated signal trials 
and calculates a sensitivity flux from passing fraction.
Plots point source sensitivity over a range of decs
and produces a bias plot. 
Written by Jessie Thwaites
July 2022
"""  
import numpy  as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from scipy.optimize       import curve_fit
from scipy.stats          import chi2
import pickle

from skylab.ps_injector   import PointSourceInjector
import sys
if '/data/user/jthwaites/gw_o4' not in sys.path:
    sys.path.append('/data/user/jthwaites/gw_o4')
from config_GW            import config

fontsize=15

def chi2Fit(x, A):
    return chi2.cdf(x, A)

sensitivity_flux=[]
sensitivity_ns=[]
decs=[-67.5, -45., -22.5, 0., 22.5, 45., 67.5]

for dec in decs:
    sens_trials=[f'./sens_trials/ps_sens_{str(dec)}_trials_{str(pid)}.pkl' for pid in range(0,200)]

    passing_frac=[]
    ns_fit_mean=[]
    ns_fit_1sigma=[]
    ns_inj=[]
    for i in range(len(sens_trials)):
        with open(sens_trials[i], 'rb') as f:
            result=pickle.load(f)
            passing_frac.append(result['passFrac'][0])
            ns_fit_mean.append(np.mean(result['ns_fit']))
            ns_fit_1sigma.append(np.std(result['ns_fit']))
            ns_inj.append(result['ns_inj'])

    #### Calculate sensitivity #####
    parameters, covariance = curve_fit(chi2Fit, ns_inj[:16], passing_frac[:16])
    n_inj_range=np.arange(0., ns_inj[16], 0.1)
    fit_chi2=chi2.cdf(n_inj_range, parameters[0])
    idx = np.where(abs(fit_chi2 - 0.9)==min(abs(fit_chi2 - 0.9)))[0][0]
    sensitivity=n_inj_range[idx]
    print('Sensitivity at dec = %.1f: %.2f events'%(dec, sensitivity))
    llh = config(['GFUOnline_v001p03','IC86, 2011-2018'],gamma=2.,ncpu=2, days=5,
              time_mask=[500./3600./24.,57982.52852350], poisson=True)

    inj = PointSourceInjector(E0=1000.)
    inj.fill(np.deg2rad(dec),llh.exp,llh.mc,llh.livetime,
         temporal_model=llh.temporal_model)
    sensitivity_ns.append(sensitivity)
    #flux at E0=1 TeV, convert to [GeV cm^-2 s]
    sensitivity_flux.append(inj.mu2flux(sensitivity)*1e9)

    #### Making passing fract curve #####
    mpl.rcParams.update({'font.size':fontsize})
    fig,ax = plt.subplots(figsize = (10,6))
    ax.tick_params(labelsize=fontsize)

    plt.plot(ns_inj[:16], passing_frac[:16], 'o', label='passing fraction')
    plt.plot(n_inj_range, fit_chi2, label='Chi2CDF, dof=%.2f'%parameters[0])
    plt.plot(n_inj_range,[0.9]*len(n_inj_range), '--', color='k')
    plt.plot([sensitivity,sensitivity],[0.,1.],'--', color='k')

    plt.xlabel(r'n_inj')
    plt.ylabel('Fraction TS > threshold')
    plt.title(f'Passing fraction for point source, dec = {str(dec)}')
    plt.legend(loc=0)

    plt.savefig(f'./plots/passing_frac_dec{str(dec)}.png')

    ### Make bias plot ###
    plt.clf()
    plt.plot(ns_inj,ns_fit_mean)
    plt.plot(ns_inj,ns_inj,'--', color='C0')
    plt.fill_between(ns_inj, 
                     [ns_fit_mean[i]-ns_fit_1sigma[i] for i in range(len(ns_fit_mean))], 
                     [ns_fit_mean[i]+ns_fit_1sigma[i] for i in range(len(ns_fit_mean))], 
                     alpha=0.2)

    plt.xlabel('n_inj')
    plt.ylabel('n_fit')
    plt.title(f'Bias for point source at dec= {str(dec)}')
    plt.savefig(f'./plots/bias_dec{str(dec)}.png')

plt.clf()
plt.plot(decs, sensitivity_flux)
plt.xlabel('declination')
plt.ylabel(r'Sensitivity flux $E^2$ dN/dE at 1 TeV [GeV cm$^-2$]')
plt.title('Point source sensitivity flux (gamma=2.0)')
plt.yscale('log')
plt.xlim([-90,90])
plt.savefig(f'./plots/sensitivity_flux_v_dec.png')

with open('./calculated_sensitivities.pickle','wb') as f:
    pickle.dump({'dec': decs,
                 'sens_ns':sensitivity_ns,
                 'sens_flux':sensitivity_flux}, f)
