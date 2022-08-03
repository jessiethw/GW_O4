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

import scipy as sp
from numpy.lib.recfunctions   import append_fields
from scipy.optimize       import curve_fit
from scipy.stats          import chi2
from scipy import sparse
from scipy.special import erfinv
import pickle

from skylab.ps_injector   import PointSourceInjector
import sys
if '/data/user/jthwaites/gw_o4' not in sys.path:
    sys.path.append('/data/user/jthwaites/gw_o4')
from config_GW            import config
from fast_response        import sensitivity_utils

fontsize=15

sensitivity_flux=[]
sensitivity_ns=[]
fit_names=[]
decs=[-67.5, -45., -22.5, 0., 22.5, 45., 67.5]

for dec in decs:
    sens_trials=[f'./sens_trials/point_source/ps_sens_{str(dec)}_trials_{str(pid)}.pkl' for pid in range(0,200)]

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
    passing = np.array(passing_frac[:16], dtype=float)
    errs = sensitivity_utils.binomial_error(passing, 1000.)
    fits, plist = [], []
    for func, func_name in [(sensitivity_utils.chi2cdf, 'chi2'),
                            (sensitivity_utils.erfunc, 'Error function'),
                            (sensitivity_utils.incomplete_gamma, 'Incomplete gamma')]:
        try:
            sens_fit=sensitivity_utils.sensitivity_fit(ns_inj[:16], passing, errs, func)
            if str(sens_fit['chi2']) != 'nan' and str(sens_fit['pval']) != 'nan':
                fits.append(sens_fit)
                plist.append(fits[-1]['pval'])
        except:
            print(f"{func_name} fit failed in upper limit calculation")

    plist = np.array(plist)
    best_fit_ind= np.argmax(plist)
    
    #n_inj_range=np.arange(0., ns_inj[16], 0.1)
    sensitivity=fits[best_fit_ind]['sens']
    fits[best_fit_ind]['ls'] = '-'
    fit_names.append(fits[best_fit_ind]['name'])

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

    for fit_dict in fits:
        label=r'{}: $\chi^2$ = {:.2f}, d.o.f. = {}'.format(fit_dict['name'], fit_dict['chi2'], fit_dict['dof'])
        ax.plot(fit_dict['xfit'], fit_dict['yfit'], 
                label = label, ls = fit_dict['ls'])
        if fit_dict['ls'] == '-':
            ax.axhline(0.9, color = 'm', linewidth = 0.3, linestyle = '-.')
            ax.axvline(fit_dict['sens'], color = 'm', linewidth = 0.3, linestyle = '-.')
            ax.text(3.5, 0.8, 'Sens. = {:.2f} events'.format(fit_dict['sens']), fontsize = fontsize)
            ax.text(3.5, 0.7, ' = {:.1e}'.format(inj.mu2flux(sensitivity)*1e9) + r' GeV cm$^-2$', fontsize=fontsize)
    ax.errorbar(ns_inj[:16], passing, yerr=errs, capsize = 3, linestyle='', marker = 's', markersize = 2)
    ax.legend(loc=0, fontsize = fontsize)
    ax.set_xlabel('n inj', fontsize = fontsize)
    ax.set_ylabel(r'Fraction TS $>$ threshold', fontsize = fontsize)

    plt.title(f'Passing fraction for point source, dec = {str(dec)}')

    plt.savefig(f'./plots/passing_frac_dec{str(dec)}.png')

    ### Make bias plot ###
    plt.clf()
    plt.plot(ns_inj,ns_fit_mean)
    plt.plot(ns_inj,ns_inj,'--', color='C0')
    plt.fill_between(ns_inj, 
                    [ns_fit_mean[i]-ns_fit_1sigma[i] for i in range(len(ns_fit_mean))], 
                    [ns_fit_mean[i]+ns_fit_1sigma[i] for i in range(len(ns_fit_mean))], 
                    alpha=0.2)

    plt.xlabel('n inj')
    plt.ylabel('n fit')
    plt.title(f'Bias for point source at dec= {str(dec)}')
    plt.savefig(f'./plots/bias_dec{str(dec)}.png')

plt.clf()
dec_range = np.linspace(-1,1,35)
o3_sens = [1.15, 1.06, .997, .917, .867, .802, .745, .662,
            .629, .573, .481, .403, .332, .250, .183, .101,
            .035, .0286, .0311, .0341, .0361, .0394, .0418,
            .0439, .0459, .0499, .0520, .0553, .0567, .0632,
            .0679, .0732, .0788, .083, .0866]
o3_sens = np.array(o3_sens)
plt.plot(dec_range, o3_sens, label='O3 (realtime)')
plt.plot(decs, sensitivity_flux, 'o', label='O4 (this work)')
plt.xlabel('declination')
plt.ylabel(r'Sensitivity flux E$^2$ dN/dE at 1 TeV [GeV cm$^-2$]')
plt.title('Point source sensitivity flux (gamma=2.0)')
plt.yscale('log')
plt.xlim([-90,90])
plt.legend(loc=0)
plt.savefig(f'./plots/sensitivity_flux_v_dec.png')

with open('./calculated_sensitivities.pickle','wb') as f:
    pickle.dump({'dec': decs,
                 'sens_ns':sensitivity_ns,
                 'sens_flux':sensitivity_flux,
                 'best_fit_name':fit_names}, f)
