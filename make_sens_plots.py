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
import healpy as hp

import scipy as sp
from numpy.lib.recfunctions   import append_fields
from scipy.optimize       import curve_fit
from scipy.stats          import chi2
from scipy import sparse
from scipy.special import erfinv
import pickle
import argparse

from skylab.ps_injector   import PointSourceInjector
import sys
if '/data/user/jthwaites/gw_o4' not in sys.path:
    sys.path.append('/data/user/jthwaites/gw_o4')
from config_GW            import config
from fast_response        import sensitivity_utils

p = argparse.ArgumentParser(description="Calculates Sensitivity and makes plots",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--fontsize", default=15, type=int,
                help="fontsize for plots (default=15)")
p.add_argument("--reload_sens", default=False, type=bool,
                help='choice to reload saved ps sensitivities for sens vs dec plot')
p.add_argument("--with_map", default=False, type=bool,
                help='calculate a sensitivity for a gw event with a spatial prior map')
args = p.parse_args()

def calc_sensitivty(passing_frac, flux_inj):
    passing = np.array(passing_frac, dtype=float)
    errs = sensitivity_utils.binomial_error(passing, 1000.)
    
    fits, plist = [], []
    for func, func_name in [(sensitivity_utils.chi2cdf, 'chi2'),
                            (sensitivity_utils.erfunc, 'Error function'),
                            (sensitivity_utils.incomplete_gamma, 'Incomplete gamma')]:
        try:
            if func_name=='chi2':
                p0=[0.3, 0.01, 0.02]
            #incomplete gamma not working rn
            #elif func_name=='Incomplete gamma':
            #    p0=[2., 3.]
            else: p0=None
            sens_fit=sensitivity_utils.sensitivity_fit(flux_inj, passing, errs, func, p0=p0)
            if str(sens_fit['chi2']) != 'nan' and str(sens_fit['pval']) != 'nan':
                fits.append(sens_fit)
                plist.append(fits[-1]['pval'])
            #else: 
            #    print(f"{func_name} fit failed in upper limit calculation")
        except:
            print(f"{func_name} fit failed in upper limit calculation")

    plist = np.array(plist)
    best_fit_ind= np.argmax(plist)
    
    sensitivity=fits[best_fit_ind]['sens']
    fits[best_fit_ind]['ls'] = '-'

    if not args.with_map:
        print('Sensitivity at dec = %.1f: %.2f events'%(dec, sensitivity))
    else:
        print(f'Sensitivity = {sensitivity} GeV cm^-2')
    return sensitivity, fits, errs

def ipixs_in_percentage(skymap, percentage):
    #taken directly from FastResponseAnalysis.py
    """Finding ipix indices confined in a given percentage.
        
    Input parameters
    ----------------
    skymap: ndarray
        array of probabilities for a skymap
    percentage : float
        fractional percentage from 0 to 1  
    Return
    ------- 
    ipix : numpy array
        indices of pixels within percentage containment
    """

    sort = sorted(skymap, reverse = True)
    cumsum = np.cumsum(sort)
    index, value = min(enumerate(cumsum),key=lambda x:abs(x[1]-percentage))

    index_hpx = list(range(0,len(skymap)))
    hpx_index = np.c_[skymap, index_hpx]

    sort_2array = sorted(hpx_index,key=lambda x: x[0],reverse=True)
    value_contour = sort_2array[0:index]

    j = 1 
    table_ipix_contour = [ ]
    for i in range (0, len(value_contour)):
        ipix_contour = int(value_contour[i][j])
        table_ipix_contour.append(ipix_contour)
    
    ipix = table_ipix_contour
          
    return np.asarray(ipix,dtype=int)

reload=args.reload_sens
if args.with_map:
    reload=True #don't recalculate ps sensitivities

fontsize=args.fontsize
sensitivity_flux=[]
sensitivity_ns=[]
#decs=[-67.5, -45., -22.5, 0., 22.5, 45., 67.5]
decs= np.linspace(-85,85,35)

if not reload:
    for dec in decs:
        sens_trials=[f'./sens_trials/point_source/ps_sens_{str(dec)}_trials_{str(pid)}.pkl' 
                    for pid in range(0,200)]

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
        sensitivity, fits, errs = calc_sensitivty(passing_frac[:16], ns_inj[:16])
        
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
        ax.errorbar(ns_inj[:16], passing_frac[:16], yerr=errs, capsize = 3, linestyle='', marker = 's', markersize = 2)
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
    
    with open('./calculated_sensitivities.pickle','wb') as f:
        pickle.dump({'dec': decs,
                 'sens_ns':sensitivity_ns,
                 'sens_flux':sensitivity_flux}, f)
else: 
    print('Reloading calculated sensitivities')

    mpl.rcParams.update({'font.size':fontsize})
    fig,ax = plt.subplots(figsize = (10,6))
    ax.tick_params(labelsize=fontsize)  

    with open('./calculated_sensitivities.pickle','rb') as f:
        sens=pickle.load(f)
        sensitivity_flux=sens['sens_flux']
        decs=sens['dec']

if args.with_map:
    sens_trials=[f'./sens_trials/S191216ap/S191216ap_prior_sens_trials_{str(pid)}.pkl' 
                    for pid in range(0,50)]
    passing_frac=[]
    flux_fit_mean=[]
    flux_fit_1sigma=[]
    flux_inj=[]
    for i in range(len(sens_trials)):
        with open(sens_trials[i], 'rb') as f:
                result=pickle.load(f)
                passing_frac.append(result['passFrac'][0])
                flux_fit_mean.append(np.mean(result['flux_fit'])*1e9)
                flux_fit_1sigma.append(np.std(result['flux_fit'])*1e9)
                flux_inj.append(result['flux_inj'][0]*1e9)

    #### Calculate sensitivity #####
    sensitivity, fits, err = calc_sensitivty(passing_frac[:16], flux_inj[:16])

    ### Get map min/max dec ###
    skymap, skymap_header = hp.read_map('/data/user/jthwaites/gw_o4/sens_trials/S191216ap/S191216ap.fits.gz',
                                        h=True, verbose=False)
    nside=hp.pixelfunc.get_nside(skymap)
    # In FRA.py - need to find a way to use this?
    ipix_90=ipixs_in_percentage(skymap, 0.9)
    src_theta, src_phi = hp.pix2ang(nside, ipix_90)
    src_dec = np.pi/2. - src_theta
    src_dec = np.unique(src_dec)

    max_dec=max(np.sin(src_dec))
    min_dec=min(np.sin(src_dec))
    best_dec=np.sin(np.pi/2. - hp.pix2ang(nside, np.where(skymap==max(skymap))[0][0])[0])
    

    mpl.rcParams.update({'font.size':fontsize})
    fig,ax = plt.subplots(figsize = (10,6))
    ax.tick_params(labelsize=fontsize)    

    for fit_dict in fits:
        label=r'{}: $\chi^2$ = {:.2f}, d.o.f. = {}'.format(fit_dict['name'], fit_dict['chi2'], fit_dict['dof'])
        ax.plot(fit_dict['xfit'], fit_dict['yfit'], label = label, ls = fit_dict['ls'])
        if fit_dict['ls'] == '-':
            ax.axhline(0.9, color = 'm', linewidth = 0.3, linestyle = '-.')
            ax.axvline(fit_dict['sens'], color = 'm', linewidth = 0.3, linestyle = '-.')
            ax.text(0.05, 0.8, 'Sens. = {:.4f}'.format(fit_dict['sens'])+ r' GeV cm$^-2$', fontsize = fontsize)
    ax.errorbar(flux_inj[:16], passing_frac[:16], yerr=err, capsize = 3, linestyle='', marker = 's', markersize = 2)
    ax.legend(loc=0, fontsize = fontsize)
    plt.xlim([-0.02, max(flux_inj[:16])+0.02])
    ax.set_xlabel('flux inj', fontsize = fontsize)
    ax.set_ylabel(r'Fraction TS $>$ threshold', fontsize = fontsize)

    plt.title(f'Passing fraction for S191216ap at 1 TeV'+r' [GeV cm$^{-2}$]')
    plt.savefig(f'./plots/S191216ap_passing_frac.png')
    
    ### Make bias plot ###
    plt.clf()
    plt.plot(flux_inj,flux_fit_mean)
    plt.plot(flux_inj,flux_inj,'--', color='C0')
    plt.fill_between(flux_inj, 
                    [flux_fit_mean[i]-flux_fit_1sigma[i] for i in range(len(flux_fit_mean))], 
                    [flux_fit_mean[i]+flux_fit_1sigma[i] for i in range(len(flux_fit_mean))], 
                    alpha=0.2)

    plt.xlabel('Flux inj')
    plt.ylabel('Flux fit')
    plt.title(f'Bias for S191216ap spatial prior at 1 TeV '+r'[GeV cm$^-2$]')
    plt.savefig(f'./plots/S191216ap_bias.png')

plt.clf()
o3_sens = [1.15, 1.06, .997, .917, .867, .802, .745, .662,
            .629, .573, .481, .403, .332, .250, .183, .101,
            .035, .0286, .0311, .0341, .0361, .0394, .0418,
            .0439, .0459, .0499, .0520, .0553, .0567, .0632,
            .0679, .0732, .0788, .083, .0866]
o3_sens = np.array(o3_sens)

plt.plot(np.sin(np.deg2rad(decs)), sensitivity_flux, label='O4 (this work)')
if args.with_map:
    dec_errs=[[best_dec-min_dec], [max_dec-best_dec]]
    plt.errorbar(best_dec, sensitivity, xerr=dec_errs, marker = 'o', label='S191216ap (GW skymap best-fit)')
else:
    plt.plot(np.sin(np.deg2rad(decs)), o3_sens, label='O3 (realtime)')

plt.xlabel('sin(declination)')
plt.ylabel(r'Sensitivity flux E$^2$ dN/dE at 1 TeV [GeV cm$^-2$]')
plt.title('Point source sensitivity flux (gamma=2.0)')
plt.yscale('log')
plt.xlim([-1,1])
plt.legend(loc=0)
if args.with_map:
    plt.savefig(f'./plots/sensitivity_flux_w_prior.png')
else:
    plt.savefig(f'./plots/sensitivity_flux_v_dec.png')
