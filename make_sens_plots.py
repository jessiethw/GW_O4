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
import pickle
import argparse

from skylab.ps_injector       import PointSourceInjector
from skylab.priors            import SpatialPrior
from astropy.time             import Time
import sys
if '/data/user/jthwaites/gw_o4' not in sys.path:
    sys.path.append('/data/user/jthwaites/gw_o4')
from config_GW            import config
from fast_response        import sensitivity_utils
import glob

p = argparse.ArgumentParser(description="Calculates Sensitivity and makes plots")
p.add_argument("--fontsize", default=15, type=int,
                help="fontsize for plots (default=15)")
p.add_argument("--reload_sens", default=False, type=bool,
                help='choice to reload saved ps sensitivities for sens vs dec plot')
p.add_argument("--with_map", default=False, type=bool,
                help='calculate a sensitivity for a gw event with a spatial prior map')
p.add_argument("--version", default='v001p02', type=str,
                help='version of GFUOnline to use (default= v001p02)')
p.add_argument("--nside", default=256, type=int,
		        help='nside to use for map if using spatial prior')
p.add_argument('--tw', default=1000., type=float,  #[-1, +14]day: 1382400
                help='time window to use (default =1000.)')
args = p.parse_args()

if args.with_map: name='S191216ap'
else: name='point_source'

if int(args.tw) != 1000:
    suffix='_2week'
    name=name+suffix
else: suffix=''

def calc_passing(TS_list, bg_trials_path, dec=None, dp=False):
    '''Function to calculate passing fractions for sensitivity
    and discovery potentials. 
    dp: set to true for discovery potential
    Returns a passing fraction for sensitivity or discovery potential'''
    try:
        sens_ts=[t[0] for t in TS_list]
    except:
        sens_ts=TS_list
    
    bg_TS = np.array([])
    if dec is not None: 
        saved_pkls=glob.glob(bg_trials_path+f'/*{dec}*.pkl')
    else: 
        saved_pkls=glob.glob(bg_trials_path+'/*.pkl')
    for file in saved_pkls:
        with open(file, 'rb') as f:
            bg = pickle.load(f)
            try: 
                bg_TS = np.concatenate((bg_TS, bg['TS_List']))
            except:
                ts = [t[0] for t in bg['TS_List']] #sometimes saves as a list of lists
                bg_TS = np.concatenate((bg_TS, ts))
    
    bg_TS[bg_TS<0.]=0.
    
    if dp:
        ts = np.percentile(bg_TS, 100-0.13)
    else:
        ts = np.median(bg_TS)
    
    npass = 0
    for TS_i in sens_ts:
        if TS_i > ts:
            npass+=1
    P = float(npass)/len(sens_ts)

    if P==0.:
        P=0.0001
    if P==1.:
        P=0.9999

    return P

def calc_sensitivty(passing_frac, flux_inj, flux=True):
    passing = np.array(passing_frac, dtype=float)
    flux_inj=np.array(flux_inj)

    errs = sensitivity_utils.binomial_error(passing, 1000.)

    fits, plist = [], []
    for func, func_name in [(sensitivity_utils.chi2cdf, 'chi2'),
                            (sensitivity_utils.erfunc, 'Error function'),
                            (sensitivity_utils.incomplete_gamma, 'Incomplete gamma')]:
        try:
            sens_fit=sensitivity_utils.sensitivity_fit(flux_inj, passing, errs, func, p0=None)
            if str(sens_fit['chi2']) != 'nan' and str(sens_fit['pval']) != 'nan':
                fits.append(sens_fit)
                plist.append(fits[-1]['pval'])
            else: 
                print(f"{func_name} fit failed in upper limit calculation")
        except:
            print(f"{func_name} fit failed in upper limit calculation")

    plist = np.array(plist)
    best_fit_ind= np.argmax(plist)
    
    sensitivity=fits[best_fit_ind]['sens']
    fits[best_fit_ind]['ls'] = '-'

    if not args.with_map:
        print('Sensitivity at dec = %.1f: %.2f events'%(dec, sensitivity))
    elif not flux:
        print(f'Sensitivity  = {sensitivity} events')
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

def make_passing_frac_curve(fits, sensitivity, errs, ns_inj, passing_frac, dec=None, flux=False):
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
            #point source: use PS injector to get sensitivity on plot
            if dec is not None:
                ax.text(3.5, 0.7, ' = {:.1e}'.format(inj.mu2flux(sensitivity)*1e9) + r' GeV cm$^-2$', fontsize=fontsize)
    ax.errorbar(ns_inj, passing_frac, yerr=errs, capsize = 3, linestyle='', marker = 's', markersize = 2)
    ax.legend(loc=0, fontsize = fontsize)
    if flux:
        ax.set_xlabel('flux inj', fontsize = fontsize)
        ax.set_xlim([min(ns_inj)-0.02, max(ns_inj)+0.02])
    else:
        ax.set_xlabel('n inj', fontsize = fontsize)
    ax.set_ylabel(r'Fraction TS $>$ threshold', fontsize = fontsize)

    if dec is not None:
        plt.title(f'Passing fraction for point source, dec = {str(dec)}')
        plt.savefig(f'./plots/passing_frac_dec{str(dec)}{suffix}.png')
    elif dec is None and flux: 
        plt.title(f'Passing fraction for S191216ap at 1 TeV'+r' [GeV cm$^{-2}$]')
        plt.savefig(f'./plots/{args.version}_S191216ap_passing_frac_flux{suffix}.png')
    else: 
        plt.title(f'Passing fraction for S191216ap at 1 TeV'+r' [GeV cm$^{-2}$]')
        plt.savefig(f'./plots/{args.version}_S191216ap_passing_frac_flux{suffix}.png')

reload=args.reload_sens
if args.with_map:
    reload=True #don't recalculate ps sensitivities

fontsize=args.fontsize
sensitivity_flux=[]
sensitivity_ns=[]
dp_flux=[]
dp_ns=[]
#decs=[-67.5, -45., -22.5, 0., 22.5, 45., 67.5]
decs= np.linspace(-85,85,35)

if not reload:
    for dec in decs:
        sens_trials=[f'./sens_trials/{name}/ps_sens_{str(dec)}_trials_{str(pid)}{suffix}.pkl' 
                    for pid in range(0,200)]
        #sens_trials=sorted(glob.glob(f'./sens_trials/{name}/ps_sens_{str(dec)}_trials_*.pkl'))

        passing_frac=[]
        ns_fit_mean=[]
        ns_fit_1sigma=[]
        ns_inj=[]
        dp_pf=[]
        for saved_trial in sens_trials:
            with open(saved_trial, 'rb') as f:
                result=pickle.load(f)
                #passing_frac.append(result['passFrac'][0])
                passing_frac.append(calc_passing(result['TS_List'], f'./bg_trials/{name}/', dec=dec))
                dp_pf.append(calc_passing(result['TS_List'], f'./bg_trials/{name}/', dec=dec, dp=True))
                ns_fit_mean.append(np.mean(result['ns_fit']))
                ns_fit_1sigma.append(np.std(result['ns_fit']))
                ns_inj.append(result['ns_inj'])

        #### Calculate sensitivity #####
        sensitivity, fits, errs = calc_sensitivty(passing_frac[1:16], ns_inj[1:16])
        dp, fits_dp, errs_dp = calc_sensitivty(dp_pf[1:21], ns_inj[1:21])

        llh = config([f'GFUOnline_{args.version}','IC86, 2011-2018'],gamma=2.,ncpu=2, days=5,
              time_mask=[500./3600./24.,57982.52852350], poisson=True)

        inj = PointSourceInjector(E0=1000.)
        inj.fill(np.deg2rad(dec),llh.exp,llh.mc,llh.livetime,
                temporal_model=llh.temporal_model)

        sensitivity_ns.append(sensitivity)
        dp_ns.append(dp)
        #flux at E0=1 TeV, convert to [GeV cm^-2 s]
        sensitivity_flux.append(inj.mu2flux(sensitivity)*1e9)
        dp_flux.append(inj.mu2flux(dp)*1e9)

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
        ax.errorbar(ns_inj[1:16], passing_frac[1:16], yerr=errs, capsize = 3, linestyle='', marker = 's', markersize = 2)
        
        for fit_dict in fits_dp:
            label=r'{}: $\chi^2$ = {:.2f}, d.o.f. = {}'.format(fit_dict['name'], fit_dict['chi2'], fit_dict['dof'])
            ax.plot(fit_dict['xfit'], fit_dict['yfit'], 
                label = label, ls = fit_dict['ls'])
            if fit_dict['ls'] == '-':
                ax.axhline(0.9, color = 'm', linewidth = 0.3, linestyle = '-.')
                ax.axvline(fit_dict['sens'], color = 'm', linewidth = 0.3, linestyle = '-.')
        ax.errorbar(ns_inj[1:21], dp_pf[1:21], yerr=errs_dp, capsize = 3, linestyle='', marker = 's', markersize = 2)
        
        ax.legend(loc=0, fontsize = fontsize)
        ax.set_xlabel('n inj', fontsize = fontsize)
        ax.set_ylabel(r'Fraction TS $>$ threshold', fontsize = fontsize)

        plt.title(f'Passing fraction for point source, dec = {str(dec)}')
        plt.savefig(f'./plots/passing_frac_dec{str(dec)}{suffix}.png')
        
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
        plt.savefig(f'./plots/bias_dec{str(dec)}{suffix}.png')
    
    with open(f'./calculated_sensitivities{suffix}.pickle','wb') as f:
        pickle.dump({'dec': decs,
                 'sens_ns':sensitivity_ns,
                 'sens_flux':sensitivity_flux,
                 'dp_flux':dp_flux,
                 'dp_ns':dp_ns}, f)
else: 
    print('Reloading calculated sensitivities')

    mpl.rcParams.update({'font.size':fontsize})
    fig,ax = plt.subplots(figsize = (10,6))
    ax.tick_params(labelsize=fontsize)  

    with open(f'./calculated_sensitivities{suffix}.pickle','rb') as f:
        sens=pickle.load(f)
        sensitivity_flux=sens['sens_flux']
        dp_flux=sens['dp_flux']
        decs=sens['dec']

if args.with_map:
    sens_trials=[f'./sens_trials/{name}/{args.version}_S191216ap_prior_sens_trials_{str(pid)}{suffix}.pkl' 
                    for pid in range(0,50)]
    passing_frac=[]
    ns_fit_mean=[]
    ns_fit_1sigma=[]
    ns_inj=[]
    flux_inj=[]
    ns_fit=[]
    gamma_fit_mean=[]
    gamma_fit_std=[]
    for i in range(len(sens_trials)):
        with open(sens_trials[i], 'rb') as f:
                result=pickle.load(f)
                #passing_frac.append(result['passFrac'][0])
                passing_frac=calc_passing(result['TS_List'], f'./bg_trials/{name}/')
                ns_fit_mean.append(np.mean(result['ns_fit']))
                ns_fit_1sigma.append(np.std(result['ns_fit']))
                ns_inj.append(np.mean(result['ns_inj']))
                ns_fit.append(result['ns_fit'])
                gamma_fit_mean.append(np.mean(result['gamma_fit']))
                gamma_fit_std.append(np.std(result['gamma_fit']))
                if result['flux_inj']: #if list is empty, no flux injected
                    flux_inj.append(np.mean(result['flux_inj'])*1e9)
                else:
                    flux_inj.append(0.)

    #### Calculate sensitivity #####
    sensitivity, fits, err = calc_sensitivty(passing_frac[1:20], ns_inj[1:20], flux=False)

    ### Get map min/max dec ###
    skymap, skymap_header = hp.read_map('/data/user/jthwaites/gw_o4/sens_trials/S191216ap/S191216ap.fits.gz',
                                        h=True, verbose=False)
    nside=hp.pixelfunc.get_nside(skymap)
    if nside != args.nside:
        skymap = hp.pixelfunc.ud_grade(skymap,nside_out=args.nside,power=-2)
        nside=args.nside

    spatial_prior = SpatialPrior(skymap, containment = 0.99, allow_neg=False)

    ipix_90=ipixs_in_percentage(skymap, 0.9)
    src_theta, src_phi = hp.pix2ang(nside, ipix_90)
    src_dec = np.pi/2. - src_theta
    src_dec = np.unique(src_dec)

    max_dec=max(np.sin(src_dec))
    min_dec=min(np.sin(src_dec))
    best_dec=np.sin(np.pi/2. - hp.pix2ang(nside, np.where(skymap==max(skymap))[0][0])[0])
    
    make_passing_frac_curve(fits, sensitivity, err, ns_inj[1:20], passing_frac[1:20], dec=None)
    
    ### Make bias plot ###
    for j in [0,1]:
        plt.clf()
        if j==0:
            plt.plot(ns_inj,ns_fit_mean)
            plt.fill_between(ns_inj, 
                    [ns_fit_mean[i]-ns_fit_1sigma[i] for i in range(len(ns_fit_mean))], 
                    [ns_fit_mean[i]+ns_fit_1sigma[i] for i in range(len(ns_fit_mean))], 
                    alpha=0.2)
        else: 
            for i in range(len(ns_inj)):
                plt.plot([ns_inj[i]]*len(ns_fit[i]), ns_fit[i], '.', color='C0')
        plt.plot(ns_inj,ns_inj,'--', color='C1', lw=2.)

        plt.xlabel('n inj')
        plt.ylabel('n fit')
        plt.title(f'Bias for S191216ap spatial prior at 1 TeV')
        #plt.xlabel('Flux inj')
        #plt.ylabel('Flux fit')
        #plt.title(f'Bias for S191216ap spatial prior at 1 TeV '+r'[GeV cm$^-2$]')
        if j==0: plt.savefig(f'./plots/{args.version}_S191216ap_bias{suffix}.png')
        else: plt.savefig(f'./plots/{args.version}_S191216ap_bias_scatter{suffix}.png')
    
    #bias plot for gamma
    plt.clf()
    plt.plot(ns_inj,gamma_fit_mean)
    plt.fill_between(ns_inj, 
        [gamma_fit_mean[i]-gamma_fit_std[i] for i in range(len(gamma_fit_mean))], 
        [gamma_fit_mean[i]+gamma_fit_std[i] for i in range(len(gamma_fit_mean))], 
        alpha=0.2)
    plt.plot(ns_inj,[2.0]*len(ns_inj),'--', color='C1', lw=2.)

    plt.xlabel('n inj')
    plt.ylabel('gamma fit')
    plt.title(f'Spectral index fit bias for S191216ap spatial prior')
    plt.savefig(f'./plots/{args.version}_S191216ap_gamma_bias{suffix}.png')
    

plt.clf()
o3_sens = [1.15, 1.06, .997, .917, .867, .802, .745, .662,
            .629, .573, .481, .403, .332, .250, .183, .101,
            .035, .0286, .0311, .0341, .0361, .0394, .0418,
            .0439, .0459, .0499, .0520, .0553, .0567, .0632,
            .0679, .0732, .0788, .083, .0866]
o3_sens = np.array(o3_sens)

plt.plot(np.sin(np.deg2rad(decs)), sensitivity_flux, label='O4 sensitivity')
plt.plot(np.sin(np.deg2rad(decs)), dp_flux, label=r'O4 3$\sigma$ 90\% DP')

if args.with_map:
    llh, inj = config([f'GFUOnline_{args.version}','IC86, 2011-2018'],gamma=2.,ncpu=2, days=5,
            spatial_prior = spatial_prior, time_mask=[500./3600./24.,57982.52852350], poisson=True)
    
    dec_errs=[[best_dec-min_dec], [max_dec-best_dec]]
    plt.errorbar(best_dec, inj.mu2flux(sensitivity)*1e9, xerr=dec_errs, marker = 'o', label='S191216ap (GW skymap best-fit)')
#else:
#    plt.plot(np.sin(np.deg2rad(decs)), o3_sens, label='O3 (realtime)')

plt.xlabel('sin(declination)')
plt.ylabel(r'Sensitivity flux E$^2$ dN/dE at 1 TeV [GeV cm$^-2$]')
plt.title('Point source sensitivity flux (gamma=2.0)')
plt.yscale('log')
plt.xlim([-1,1])
plt.legend(loc=0)
if args.with_map:
    plt.savefig(f'./plots/{args.version}_sensitivity_flux_w_prior{suffix}.png')
else:
    plt.savefig(f'./plots/sensitivity_flux_v_dec{suffix}.png')

