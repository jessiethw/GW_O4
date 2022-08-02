''' Script to make plots for O4 saved mocks

    Author: Jessie Thwaites
    Date:   July 2022
'''
  
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import glob
import pickle
from datetime import date

def make_bg_pval_dist(fontsize=15):
    all_maps_saved_pkl=sorted(glob.glob('/data/user/jthwaites/o4-mocks/*/*.pickle'))[::-1]
    saved_mock_pkl=[all_maps_saved_pkl[0]]

    for mock in all_maps_saved_pkl:
        event_name=mock.split('-')[-3]
        event_name=event_name.split('/')[-1]
        if event_name not in saved_mock_pkl[-1]:
            saved_mock_pkl.append(mock)

    all_mocks={}
    print('Loading %i mocks (may take a while)'%(len(saved_mock_pkl)))
    for mock in saved_mock_pkl:
        with open(mock,'rb') as f:
            result=pickle.load(f)
            all_mocks[result['name']]=result['p']
    print('Done loading mocks.')

    mpl.rcParams.update({'font.size':fontsize})
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.tick_params(labelsize=fontsize)

    p_x_vals = np.logspace(-2.5,0.,16)
    n, bins, patches = plt.hist([all_mocks[name] for name in all_mocks.keys()], 
                                weights = np.ones(len(all_mocks)) / len(all_mocks), bins = p_x_vals)
    
    lt_10per = sum(n[bins[:-1]<=0.1])
    lt_1per=sum(n[bins[:-1]<=0.01])
    
    plt.plot([0.1,0.1], [1e-3, 1e0],linestyle='dotted', label='%.1f %% of p-values < 0.1'%(lt_10per*100.))
    plt.plot([0.01, 0.01], [1e-3, 1e0], linestyle='dashed',label='%.1f %% of p-values < 0.01'%(lt_1per*100.))

    plt.step(p_x_vals[1:], np.diff(p_x_vals), label = 'Uniform p-value expectation', lw = 3.)
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.grid(which = 'both', alpha = 0.2)
    plt.xlim(1.1e0,1e-3)
    plt.ylim(10**-2.5, 1e0)
    plt.xlabel('p-value', fontsize = fontsize)
    plt.ylabel('Fraction of Analyses', fontsize = fontsize)
    plt.legend(loc = 1, fontsize = fontsize)
    plt.title('Histogram of %i mock p-values as of %s'%(len(saved_mock_pkl),str(date.today())))

    save_path='/data/user/jthwaites/o4-mocks/mock_pvals_%s.png'%str(date.today())
    plt.savefig(save_path)
    print('Figure saved to file: ', save_path)
    
make_bg_pval_dist()