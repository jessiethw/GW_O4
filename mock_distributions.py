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
from fast_response import web_utils

def make_bg_pval_dist(fontsize=15, lower_y_bound=-3):
    # function to make pval dist. lower_y_bound arg gives the exponent to set the lower y-axis 
    # limit, e.g. 10^-3
    all_maps_saved_pkl=sorted(glob.glob('/data/user/jthwaites/o4-mocks/*/*.pickle'))[::-1]
    saved_mock_pkl=[all_maps_saved_pkl[0]]

    for mock in all_maps_saved_pkl:
        event_name=mock.split('-')[-3]
        event_name=event_name.split('/')[-1]
        if event_name not in saved_mock_pkl[-1]:
            saved_mock_pkl.append(mock)

    all_mocks={}
    print('Loading %i mocks (may take a while)'%(len(saved_mock_pkl)))
    i=0
    for mock in saved_mock_pkl:
        with open(mock,'rb') as f:
            result=pickle.load(f)
            all_mocks[result['name']]=result['p']
        i+=1
        if (i/len(saved_mock_pkl))%10==0:
            print('%i loaded'%((i/len(saved_mock_pkl))%10))
    print('Done loading mocks.')

    mpl.rcParams.update({'font.size':fontsize})
    plt.figure(figsize = (10,6), dpi=300)
    #ax.tick_params(labelsize=fontsize)

    p_x_vals = np.logspace(-2.5,0.,16)
    n, bins, patches = plt.hist([all_mocks[name] for name in all_mocks.keys()], 
                                weights = np.ones(len(all_mocks)) / len(all_mocks), bins = p_x_vals)
    
    lt_10per = sum(n[bins[:-1]<=0.1])
    lt_1per=sum(n[bins[:-1]<=0.01])
    
    uniform_bins=np.logspace(lower_y_bound,0.,int(abs(lower_y_bound*7))+1) #evenly spaced bins in logspace
    plt.step(uniform_bins[1:], np.diff(uniform_bins), label = 'Uniform p-value expectation', lw = 3.)
    plt.plot([0.1,0.1], [10**lower_y_bound, 1e0],linestyle='dotted', label=f'{lt_10per*100.:.2f} \% of p-values $<$ 0.1')
    plt.plot([0.01, 0.01], [10**lower_y_bound, 1e0], linestyle='dashed',label=f'{lt_1per*100.:.2f} \% of p-values $<$ 0.01')

    plt.xscale('log')
    plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.grid(which = 'both', alpha = 0.2)
    plt.xlim(1.1e0,1e-3)
    plt.ylim(10**lower_y_bound, 1e0)
    plt.xlabel('p-value', fontsize = fontsize)
    plt.ylabel('Fraction of Analyses', fontsize = fontsize)
    plt.legend(loc = 1, fontsize = fontsize)
    plt.title('{} Mock p-values as of {}'.format(len(saved_mock_pkl), str(date.today()),fontsize=20))

    save_path='/data/user/jthwaites/o4-mocks/mock_pvals_%s.png'%str(date.today())
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print('Figure saved to file: ', save_path)
    
make_bg_pval_dist()