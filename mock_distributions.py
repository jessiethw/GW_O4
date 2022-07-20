''' Script to make plots for O4 saved mocks

    Author: Jessie Thwaites
    Date:   July 2022
'''
  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import pickle
from datetime import date

def make_bg_pval_dist(fontsize=15):
    saved_mock_pkl=glob.glob('/data/user/jthwaites/o4-mocks/*/*.pickle')

    all_mocks={}
    print('Loading %i mocks (may take a while)'%(len(saved_mock_pkl)))
    for mock in saved_mock_pkl:
        with open(mock,'rb') as f:
            result=pickle.load(f)
            all_mocks[result['name']]=result['p']
    print('Done loading mocks.')

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.tick_params(labelsize=fontsize)
    mpl.rcParams.update({'font.size':fontsize})

    p_x_vals = np.logspace(-3,0.,15)
    plt.figure(figsize = (10,6), dpi=300)
    n, bins, patches = plt.hist([all_mocks[name] for name in all_mocks.keys()], 
                                weights = np.ones(len(all_mocks)) / len(all_mocks), bins = p_x_vals)
    print(len(p_x_vals), len(bins),len(n))
    lt_10per = sum(n[bins[:-1]<=-1])
    lt_1per=sum(n[bins[:-1]<=-2])
    plt.plot([0.1,0.1], [1e-2, 1e0], label='%.1f% < 0.1'%(lt_10per))
    plt.plot([0.01, 0.01], [1e-2, 1e0], label='%.1f% < 0.01'%(lt_1per))

    #plt.step(p_x_vals[1:], np.diff(p_x_vals), label = 'Uniform p-value expectation', lw = 3.)
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.grid(which = 'both', alpha = 0.2)
    plt.xlim(1.1e0,1e-3)
    plt.ylim(1e-2, 1e0)
    plt.xlabel('p-value', fontsize = fontsize)
    plt.ylabel('Fraction of Analyses', fontsize = fontsize)
    plt.legend(loc = 1, fontsize = fontsize)
    plt.title('Histogram of %i mock p-values as of %s'%(len(saved_mock_pkl),str(date.today())))

    save_path='/data/user/jthwaites/o4-mocks/mock_pvals_%s.png'%str(date.today())
    plt.savefig(save_path)
    print('Figure saved to file: ', save_path)

make_bg_pval_dist()