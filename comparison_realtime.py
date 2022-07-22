''' Script to make additional plots comparing realtime runs to 
    replicated runs with FRA GW followup

    Author: Jessie Thwaites
    Date:   June 2022
'''

import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import glob
import pickle

def make_p_comp(fontsize=15):
    #make a p value comparison between realtime and offline runs of o3 gws

    data_files=glob.glob('/data/user/jthwaites/o3-followups-jessie/convention_negTS/*/*.pickle') 
    fra_p={}
    for file in data_files:
        with open(file,'rb') as f:
            result=pickle.load(f)
            fra_p[result['name']]=result['p']

    #unfortunately, saved data does not exist for realtime runs, so this will be hardcoded:
    realtime_p={'S190408an-1-Preliminary': 0.1007 ,'S190412m-1-Preliminary': 1.0, 'S190421ar-3-Update':1.0,
            'S190425z-2-Update':1.0, 'S190426c-5-Update': 0.1621, 'S190503bf-1-Preliminary': 1.0,
            'S190510g-4-Update':1.0, 'S190512at-3-Update':1.0, 'S190513bm-2-Initial':1.0,
            'S190517h-2-Initial':0.937, 'S190519bj-2-Initial':1.0, 'S190521g-2-Initial':1.0,
            'S190521r-1-Preliminary':0.0683, 'S190602aq-2-Initial':0.1750, 'S190630ag-3-Update':0.6862,
            'S190701ah-3-Update':1.0, 'S190706ai-3-Update':1.0, 'S190707q-3-Update':0.61,
            'S190718y-3-Initial':0.7321, 'S190720a-3-Update':0.9767, 'S190727h-3-Update':0.9521,
            'S190728q-5-Update':0.0136, 'S190814bv-5-Update':0.9425, 'S190828j-4-Update':1.0,
            'S190828l-3-Update':0.6344, 'S190901ap-3-Update':0.3467, 'S190910d-3-Update':0.7565,
            'S190910h-3-Update':0.2081, 'S190915ak-3-Update':0.5105, 'S190923y-2-Initial':0.4546,
            'S190924h-3-Update':0.1862, 'S190930s-3-Update':0.3147, 'S190930t-2-Initial':0.3032,
            'S191105e-3-Update':0.4998, 'S191109d-2-Initial':0.0199, 'S191129u-3-Update':0.2440,
            'S191204r-4-Initial':0.8631, 'S191205ah-2-Preliminary':0.9818, 'S191213g-4-Update':0.5745,
            'S191215w-4-Update':1.0, 'S191216ap-4-Update':0.0598, 'S191222n-4-Update':0.9455,
            'S200105ae-3-Update':0.7186, 'S200112r-3-Initial':0.4286, 'S200114f-3-Initial':0.1646,
            'S200115j-3-Preliminary':0.4123, 'S200128d-3-Initial':0.2968, 'S200129m-3-Initial':0.0414,
            'S200208q-4-Update':0.9554, 'S200213t-5-Update':0.0055, 'S200219ac-3-Update':0.9127,
            'S200224ca-3-Initial':0.8043, 'S200225q-3-Initial':0.2325, 'S200302c-4-Update':0.0576,
            'S200311bg-4-Update':1.0, 'S200316bj-3-Initial':0.011}

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.tick_params(labelsize=fontsize)
    mpl.rcParams.update({'font.size':fontsize})

    plt.hist2d([fra_p[name] for name in fra_p.keys()], [realtime_p[name] for name in fra_p.keys()], bins=(15,15))

    plt.title('Comparison of p-values in realtime and offline')
    plt.xlabel('New p-value (offline, FRA)',fontsize=fontsize) 
    plt.ylabel('Old p-value (realtime)',fontsize=fontsize)
    cbar=plt.colorbar()
    cbar.set_label('Number of analyses')

    plt.savefig('/data/user/jthwaites/o3-followup-jessie/pval_comp_hist.png')

def llama_uml_pval_comp(fontsize=15):
    #comparison plots between llama and uml analyses pvalues
    #from o3 gws

    import pandas as pd
    o3_gws=pd.read_csv('./llama_uml_pval_comp.csv')
    llama_p=o3_gws['llama_p'].values
    uml_p=o3_gws['uml_p'].values

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.tick_params(labelsize=fontsize)
    mpl.rcParams.update({'font.size':fontsize})

    xy=np.linspace(0,1.0)
    plt.scatter(uml_p,llama_p)
    plt.plot(xy,xy, '--',color='gray')
    plt.title('Comparison of LLAMA and UML p-values in O3')
    plt.xlabel('UML p-value',fontsize=fontsize)
    plt.ylabel('LLAMA p-value',fontsize=fontsize)
    plt.savefig('uml_llama_p_scatter.png')

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.tick_params(labelsize=fontsize)
    plt.hist2d(uml_p,llama_p, bins=(15,15))
    plt.title('2D histogram comparison of LLAMA/UML pvalues in O3')
    plt.xlabel('UML p-value',fontsize=fontsize)
    plt.ylabel('LLAMA p-value',fontsize=fontsize)
    cbar=plt.colorbar()

    plt.savefig('/data/user/jthwaites/o3-followup-jessie/uml_llama_p_2dhist.png')

def compare_conventions(fontsize=15):
    #compare TS>=0 convention to neg TS allowed convention directly
    #by looking at p-values in each case
    old_runs=glob.glob('/data/user/jthwaites/o3-followups-jessie/convention_negTS/*/*.pickle') 
    new_runs=glob.glob('/data/user/jthwaites/o3-followups-jessie/convention_posTS/*/*.pickle') 
    fra_p={}
    for file in old_runs:
        with open(file,'rb') as f:
            result=pickle.load(f)
            fra_p[result['name']]=[result['p']]
    for file in new_runs:
        with open(file,'rb') as f:
            result=pickle.load(f)
            fra_p[result['name'][:-2]].append(result['p'])

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.tick_params(labelsize=fontsize)
    mpl.rcParams.update({'font.size':fontsize})

    plt.hist2d([fra_p[name][0] for name in sorted(fra_p.keys())], 
            [fra_p[name][1] for name in sorted(fra_p.keys())], bins=(15,15))

#    for name in sorted(fra_p.keys()):
#        print(name, fra_p[name])

    plt.title('Comparison of TS conventions')
    plt.xlabel('p-value, negative TS allowed',fontsize=fontsize) 
    plt.ylabel('p-value, TS>=0',fontsize=fontsize)
    cbar=plt.colorbar()
    cbar.set_label('Number of analyses')

    plt.savefig('/data/user/jthwaites/o3-followup-jessie/convention_comparison_hist.png')

compare_conventions()
#make_p_comp()
#llama_uml_pval_comp()