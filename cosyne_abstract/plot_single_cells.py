import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage import filters
import os
os.sys.path.append('../')

import utilities as u
import preprocessing as pp
from functools import reduce
import pickle



def plot_highSI_cells(sess,dirbase):

    VRDat,C,S,A = pp.load_scan_sess(sess,analysis='s2p',fneu_coeff=0.7)
    S_trial_mat, occ, edges, centers = u.make_pos_bin_trial_matrices(S,VRDat.pos._values,VRDat.tstart._values,VRDat.teleport._values,bin_size=10)
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    # mean morph by adding up morph and jitters
    effmorph = trial_info['morphs']+trial_info['towerJitter']+trial_info['bckgndJitter']
    effmorph_s = np.sort(effmorph)
    morph_sort = np.argsort(effmorph)

    # load place cell results
    fname = os.path.join("G:\\My Drive\\Figures\\TwoTower\\PlaceCells\\S",
                     mouse,"%s_%s_%d_results.pkl" % (mouse,sess['DateFolder'],sess['SessionNumber']))
    with open(fname,"rb") as f:
        res= pickle.load(f)
    SI = res['SI'][0]['all']+res['SI'][1]['all']
    order = np.argsort(SI)[::-1]

    for i,o in enumerate(order[:100].tolist()):
        f,ax = plt.subplots(1,2,figsize=(10,10))
        S_i = np.copy(S_trial_mat[:,:,o])
        nan_inds = np.isnan(S_i)
        S_i_nanless = np.copy(S_i)
        S_i_nanless[nan_inds]=0
        One = np.ones(S_i.shape)
        One[nan_inds]=.001
        S_i_nanless= filters.gaussian_filter(S_i_nanless,[0,3])
        One = filters.gaussian_filter(One,[0,3])
        S_i = S_i_nanless/One
        S_i[nan_inds]=np.nan
        S_i/=np.nanmean(S_i.ravel())
        ax[0].imshow(S_i[morph_sort,:],cmap='magma')
        ax[1].imshow(S_i,cmap='magma')
        tick_inds = np.arange(0,S_i.shape[0],10)
        ax[0].set_yticks(tick_inds)
        tick_labels = ["%.2f" % effmorph_s[i] for i in tick_inds]
        ax[0].set_yticklabels(tick_labels)
        ax[0].set_ylabel('Mean Morph')
        ax[1].set_ylabel('Trial #')

        f.savefig(os.path.join(dirbase,"%d.pdf" % i),format='pdf')



if __name__ == '__main__':

    # mice = ['4139219.2','4139219.3','4139224.2','4139224.3','4139224.5',
    # '4139251.1','4139260.1','4139260.2','4139261.2','4139265.3','4139265.4',
    # '4139265.5','4139266.3']
    mice = ['4139251.1','4139260.1','4139260.2','4139261.2','4139265.3','4139265.4',
    '4139265.5','4139266.3']

    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])
    tracks = 'TwoTower_noTimeout|TwoTower_Timeout|Reversal_noTimeout|Reversal|TwoTower_foraging'
    df = df[df['Track'].str.contains(tracks,regex=True)]


    for mouse in mice:
            df_mouse = df[df['MouseName'].str.match(mouse)]
            for i in range(df_mouse.shape[0]):
                sess = df_mouse.iloc[i]
                dirbase = os.path.join("G:\\My Drive\\Figures\\TwoTower\\PlaceCells\\SingleCells",
                        mouse,"%s_%s" %(sess['DateFolder'],sess['SessionNumber']))

                try:
                    os.makedirs(dirbase)
                except:
                    print("directory already made")


                try:
                    plot_highSI_cells(sess,dirbase)
                except:
                    print(sess)
