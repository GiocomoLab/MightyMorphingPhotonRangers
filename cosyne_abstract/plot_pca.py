import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt
import utilities as u
import preprocessing as pp
import behavior as b
import pickle
from sklearn.decomposition import PCA
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D

def plot_pca(C,VRDat,pcnt):
    pca = PCA()
    trialMask = (VRDat['pos']>0) & (VRDat['pos']<445)
    X = pca.fit_transform(C)

    print(X.shape)
    # skree plots
    f = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #f,axarr = plt.subplots(2,2,figsize=[15,15])
    axarr = []

    XX = X[trialMask[:X.shape[0]]]
    XX = XX[::5,:]
    morph = VRDat.loc[trialMask,'morph']._values
    morph = morph[::5]
    pos = VRDat.loc[trialMask,'pos']._values
    pos = pos[::5]

    time = VRDat.loc[trialMask,'time']._values
    time = time[::5]


    ax=f.add_subplot(131,projection='3d')
    s_cxt=ax.scatter(XX[:,0],XX[:,1],XX[:,2],c=morph,cmap='cool',s=2)


    aax = f.add_subplot(132,projection='3d')
    s_pos=aax.scatter(XX[:,0],XX[:,1],XX[:,2],c=pos,cmap='magma',s=2)

    aaax = f.add_subplot(133,projection='3d')
    s_pos=aaax.scatter(XX[:,0],XX[:,1],XX[:,2],c=time,cmap='viridis',s=2)

    return f,[ax, aax, aaax]


if __name__ == '__main__':
    mice = ['4139219.2', '4139219.3', '4139224.2', '4139224.3', '4139224.5',
     '4139251.1','4139251.2','4139260.1','4139260.2','4139261.2','4139266.3','4139265.4',
     '4139265.3','4139265.5']
    df = pp.load_session_db()
    df = df[df['RewardCount']>20]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])

    for mouse in mice:
        dirbase = "G:\\My Drive\\Figures\\TwoTower\\PCA\\%s\\" % mouse
        try:
            os.makedirs(dirbase)
        except:
            print("directory already made")

        df_mouse = df[df['MouseName'].str.match(mouse)]
        for i in range(df_mouse.shape[0]):
            fname = "%s\\%s_%s_%d_" % (dirbase,mouse,df_mouse['DateFolder'].iloc[i],df_mouse['SessionNumber'].iloc[i])
            try:
                VRDat,C, S, A = pp.load_scan_sess(df_mouse.iloc[i],fneu_coeff=.7,analysis='s2p')
                trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
                pcnt = np.zeros([VRDat.shape[0],])
                for i,(start,stop) in enumerate(zip(tstart_inds,teleport_inds)):
                    pcnt[start:stop] = int(trial_info['rewards'][i]>0)

                C = u.df(C)
                S = sp.ndimage.filters.gaussian_filter1d(S,10.,axis=0)
                f,ax = plot_pca(C,VRDat,pcnt)
                f.savefig(fname+"_C_pca.png",format='png')

                f,ax = plot_pca(S,VRDat,pcnt)
                f.savefig(fname+"_S_pca.png",format='png')
            except:
                print(df_mouse.iloc[i])
