import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import numpy as np
import matplotlib.pyplot as plt

import utilities as u
import preprocessing as pp
import behavior as b
import SimilarityMatrixAnalysis as sm
import scipy as sp
import PlaceCellAnalysis as pc
from SplineEncodingModel.LinearRegressionSpline import EncodingModel, NBDecodingModel
import matplotlib.gridspec as gridspec




def run_cross_val(C,VRDat,fr=15.46):
'''run population decoding for '''
    C /=fr*100 # scale down

    Shat = np.zeros(S.shape)
Shat[:]=np.nan

for i, test in enumerate(test_mask):
    test = test & (VRDat['pos']._values>0) & (VRDat['pos']._values<450)

    train = ~test & (VRDat['pos']._values>0) & (VRDat['pos']._values<450)
    print(train.shape[0])
    nb = NBDecodingModel()

    nb.poisson_fit(VRDat['pos']._values[train],effMorph[train],C[train,:])
    # predict on left out data
    Shat[test,:]=nb.poisson_predict_rate(VRDat['pos']._values[test],effMorph[test])

    mask = np.isnan(Shat[:,0])
Rvec = np.diagonal(1/(C.shape[0]-mask.sum())*np.matmul(sp.stats.zscore(C[~mask,:],axis=0).T,sp.stats.zscore(Shat[~mask,:],axis=0)))


cellmask = Rvec>.3

L_XC = np.zeros([S.shape[0],50,50])
for i, test in enumerate(test_mask):
    print('start fold ', i)
#     test = test #& (VRDat['pos']._values>0) & (VRDat['pos']._values<450)
    train = ~test & (VRDat['pos']._values>0) & (VRDat['pos']._values<450)

    nb = NBDecodingModel()
    C_fit = C[train,:]
    nb.poisson_fit(VRDat['pos']._values[train],effMorph[train],C_fit[:,cellmask])

    nb.poisson_predict_rate_decodingDM()

    # test on remaining time points
    C_decoding = C[test,:]
    C_decoding = C_decoding[:,cellmask]
    print('decoding fold', i)
    print("\t total timepoints =",test.sum())
#     infmask =
    L_XC[test,:,:]= nb.poisson_decode(C_decoding)

    # bin position
pos_plot = VRDat['pos']/450.*50

# get lick position
lick_pos = u.lick_positions(VRDat.lick._values,VRDat.pos._values)/450.*50


# get reward position
reward_pos = u.lick_positions(VRDat.reward._values,VRDat.pos._values)/450.*50

print(np.isnan(pos_plot).sum(),np.isnan(lick_pos).sum(),np.isnan(reward_pos).sum())

# L_XC[L_XC==0]=1E-10

# plot single trial results
for j, (start,stop) in enumerate(zip(tstart_inds.tolist(),teleport_inds.tolist())):
#     if j <2:
#     print(start,stop,np.isnan(np.log(L_XC)).sum(),np.isinf(np.log(L_XC)).sum())
    gs = gridspec.GridSpec(8,1)
    f = plt.figure(figsize=[10,10])


    x = np.arange(stop-start)


    m_ax = f.add_subplot(gs[0:4,:])
    m_ax.imshow(np.log(L_XC[start:stop,:,:].sum(axis=1)).T,aspect='auto',cmap='cividis',zorder=0,alpha=.4)
    m_ax.scatter(x,effMorph[start:stop]*50,marker='o',color=plt.cm.cool(effMorph[start:stop].mean()),zorder=1)
    m_ax.set_xlim([0,x[-1]])

    pos_ax = f.add_subplot(gs[4:,:])
    pos_ax.imshow(np.log(L_XC[start:stop,:,:].sum(axis=2)).T,aspect='auto',cmap='magma',zorder=0,alpha=.4)
    pos_ax.plot(x,pos_plot[start:stop],linewidth=2,zorder=1,alpha=1)
    pos_ax.scatter(x,lick_pos[start:stop],marker='x',color='green',zorder=2)
    pos_ax.scatter(x,reward_pos[start:stop],marker='o',color='blue',zorder=2)
    pos_ax.set_xlim([0,x[-1]])
