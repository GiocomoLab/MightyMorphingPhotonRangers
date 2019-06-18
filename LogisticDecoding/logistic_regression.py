import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import numpy as np
import scipy as sp
from scipy.ndimage import filters
import os
os.sys.path.append('../')

import utilities as u
import preprocessing as pp
from functools import reduce
import pickle
import matplotlib.gridspec as gridspec


def single_session(sess,smooth=True):
    VRDat,C, S, A = pp.load_scan_sess(sess)
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    S_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(S,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values)
    if smooth:
        S = sp.ndimage.filters.gaussian_filter1d(C,3,axis=0)

    bin_edges= np.arange(0,451,20)
    bin_edges[-1]=455
    nbins = bin_edges.shape[0]-1
    pos_mask = VRDat.pos._values>=0

    train_mask = pos_mask & ((VRDat.morph==1) | (VRDat.morph==0.))

    # lr = LogisticRegression(C=.001,penalty='l2',class_weight="balanced",multi_class='multinomial',solver='lbfgs')
    lr = LogisticRegressionCV(Cs=5,penalty='l2',class_weight="balanced",multi_class='multinomial')
    X = np.digitize(VRDat.pos._values,bin_edges)+nbins*VRDat.morph._values*pos_mask
    Xhat =np.zeros([X.shape[0],nbins*2])
    train_trials = (trial_info['morphs']==0) | (trial_info['morphs']==1)
    LOO = u.LOTrialO(tstart_inds[train_trials],teleport_inds[train_trials],S.shape[0])
    for i,(train,test) in enumerate(LOO):
        ttrain = train & pos_mask
        lr.fit(S[ttrain,:],X[ttrain])
        Xhat[test,:]=lr.predict_proba(S[test])

    lr.fit(S[train_mask,:],np.digitize(VRDat.pos._values[train_mask],bin_edges)+(bin_edges.shape[0]-1)*VRDat.morph._values[train_mask])
    Xhat[~train_mask]=lr.predict_proba(S[~train_mask,:])
    return Xhat

def single_session_multicontext(sess,smooth=True):
    pass

def plot_decoding(data_dict,rzone0=[250,315],rzone1=[350,415],save=False,
                prefix=None,plot_rzone=False):
    rzone0 = [i/20 for i in rzone0]
    rzone1 = [i/20 for i in rzone1]

    tstarts,teleports = data_dict['tstarts'],data_dict['teleports']
    pos_binned = data_dict['pos_binned']
    Xhat = data_dict['Xhat']
    morphs,rewards = data_dict['morphs'],data_dict['rewards']
    lick_pos = data_dict['lick pos']
    npos_bins = int(Xhat.shape[1]/2)
    # print(npos_bins)
    I = Xhat[:,:npos_bins].sum(axis=1)
    Xhat_smooth = 0*Xhat
    Xhat_smooth[:,:npos_bins]= sp.ndimage.filters.gaussian_filter(Xhat[:,:npos_bins],[2,2])
    Xhat_smooth[:,npos_bins:]= sp.ndimage.filters.gaussian_filter(Xhat[:,npos_bins:],[2,2])

    for t,(start,stop) in enumerate(zip(tstarts.tolist(),teleports.tolist())):


        gs = gridspec.GridSpec(8,1)
        f = plt.figure(figsize=[10,10])
        ax = f.add_subplot(gs[0:6,:])
        #f, ax = plt.subplots(2,1,figsize= [15,5],sharex=True)
        ax.imshow(Xhat_smooth[start:stop,:].T,aspect = 'auto',cmap='magma',alpha=.4,zorder=2)
        ax.axhline(Xhat.shape[1]/2,xmin=0,xmax=stop-start,color='white',linewidth=5,zorder=10)

        ax.plot(pos_binned[start:stop]-1, color = plt.cm.cool(0.),linewidth=2,zorder=0,alpha=.5)
        ax.plot(pos_binned[start:stop]+npos_bins-1, color = plt.cm.cool(1.),linewidth=2,zorder=0,alpha=.5)
        if plot_rzone:
            ax.fill_between(np.arange(stop-start),rzone0[0],y2 = rzone0[1],color=plt.cm.cool(0.),alpha=.2)
            ax.fill_between(np.arange(stop-start),rzone1[0],y2 = rzone1[1],color=plt.cm.cool(1.),alpha=.2)
            ax.fill_between(np.arange(stop-start),rzone0[0]+npos_bins,y2 = rzone0[1]+npos_bins,color=plt.cm.cool(0.),alpha=.2)
            ax.fill_between(np.arange(stop-start),rzone1[0]+npos_bins,y2 = rzone1[1]+npos_bins,color=plt.cm.cool(1.),alpha=.2)
        ax.set_xlim([0,stop-start])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Binned Position')
        ax.set_ylim([Xhat.shape[1],0])

        x = np.arange(stop-start)

        ax.scatter(x,lick_pos[start:stop],s=50,marker='x',color='blue',alpha=.5,zorder=1)
        ax.scatter(x,lick_pos[start:stop]+npos_bins,s=50,marker='x',color='blue',alpha=.5,zorder=1)
        ax.set_title("trial %d morph %f reward %f" % (t,morphs[t],rewards[t]))


        aax = f.add_subplot(gs[6:,:],sharex=ax)
        aax.scatter(x,I[start:stop],c=plt.cm.cool(1-I[start:stop]))
        # aax.axhline(0,xmin=0,xmax=x.shape[0])
        aax.set_xlim([0,x.shape[0]])
        aax.set_ylim([-.2,1.2])
        aax.set_xlabel('time')
        aax.set_ylabel('P(morph=0)')
        aax.set_yticks([0,1])


        if save:
            f.savefig(os.path.join(prefix,"trial%d_morph%2f_reward%d.pdf" % (t,morphs[t],int(rewards[t]))),format='pdf')
    return f, ax, aax



def confusion_matrix(data_dict,save=False,check_pcnt = True,
                    check_omissions = False,plot=True):
    Xhat,pos = data_dict['Xhat'],data_dict['pos']
    tstarts,teleports = data_dict['tstarts'],data_dict['teleports']
    morphs,pcnt,omissions = data_dict['morphs'],data_dict['pcnt'],data_dict['omissions']

    d_trial_mat, tr, edges, centers = u.make_pos_bin_trial_matrices(Xhat,pos,tstarts,teleports,bin_size=20)
    d_m_dict = u.trial_type_dict(d_trial_mat,morphs)

    keys = np.unique(morphs)
    c_all = np.zeros([d_trial_mat.shape[-1],d_trial_mat.shape[1]*keys.shape[0]])
    if check_pcnt:
        c_m0lick = np.copy(c_all)
        c_m1lick = np.copy(c_all)
    if check_omissions:
        c_omissions = np.copy(c_all)
        c_no_omit = np.copy(c_all)
    for n,key in enumerate(keys.tolist()):
        all_mask = morphs==key

        c_all[:,n*d_trial_mat.shape[1]:(n+1)*d_trial_mat.shape[1]]=np.nansum(d_trial_mat[all_mask,:,:],axis=0).T

        if check_pcnt:
            m0_mask = all_mask & (pcnt==0)
            m1_mask = all_mask & (pcnt==1)

            c_m0lick[:,n*d_trial_mat.shape[1]:(n+1)*d_trial_mat.shape[1]]=np.nansum(d_trial_mat[m0_mask,:,:],axis=0).T
            c_m1lick[:,n*d_trial_mat.shape[1]:(n+1)*d_trial_mat.shape[1]]=np.nansum(d_trial_mat[m1_mask,:,:],axis=0).T

        if check_omissions:
            o_mask = all_mask & (omissions==1)
            no_o_mask = all_mask & (omissions==0)
            c_omissions[:,n*d_trial_mat.shape[1]:(n+1)*d_trial_mat.shape[1]]=np.nansum(d_trial_mat[o_mask,:,:],axis=0).T
            c_no_omit[:,n*d_trial_mat.shape[1]:(n+1)*d_trial_mat.shape[1]]=np.nansum(d_trial_mat[no_o_mask,:,:],axis=0).T


    f,ax = plt.subplots()
    ax.imshow(np.log(c_all),cmap='viridis',vmin=0,vmax=.3,aspect='auto')
    ax.set_xlabel('True Label')
    ax.set_ylabel('Decoded Label')
    if save:
        pass

    if check_pcnt:
        c_m0lick[np.isnan(c_m0lick)]=0
        c_m1lick[np.isnan(c_m1lick)]=0

        f_pcnt,ax_pcnt = plt.subplots(2,1,figsize=[5,10])
        ax_pcnt[0].imshow(c_m0lick,cmap='viridis',vmin=0, vmax = .3,aspect='auto')
        ax_pcnt[1].imshow(c_m1lick,cmap='viridis',vmin=0, vmax = .3,aspect='auto')

        return c_all,c_m0lick,c_m1lick, (f,ax), (f_pcnt,ax_pcnt)

    if check_omissions:

        f_o,ax_o = plt.subplots(2,1,figsize=[5,10])
        ax_o[0].imshow(c_omissions,cmap='viridis',vmin=0, vmax = .3,aspect='auto')
        ax_o[1].imshow(c_no_omit,cmap='viridis',vmin=0, vmax = .3,aspect='auto')
        return c_all,c_omissions,c_no_omit, (f,ax), (f_o,ax_o)

    if not check_pcnt and not check_omissions:
        return c_all, (f,ax)


def plot_llr(llr_pos,effMorph,save=False,centers = None,nbins=5):

    if centers is None:
        centers = np.linspace(0,450,num=llr_pos.shape[1])

    morph_edges = np.linspace(0,1,num=nbins+1)
    dm = morph_edges[1]-morph_edges[0]
    morph_centers = morph_edges[1:]-dm
    morph_dig = np.digitize(effMorph,morph_edges[1:],right=True)
    morph_binned = np.zeros(effMorph.shape)
    for i,c in enumerate(morph_centers.tolist()):
        morph_binned[morph_dig==i]=c


    mu_pos = np.zeros([nbins,llr_pos.shape[1]])
    sem_pos = np.zeros([nbins,llr_pos.shape[1]])
    for b in range(nbins):
        mu_pos[b,:] = np.nanmean(llr_pos[morph_dig==b,:],axis=0)
        sem_pos[b,:]=np.nanstd(llr_pos[morph_dig==b,:],axis=0)


    # actually plot stuff
    f_mp,ax_mp = plt.subplots()
    for b in range(nbins):
        ax_mp.plot(centers,mu_pos[b,:],color=plt.cm.cool(morph_centers[b]))
    ax_mp.set_xlabel('position')
    ax_mp.set_ylabel('LLR')


    f_pos,ax_pos = plt.subplots(1,nbins,figsize=[20,5])
    for b in range(nbins):
        ax_pos[b].fill_between(centers,mu_pos[0,:]+sem_pos[0,:],y2=mu_pos[0,:]-sem_pos[0,:],
                    color=plt.cm.cool(0.),alpha=.4)
        ax_pos[b].fill_between(centers,mu_pos[-1,:]+sem_pos[-1,:],y2=mu_pos[-1,:]-sem_pos[-1,:],
                    color=plt.cm.cool(1.),alpha=.4)

        _llr = llr_pos[morph_dig==b,:]
        _em = effMorph[morph_dig==b]
        for row in range(_llr.shape[0]):
            ax_pos[b].plot(centers,_llr[row,:],color=plt.cm.cool(_em[row]),linewidth=1)
    ax_pos[0].set_xlabel('position')


    msort = np.argsort(effMorph)
    f_mat = plt.figure(figsize=[10,10])
    gs = gridspec.GridSpec(1,6)
    ax_mat = f_mat.add_subplot(gs[:,:5])
    ax_mat.imshow(-llr_pos[msort,:],cmap='cool',aspect='auto')
    ax_trial = f_mat.add_subplot(gs[:,-1])
    ax_trial.scatter(effMorph[msort][::-1],np.arange(effMorph.shape[0]),c=effMorph[msort][::-1],cmap='cool')
    ax_trial.set_ylim([0,effMorph.shape[0]])



    llr_avg = np.nanmean(llr_pos,axis=1)
    f_trial_avg ,ax_trial_avg = plt.subplots()
    ax_trial_avg.scatter(np.arange(effMorph.shape[0]),llr_avg[msort],c=effMorph[msort],cmap='cool')
    return (f_mp, ax_mp), (f_pos,ax_pos), (f_mat,ax_mat,ax_trial), (f_trial_avg,ax_trial_avg)


######################################

if __name__ == '__main__':


    # mice = ['4139265.3','4139265.4','4139265.5','4222153.1', '4222153.2', '4222154.1']
    mice = ['4139224.3','4139224.5','4139251.1','4139260.1','4139261.2']
    df = pp.load_session_db()
    df = df[df['RewardCount']>30]
    df = df[df['Imaging']==1]
    df = df.sort_values(['MouseName','DateTime','SessionNumber'])
    tracks = 'TwoTower_noTimeout|TwoTower_Timeout|Reversal_noTimeout|Reversal|TwoTower_foraging'
    df = df[df['Track'].str.contains(tracks,regex=True)]


    for mouse in mice:
            df_mouse = df[df['MouseName'].str.match(mouse)]
            for i in range(df_mouse.shape[0]):
                sess = df_mouse.iloc[i]
                dirbase = os.path.join("G:\\My Drive\\Figures\\TwoTower\\LogReg_smooth",mouse)

                try:
                    os.makedirs(dirbase)
                except:
                    print("directory already made")

                try:


                    fname = "%s\\%s_%d_Xhat.pkl" % (dirbase,sess['DateFolder'],sess['SessionNumber'])
                    print(fname)
                    if os.path.isfile(fname):
                        print("LOOCV results exist")
                        with open(fname,"rb") as f:
                            d = pickle.load(f)
                            Xhat=d['Xhat']
                    #
                    else:
                        print("LOOCV results don't exist")
                        Xhat = single_session(sess)
                        with open(fname,"wb") as f:
                            pickle.dump({'Xhat':Xhat},f)


                    VRDat,C, S, A = pp.load_scan_sess(sess)
                    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
                    S_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(S,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values)
                    effMorph = trial_info['morphs']+trial_info['bckgndJitter']+trial_info['towerJitter']
                    effMorph = (effMorph+.2)/1.4


                    #
                    half = int(Xhat.shape[1]/2)
                    num,den = Xhat[:,:half].sum(axis=1),Xhat[:,half:].sum(axis=1)
                    llr = np.log(num)-np.log(den)
                    llr_trial_mat = u.make_pos_bin_trial_matrices(llr,VRDat.pos._values,VRDat['tstart']._values,VRDat['teleport']._values,mat_only=True,bin_size=10)
                    #



                    prefix = os.path.join(dirbase,"%s_%d" % (sess['DateFolder'],sess['SessionNumber']))
                    try:
                        os.makedirs(prefix)
                    except:
                        print("prefix already made")

                    bin_edges= np.arange(0,451,20)
                    bin_edges[-1]=455
                    data_dict = {'tstarts': tstart_inds, 'teleports':teleport_inds, 'pos_binned': np.digitize(VRDat.pos._values,bin_edges),
                            'Xhat':Xhat,'lick pos':u.lick_positions(VRDat.lick._values,np.digitize(VRDat.pos._values,bin_edges)),
                            'morphs':trial_info['morphs'],'rewards':trial_info['rewards']}



                    if mouse in {'4139251.1','4139260.1','4139261.2'}:
                        plot_decoding(data_dict,rzone0=[350,415],rzone1=[250,315],save=True,
                                           prefix=prefix)
                    else:
                        # plot_decoding(data_dict,save=True, prefix=prefix,plot_rzone=False)
                        (f_mp,ax_mp), (f_pos,ax_pos), (f_mat,ax_mat,ax_trial), (f_trial_avg,ax_trial_avg) = plot_llr(llr_trial_mat,effMorph)

                        f_mp.savefig(os.path.join(dirbase,"%s_%d_llr_mean.pdf" % (sess['DateFolder'],sess['SessionNumber'])),format='pdf')
                        f_pos.savefig(os.path.join(dirbase,"%s_%d_llr_singletrials.pdf" % (sess['DateFolder'],sess['SessionNumber'])),format='pdf')
                        f_mat.savefig(os.path.join(dirbase,"%s_%d_llr_mat.pdf" % (sess['DateFolder'],sess['SessionNumber'])),format='pdf')
                        f_trial_avg.savefig(os.path.join(dirbase,"%s_%d_llr_trialavg.pdf" % (sess['DateFolder'],sess['SessionNumber'])),format='pdf')
                    # confmat,f_cmat,ax_cmat = confusion_matrix(data_dict,save=False,check_pcnt = True,
                    #                     check_omissions = False,plot=True)
                    # f.savefig(os.path.join(prefix,"trial%d_morph%2f_reward%d.pdf" % (t,morphs[t],int(rewards[t]))),format='pdf')

                except:
                    print(sess)
