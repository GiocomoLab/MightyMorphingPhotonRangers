import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter1d
import utilities as u
import preprocessing as pp
import PlaceCellAnalysis as pc
import SimilarityMatrixAnalysis as sm
from plot_pca import plot_pca
import sklearn as sk
from sklearn.decomposition import PCA
import behavior as b
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import plots as ensemble_plots
import ensemble as nmf


def set_ops(change_ops={}):
    ops = {'behavior':True,
        'PCA':True,
        'place cells':True,
        'trial simmats':True,
        'trial NMF': True,
        'savefigs':True}
    for k,v in change_ops.items():
        ops[k]=v
    return ops




def single_session_figs(sess,dir = "G:\\My Drive\\Figures\\TwoTower\\SingleSession", ops={}):

    ops = set_ops(ops)
    if ops['savefigs']:
        outdir = os.path.join(dir,sess['MouseName'],"%s_%s_%d" % (sess['Track'],sess['DateFolder'],sess['SessionNumber']))

        try:
            os.makedirs(outdir)
        except:
            print("failed to make path",outdir)



    VRDat, C, S, A = pp.load_scan_sess(sess,fneu_coeff=.7,analysis='s2p')

    S/=1546
    S[np.isnan(S)]=0.
    C[np.isnan(C)]=0.
    C/=1546
    # S /= np.nanmean(S,axis=0)[np.newaxis,:]
    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    S_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(S,
                                                VRDat['pos']._values,VRDat['tstart']._values,
                                                VRDat['teleport']._values,bin_size=10,
                                                speed = VRDat['speed']._values)
    S_trial_mat[np.isnan(S_trial_mat)]=0
    S_morph_dict = u.trial_type_dict(S_trial_mat,trial_info['morphs'])
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])

    effMorph = (trial_info['morphs']+trial_info['wallJitter']+trial_info['bckgndJitter']+.25)/1.5
    reward_pos = trial_info['reward_pos']
    reward_pos[np.isnan(reward_pos)]= 480

    if ops['behavior']:
        #lick data
        lick_trial_mat= u.make_pos_bin_trial_matrices(VRDat['lick']._values,
                                                            VRDat['pos']._values,
                                                            VRDat['tstart']._values,
                                                            VRDat['teleport']._values,
                                                            mat_only=True,bin_size=10)
        lick_morph_dict = u.trial_type_dict(lick_trial_mat,trial_info['morphs'])
        max_pos = np.copy(trial_info['max_pos'])
        max_pos[max_pos>440]=np.nan

        # plot speed data
        speed_trial_mat = u.make_pos_bin_trial_matrices(VRDat['speed']._values,
                                                        VRDat['pos']._values,
                                                        VRDat['tstart']._values,
                                                        VRDat['teleport']._values,
                                                        mat_only=True,bin_size=10)
        speed_morph_dict = u.trial_type_dict(speed_trial_mat,trial_info['morphs'])


        # # plot behavior
        # if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout','Reversal','Reversal_noTimeout','FreqMorph_Decision','FreqMorph_Timeout'):
        #     # use existing plotting functions
        #
        #
        #
        #     if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout','FreqMorph_Timeout','FreqMorph_Decision'):
        #         f_lick, (ax_lick, meanlr_ax, lickrat_ax) = b.lick_plot_task(lick_morph_dict,edges,max_pos=max_pos,smooth=False,
        #                                         rzone0=(250.,315),rzone1=(350,415))
        #     else:
        #         f_lick, (ax_lick, meanlr_ax, lickrat_ax) = b.lick_plot_task(lick_morph_dict,edges,max_pos=max_pos,smooth=False,
        #                                         rzone1=(350.,415),rzone0=(250,315))
        #
        #     if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout','FreqMorph_Timeout','FreqMorph_Decision'):
        #         f_speed,ax_speed = b.plot_speed_task(centers,speed_morph_dict,trial_info['morphs'],
        #                                                 rzone0=(250.,315),rzone1=(350,415))
        #     else:
        #         f_speed,ax_speed = b.plot_speed_task(centers,speed_morph_dict,trial_info['morphs'],
        #                                                 rzone1=(250.,315),rzone0=(350,415))
        # else:
        f_lick, axarr_lick = b.behavior_raster_foraging(lick_trial_mat/np.nanmax(lick_trial_mat.ravel()),
                                                centers,effMorph,reward_pos/480.,smooth=False)
        f_speed,axarr_speed = b.behavior_raster_foraging(speed_trial_mat/np.nanmax(speed_trial_mat.ravel()),
                                                centers,effMorph,reward_pos/480.,smooth=False)
        if ops['savefigs']:
            f_lick.savefig(os.path.join(outdir,'licks.pdf'),format='pdf')
            f_speed.savefig(os.path.join(outdir,'speed.pdf'),format='pdf')

    if ops['PCA']:
        # PCA
        pcnt = u.correct_trial_mask(trial_info['rewards'],tstart_inds,teleport_inds,S.shape[0])
        # S_sm = gaussian_filter1d(S,5,axis=0)
        # print(np.isnan(S_sm).sum(),np.isnan(S).sum())
        f_pca,[ax_pca, aax_pca, aaax_pca] = plot_pca(C,VRDat,np.array([]),plot_err=False)

        # DPCA


        # Variance explained
        if ops['savefigs']:
            f_pca.savefig(os.path.join(outdir,'PCA.pdf'),format='pdf')

    if ops['place cells']:
        ################ Place cells
        masks, FR, SI = pc.place_cells_calc(S, VRDat['pos']._values,trial_info,
                        VRDat['tstart']._values, VRDat['teleport']._values,
                        method='bootstrap',correct_only=False,speed=VRDat.speed._values,
                        win_trial_perm=True,morphlist=np.unique(trial_info['morphs']).tolist())

        # plot place cells by morph
        f_pc, ax_pc = pc.plot_placecells(S_morph_dict,masks)

        # number in each environment
        print('morph 0 place cells = %g out of %g , %f ' % (masks[0].sum(), masks[0].shape[0], masks[0].sum()/masks[0].shape[0]))
        print('morph 1 place cells = %g out of %g, %f' % (masks[1].sum(), masks[1].shape[0], masks[1].sum()/masks[1].shape[0]))

        # reward cell plot
        # make tensor for reward location centered position



        # single cell plots
        f_singlecells = pc.plot_top_cells(S_trial_mat,masks,SI,effMorph)

        # position by morph similarity matrix averaging trials
        SM = sm.morph_simmat(S_morph_dict,corr=True)
        m=np.unique(trial_info['morphs']).size
        U  = sm.morph_mean_simmat(SM,m)
        f_SM, ax_SM = sm.plot_simmat(SM,m)
        f_U,ax_U = plt.subplots()
        ax_U.imshow(U,cmap='Greys')

        if ops['savefigs']:
            f_pc.savefig(os.path.join(outdir,'placecells.pdf'),format='pdf')
            f_singlecells.savefig(os.path.join(outdir,'singlecells.pdf'),format='pdf')
            f_SM.savefig(os.path.join(outdir,'morphxpos_simmat.pdf'),format='pdf')
            f_U.savefig(os.path.join(outdir,'morph_simmat.pdf'),format='pdf')
            with open(os.path.join(outdir,'pc_masks.pkl'),'wb') as f:
                pickle.dump({'masks':masks},f)

    if ops['trial simmats']:
        # trial by trial similarity matrix
        rmask = trial_info['rewards']==0
        S_trial_mat[np.isnan(S_trial_mat)]=0
        S_trial_mat = sp.ndimage.filters.gaussian_filter1d(S_trial_mat,1,axis=1)
        if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout','FreqMorph_Timeout','FreqMorph_Decision'):
            S_tmat = np.reshape(S_trial_mat[:,:20,:],[S_trial_mat.shape[0],-1])
        else:
            S_tmat = np.reshape(S_trial_mat,[S_trial_mat.shape[0],-1])
        S_tmat = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
        S_t_rmat = np.matmul(S_tmat,S_tmat.T)

        f_stsm,axtup_stsm = sm.plot_trial_simmat(S_t_rmat,trial_info)

        lar = np.zeros(effMorph.shape)
        for trial in range(effMorph.shape[0]):
            mask0 = trial_info['morphs']==0
            mask1 = trial_info['morphs']==1
            if trial_info['morphs'][trial]==0:
                mask0[trial]=False
            elif trial_info['morphs'][trial]==1:
                mask1[trial]=False

            centroid0, centroid1 = np.nanmean(S_tmat[mask0,:],axis=0), np.nanmean(S_tmat[mask1,:],axis=0)
            centroid0/np.linalg.norm(centroid0,ord=2)
            centroid1/np.linalg.norm(centroid1,ord=2)

            lar[trial]= np.log(np.dot(S_tmat[trial,:],centroid0)/np.dot(S_tmat[trial,:],centroid1))

        f_lar,ax_lar = plt.subplots()
        ax_lar.scatter(effMorph,lar,c=effMorph,cmap='cool')
        ax_lar.scatter(effMorph[rmask],lar[rmask],c='black')
        # spectral embedding of single trial similarity matrix
        # lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
        # X = lem.fit_transform(S_t_rmat)

        [w,V]=np.linalg.eig(S_t_rmat)
        order = np.argsort(w)[::-1]
        w = w[order]
        V=V[:,order]
        X =  np.matmul(S_t_rmat,V[:,:3])
        f_embed = plt.figure(figsize=[20,20])
        ax_embed3d = f_embed.add_subplot(221, projection='3d')
        ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=effMorph,cmap='cool')


        ax_embed3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')
        ax_embed2d = f_embed.add_subplot(222)
        ax_embed2d.scatter(X[:,0],X[:,1],c=effMorph,cmap='cool')
        ax_embed2d.scatter(X[rmask,0],X[rmask,1],c='black')
        ax_embed3d = f_embed.add_subplot(223, projection='3d')
        ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
        ax_embed2d = f_embed.add_subplot(224)
        ax_embed2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

        lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
        X = lem.fit_transform(S_t_rmat)

        f_se = plt.figure(figsize=[20,20])
        ax_se3d = f_se.add_subplot(221, projection='3d')
        ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=effMorph,cmap='cool')
        ax_se3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')

        ax_se2d = f_se.add_subplot(222)
        ax_se2d.scatter(X[:,0],X[:,1],c=effMorph,cmap='cool')
        ax_se2d.scatter(X[rmask,0],X[rmask,1],c='black')

        ax_se3d = f_se.add_subplot(223, projection='3d')
        ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
        ax_se2d = f_se.add_subplot(224)
        ax_se2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

        if ops['savefigs']:
            f_stsm.savefig(os.path.join(outdir,'trial_simmat.pdf'),format='pdf')
            f_embed.savefig(os.path.join(outdir,'simmat_embed.pdf'),format='pdf')
            f_se.savefig(os.path.join(outdir,'simmat_spectembed.pdf'),format='pdf')
            f_lar.savefig(os.path.join(outdir,'lar.pdf'),format='pdf')


    if ops['trial simmats'] and ops['place cells']:
        # trial by trial similarity matrix
        cellmask = np.zeros([S.shape[1],])<1
        for k,v in masks.items():
            cellmask = cellmask | v

        S_trial_mat_pc = S_trial_mat[:,:,cellmask]
        if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout','FreqMorph_Timeout','FreqMorph_Decision'):
            S_tmat = np.reshape(S_trial_mat_pc[:,:20,:],[S_trial_mat.shape[0],-1])
        else:
            S_tmat = np.reshape(S_trial_mat_pc,[S_trial_mat.shape[0],-1])
        S_tmat = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
        S_t_rmat = np.matmul(S_tmat,S_tmat.T)

        f_stsm,axtup_stsm = sm.plot_trial_simmat(S_t_rmat,trial_info)

        lar = np.zeros(effMorph.shape)
        for trial in range(effMorph.shape[0]):
            mask0 = trial_info['morphs']==0
            mask1 = trial_info['morphs']==1
            if trial_info['morphs'][trial]==0:
                mask0[trial]=False
            elif trial_info['morphs'][trial]==1:
                mask1[trial]=False

            centroid0, centroid1 = np.nanmean(S_tmat[mask0,:],axis=0), np.nanmean(S_tmat[mask1,:],axis=0)
            centroid0/np.linalg.norm(centroid0,ord=2)
            centroid1/np.linalg.norm(centroid1,ord=2)

            lar[trial]= np.log(np.dot(S_tmat[trial,:],centroid0)/np.dot(S_tmat[trial,:],centroid1))

        f_lar,ax_lar = plt.subplots()
        ax_lar.scatter(effMorph,lar,c=effMorph,cmap='cool')
        ax_lar.scatter(effMorph[rmask],lar[rmask],c='black')
        # spectral embedding of single trial similarity matrix
        # lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
        # X = lem.fit_transform(S_t_rmat)

        [w,V]=np.linalg.eig(S_t_rmat)
        order = np.argsort(w)[::-1]
        w = w[order]
        V=V[:,order]
        X =  np.matmul(S_t_rmat,V[:,:3])
        f_embed = plt.figure(figsize=[20,20])
        ax_embed3d = f_embed.add_subplot(221, projection='3d')
        ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=effMorph,cmap='cool')


        ax_embed3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')
        ax_embed2d = f_embed.add_subplot(222)
        ax_embed2d.scatter(X[:,0],X[:,1],c=effMorph,cmap='cool')
        ax_embed2d.scatter(X[rmask,0],X[rmask,1],c='black')
        ax_embed3d = f_embed.add_subplot(223, projection='3d')
        ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
        ax_embed2d = f_embed.add_subplot(224)
        ax_embed2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

        lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
        X = lem.fit_transform(S_t_rmat)

        f_se = plt.figure(figsize=[20,20])
        ax_se3d = f_se.add_subplot(221, projection='3d')
        ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=effMorph,cmap='cool')
        ax_se3d.scatter(X[rmask,0],X[rmask,1],X[rmask,2],c='black')

        ax_se2d = f_se.add_subplot(222)
        ax_se2d.scatter(X[:,0],X[:,1],c=effMorph,cmap='cool')
        ax_se2d.scatter(X[rmask,0],X[rmask,1],c='black')

        ax_se3d = f_se.add_subplot(223, projection='3d')
        ax_se3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
        ax_se2d = f_se.add_subplot(224)
        ax_se2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

        if ops['savefigs']:
            f_stsm.savefig(os.path.join(outdir,'trial_simmat_pc.pdf'),format='pdf')
            f_embed.savefig(os.path.join(outdir,'simmat_embed_pc.pdf'),format='pdf')
            f_se.savefig(os.path.join(outdir,'simmat_spectembed_pc.pdf'),format='pdf')
            f_lar.savefig(os.path.join(outdir,'lar_pc.pdf'),format='pdf')


    if ops['trial NMF']:

        S_tmat = np.reshape(S_trial_mat,[S_trial_mat.shape[0],-1])
        trialmask = np.zeros([S_tmat.shape[0]])
        trialmask[np.random.permutation(S_tmat.shape[0])[:int(S_tmat.shape[0]/2)]]=1
        trialmask = trialmask>0.
        trialmask[:3]=False

        results = nmf.fit_ensemble(S_tmat[trialmask,:],np.arange(1,11),n_replicates=5)
        f_ens,ax_ens = plt.subplots()
        ensemble_plots.plot_rmse(results,ax=ax_ens)

        # assuming 2 is the number of components
        W_train,H_train = results[2]['factors'][0][0], results[2]['factors'][0][1]
        W_test = np.matmul(S_tmat[~trialmask,:],H_train.T)
        f_nmf,ax_nmf = plt.subplots(3,2,figsize=[10,15])
        xtrain_arr = np.arange(S_tmat.shape[0])[trialmask]
        xtest_arr = np.arange(S_tmat.shape[0])[~trialmask]
        ax_nmf[0,0].scatter(xtrain_arr,W_train[:,0],c=effMorph[trialmask],cmap='cool')
        ax_nmf[0,0].set_title('train component 1')
        ax_nmf[0,1].scatter(xtest_arr,W_test[:,0],c=effMorph[~trialmask],cmap='cool')
        ax_nmf[0,1].set_title('test component 1')

        ax_nmf[1,0].scatter(xtrain_arr,W_train[:,1],c=effMorph[trialmask],cmap='cool')
        ax_nmf[1,0].set_title('train component 2')
        ax_nmf[1,1].scatter(xtest_arr,W_test[:,1],c=effMorph[~trialmask],cmap='cool')
        ax_nmf[1,1].set_title('test component 2')

        ax_nmf[2,0].scatter(W_train[:,0],W_train[:,1],c=effMorph[trialmask],cmap='cool')
        ax_nmf[2,0].set_title('train')
        ax_nmf[2,1].scatter(W_test[:,0],W_test[:,1],c=effMorph[~trialmask],cmap='cool')
        ax_nmf[2,1].set_title('test')

        if ops['savefigs']:
            f_ens.savefig(os.path.join(outdir,'nmf_fitting.pdf'),format='pdf')
            f_nmf.savefig(os.path.join(outdir,'nmf_traintest.pdf'),format='pdf')


    return
