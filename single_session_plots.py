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




def single_session_figs(sess,savefigs = True,dir = "G:\\My Drive\\Figures\\TwoTower\\SingleSession"):

    VRDat, C, S, A = pp.load_scan_sess(sess,fneu_coeff=.7,analysis='s2p')
    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    S_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(S,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values,bin_size=10)
    S_morph_dict = u.trial_type_dict(S_trial_mat,trial_info['morphs'])
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])

    effMorph = (trial_info['morphs']+trial_info['wallJitter']+trial_info['bckgndJitter']+.2)/1.4
    reward_pos = trial_info['reward_pos']
    reward_pos[np.isnan(reward_pos)]= 480

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


    # plot behavior
    if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout','Reversal','Reversal_noTimeout'):
        # use existing plotting functions



        if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout'):
            f_lick, (ax_lick, meanlr_ax, lickrat_ax) = b.lick_plot_task(lick_morph_dict,edges,max_pos=max_pos,smooth=False,
                                            rzone0=(250.,315),rzone1=(350,415))
        else:
            f_lick, (ax_lick, meanlr_ax, lickrat_ax) = b.lick_plot_task(lick_morph_dict,edges,max_pos=max_pos,smooth=False,
                                            rzone1=(350.,415),rzone0=(250,315))

        if sess['Track'] in ('TwoTower_noTimeout','TwoTower_Timeout'):
            f_speed,ax_speed = b.plot_speed_task(centers,speed_morph_dict,trial_info['morphs'],
                                                    rzone0=(250.,315),rzone1=(350,415))
        else:
            f_speed,ax_speed = b.plot_speed_task(centers,speed_morph_dict,trial_info['morphs'],
                                                    rzone1=(250.,315),rzone0=(350,415))
    else:
        f_lick, axarr_lick = b.behavior_raster_foraging(lick_trial_mat/np.nanmax(lick_trial_mat.ravel()),
                                                centers,effMorph,reward_pos/480.,smooth=False)
        f_speed,axarr_speed = b.behavior_raster_foraging(speed_trial_mat/np.nanmax(speed_trial_mat.ravel()),
                                                centers,effMorph,reward_pos/480.,smooth=False)


    # PCA
    pcnt = u.correct_trial_mask(trial_info['rewards'],tstart_inds,teleport_inds,S.shape[0])
    S_sm = gaussian_filter1d(S,5,axis=0)
    f_pca,[ax_pca, aax_pca, aaax_pca] = plot_pca(S_sm,VRDat,np.array([]),plot_err=False)

    # DPCA


    # Variance explained


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


    # trial by trial similarity matrix
    S_trial_mat[np.isnan(S_trial_mat)]=0
    S_trial_mat = sp.ndimage.filters.gaussian_filter1d(S_trial_mat,1,axis=1)
    S_tmat = np.reshape(S_trial_mat,[S_trial_mat.shape[0],-1])
    S_tmat = S_tmat/np.linalg.norm(S_tmat,ord=2,axis=-1)[:,np.newaxis]
    S_t_rmat = np.matmul(S_tmat,S_tmat.T)

    f_stsm,axtup_stsm = sm.plot_trial_simmat(S_t_rmat,trial_info,vmax=.7)


    # spectral embedding of single trial similarity matrix
    lem = sk.manifold.SpectralEmbedding(affinity='precomputed',n_components=3)
    X = lem.fit_transform(S_t_rmat)
    f_embed = plt.figure(figsize=[20,20])
    ax_embed3d = f_embed.add_subplot(221, projection='3d')
    ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=effMorph,cmap='cool')
    ax_embed2d = f_embed.add_subplot(222)
    ax_embed2d.scatter(X[:,0],X[:,1],c=effMorph,cmap='cool')
    ax_embed3d = f_embed.add_subplot(223, projection='3d')
    ax_embed3d.scatter(X[:,0],X[:,1],X[:,2],c=np.arange(X.shape[0]),cmap='viridis')
    ax_embed2d = f_embed.add_subplot(224)
    ax_embed2d.scatter(X[:,0],X[:,1],c=np.arange(X.shape[0]),cmap='viridis')

    if savefigs:
        outdir = os.path.join(dir,sess['MouseName'],"%s_%s_%d" % (sess['Track'],sess['DateFolder'],sess['SessionNumber']))

        try:
            os.makedirs(outdir)
        except:
            print("failed to make path",outdir)


        f_pc.savefig(os.path.join(outdir,'placecells.pdf'),format='pdf')
        f_pca.savefig(os.path.join(outdir,'PCA.pdf'),format='pdf')
        f_lick.savefig(os.path.join(outdir,'licks.pdf'),format='pdf')
        f_speed.savefig(os.path.join(outdir,'speed.pdf'),format='pdf')
        f_singlecells.savefig(os.path.join(outdir,'singlecells.pdf'),format='pdf')
        f_SM.savefig(os.path.join(outdir,'morphxpos_simmat.pdf'),format='pdf')
        f_U.savefig(os.path.join(outdir,'morph_simmat.pdf'),format='pdf')
        f_stsm.savefig(os.path.join(outdir,'trial_simmat.pdf'),format='pdf')
        f_embed.savefig(os.path.join(outdir,'simmat_embed.pdf'),format='pdf')

    return
