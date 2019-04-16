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




def single_session_figs(sess,savefigs = True):

    VRDat, C, S, A = pp.load_scan_sess(sess,fneu_coeff=.7,analysis='s2p')
    # get trial by trial info
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    S_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(S,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values)
    S_morph_dict = u.trial_type_dict(S_trial_mat,trial_info['morphs'])
    occ_morph_dict = u.trial_type_dict(occ_trial_mat,trial_info['morphs'])


    # plot behavior
    if sess.scene in ('TwoTower_noTimeout','TwoTower_Timeout','Reversal','Reversal_noTimeout'):
        # use existing plotting functions

        #plot lick data
        lick_trial_mat= u.make_pos_bin_trial_matrices(VRDat['lick']._values,
                                                            VRDat['pos']._values,
                                                            VRDat['tstart']._values,
                                                            VRDat['telepot']._values,
                                                            mat_only=True)
        lick_morph_dict = u.trial_type_dict(lick_trial_mat,trial_info['morphs'])
        max_pos = np.copy(trial_info['max_pos'])
        max_pos[max_pos>440]=np.nan

        if sess.scene in ('TwoTower_noTimeout','TwoTower_Timeout'):
            f_lick, (ax_lick, meanlr_ax, lickrat_ax) = b.lick_plot_task(lick_morph_dict,edges,max_pos=max_pos,smooth=False,
                                            rzone1=(250.,315),rzone0=(350,415))
        else:
            f_lick, (ax_lick, meanlr_ax, lickrat_ax) = b.lick_plot_task(lick_morph_dict,edges,max_pos=max_pos,smooth=False,
                                            rzone1=(350,415),rzone0=(250.,315))

        # plot speed data
        speed_trial_mat = u.make_pos_bin_trial_matrices(VRDat['speed']._values,
                                                        VRDat['pos']._values,
                                                        VRDat['tstart']._values,
                                                        VRDat['telepot']._values,
                                                        mat_only=True)
        speed_morph_dict = u.trial_type_dict(speed_trial_mat,trial_info['morphs'])
        if sess.scene in ('TwoTower_noTimeout','TwoTower_Timeout'):
            f_speed,ax_speed = b.plot_speed_task(centers,speed_morph_dict,trial_info['morphs'])
        else:
            f_speed,ax_speed = b.plot_speed_task(centers,speed_morph_dict,trial_info['morphs'],
                                                rzone1=(350,415),rzone0=(250.,315))

    else:
        pass

    # speed v position



    # licks v position



    # PCA
    pcnt = u.correct_trial_mask(trial_info['rewards'],tstart_inds,teleport_inds)
    S_sm = gaussian_filter1d(S,5,axis=0)
    f_pca,[ax_pca, aax_pca, aaax_pca] = plot_pca(S_sm,VRDat,[],plot_error=False)

    # DPCA


    # Variance explained

    # angle between subspaces



    ################ Place cells
    masks, FR, SI = pc.place_cells_calc(S, VRDat['pos']._values,trial_info,
                    VRDat['tstart']._values, VRDat['teleport']._values,
                    method='bootstrap',correct_only=False,speed=VRDat.speed._values,
                    win_trial_perm=True)

    # plot place cells by morph
    f_pc, ax_pc = pc.plot_placecells(T_morph_dict,masks)

    f_pc.savefig('')
    # number in each environment
    print('morph 0 place cells = %g out of %g , %f ' % (masks[0].sum(), masks[0].shape[0], masks[0].sum()/masks[0].shape[0]))
    print('morph 1 place cells = %g out of %g, %f' % (masks[1].sum(), masks[1].shape[0], masks[1].sum()/masks[1].shape[0]))

    # reward cell plot
    # make tensor for reward location centered position


    gs = gridspec.GridSpec(20,20)

    # single cell plots




    # position by morph similarity matrix averaging trials

    # trial by trial similarity matrix
#
#
# def run_analyses_all_sessions(saveFigs = True):
#     df = load_session_db()
#     df = df[df['RewardCount']>10]
#     df = df[df['Imaging']==1]
#
#     serverDir = "G:\My Drive\Figures\TwoTower"
#     for i in range(df.shape[0]):
#         try:
#             sess = df.iloc[i]
#             #try:
#             #    print(os.path.join(serverDir,sess['MouseName']))
#             #    os.makedirs(os.path.join(serverDir,sess['MouseName']))
#             #except:
#             #    print("make dirs failed")
#             #    pass
#             fbase = os.path.join(serverDir,sess['MouseName'])
#             filestr = "%s_%s_%d" % (sess['DateFolder'] , sess["Track"], sess["SessionNumber"])
#             print(filestr)
#             data_TO = run_behavior(sess,save=False,fbase=fbase,filestr=filestr)
#
#             # load calcium data
#             info = loadmat_sbx(sess['scanmat'])['info']
#             ca_dat = load_ca_mat(sess['scanfile'])
#
#             C = ca_dat['C_dec'][info['frame'][0]:info['frame'][-1]+1]
#             S = ca_dat['S_dec'][info['frame'][0]:info['frame'][-1]+1]
#
#             frame_diff = data_TO.shape[0]-C.shape[0]
#             if frame_diff>0:
#                 data_TO = data_TO.iloc[:-frame_diff]
#
#             C_z = sp.stats.zscore(C,axis=0)
#             S_z = sp.stats.zscore(S,axis=0)
#             S_z_smooth = gaussian_filter1d(S_z,3,axis=0)
#             S_smooth = gaussian_filter1d(S,3,axis=0)
#
#             # C
#             pcs = run_PCA(C_z,data_TO,save=True,fbase=fbase,filestr=filestr+"_C_z")
#             #s
#             #pcs_s = run_PCA(S_z_smooth,data_TO,save=True,fbase=fbase,filestr=filestr+"_S_smooth")
#
#
#             #run_placecells(C,data_TO,save=True,fbase=fbase,filestr=filestr)
#
#             # C
#             simmats = run_simmat(C_z,data_TO,save=True,fbase=fbase,filestr=filestr+"_C_z")
#             #s
#             #simmats_s= run_simmat(S_smooth,data_TO,save=True,fbase=fbase,filestr=filestr+"_S_smooth")
#         except:
#             print(i)
#     return
#
#
#
# def run_behavior(sess,save=False,fbase = None, filestr = None, ratio = False):
#     '''make behavior plots'''
#     data_TO = behavior_dataframe(sess['data file'],sess['scanmat'],concat=False)
#     trial_mat, bin_edges, bin_centers = make_pos_bin_trial_matrices(data_TO[['speed','morph','lick rate','reward','lick']]._values,
#                                               data_TO['pos']._values,
#                                               data_TO['tstart']._values,
#                                               data_TO['teleport']._values,bin_size=5)
#
#     morph_vec,count = sp.stats.mode(trial_mat[:,:,1],axis=1,nan_policy='omit')
#     morph_vec = np.squeeze(morph_vec)
#
#      # speed vs position
#     speed_dict = trial_type_dict(trial_mat[:,:,0],morph_vec)
#
#     f_speed,ax = plot_speed(bin_centers,speed_dict,morph_vec)
#
#
#
#     # plot licking behavior
#     lick_dict = trial_type_dict(trial_mat[:,:,4],morph_vec)
#     lick_mat = np.squeeze(trial_mat[:,:,2])
#     lick_mat_norm = lick_dict['all']/np.amax(lick_dict['all'])
#     lick_norm_dict = trial_type_dict(lick_mat_norm,morph_vec)
#     f_licks,axes = lick_plot(lick_dict,bin_edges,smooth=False,ratio=ratio)
#
#
#     trial_info = by_trial_info(data_TO)
#     pcnt_mean = avg_by_morph(trial_info['morphs'],trial_info['pcnt'])
#     f_pcntcorr,ax = plt.subplots(figsize=[5,5])
#     # morph_vals = np.arange(0,1.25,.25)
#     ax.plot(np.sort(np.unique(morph_vec)),pcnt_mean,color='black')
#     #ax.plot(morph_vals,pcnt_mean_post,color='red')
#     ax.set_ylabel("P(licked at second tower)")
#     ax.set_xlabel("morph")
#     ax.set_ylim([0,1])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     # zone speeds
#     zone0_speed = avg_by_morph(trial_info['morphs'],trial_info['zone0_speed'])
#     zone1_speed = avg_by_morph(trial_info['morphs'],trial_info['zone1_speed'])
#     f_rzonespeed,ax = plt.subplots(1,2,figsize=[10,5])
#     #morph_vals = np.arange(0,1.25,.25)
#     ax[0].plot(np.sort(np.unique(morph_vec)),zone0_speed,color='black')
#     ax[0].scatter(trial_info['morphs'],trial_info['zone0_speed'],color='black',s=5)
#     ax[0].set_ylabel("cm/s")
#     ax[0].set_xlabel("morph")
#     ax[0].set_title("zone 0 speed")
#     ax[0].spines['top'].set_visible(False)
#     ax[0].spines['right'].set_visible(False)
#
#
#     ax[1].plot(np.sort(np.unique(morph_vec)),zone1_speed,color='black')
#     ax[1].scatter(trial_info['morphs'],trial_info['zone1_speed'],color='black',s=5)
#     ax[1].set_ylabel("cm/s")
#     ax[1].set_xlabel("morph")
#     ax[1].set_title("zone 1 speed")
#     ax[1].spines['top'].set_visible(False)
#     ax[1].spines['right'].set_visible(False)
#
#     #position of first lick
#     pos_lick = avg_by_morph(trial_info['morphs'],trial_info['pos_lick'])
#     f_firstlick,ax = plt.subplots(figsize=[10,5])
#     ax.plot(np.sort(np.unique(morph_vec)),pos_lick,color='black')
#     ax.scatter(trial_info['morphs'],trial_info['pos_lick'],color='black',s=5)
#     ax.set_ylabel("cm/s")
#     ax.set_xlabel("morph")
#     ax.set_title("position of first lick")
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     if save:
#         try:
#             os.makedirs(os.path.join(fbase,'firstLicks'))
#             os.makedirs(os.path.join(fbase,'speed'))
#             os.makedirs(os.path.join(fbase,'pcntcorr'))
#             os.makedirs(os.path.join(fbase,'rzonespeed'))
#         except:
#             pass
#
#
#         f_firstlick.savefig(os.path.join(fbase,'firstLicks',filestr+".pdf"),format='pdf')
#         f_firstlick.savefig(os.path.join(fbase,'firstLicks',filestr+".svg"),format='svg')
#
#         f_speed.savefig(os.path.join(fbase,'speed',filestr+".pdf"),format='pdf')
#         f_speed.savefig(os.path.join(fbase,'speed',filestr+".svg"),format='svg')
#
#         f_pcntcorr.savefig(os.path.join(fbase,'pcntcorr',filestr+".pdf"),format='pdf')
#         f_pcntcorr.savefig(os.path.join(fbase,'pcntcorr',filestr+".svg"),format='svg')
#
#         f_rzonespeed.savefig(os.path.join(fbase,'rzonespeed',filestr+".pdf"),format='pdf')
#         f_rzonespeed.savefig(os.path.join(fbase,'rzonespeed',filestr+".svg"),format='svg')
#
#
#
#     return data_TO
#
#
# def run_PCA(C,data_TO,save=False,fbase = None,filestr=None):
#     pca = PCA()
#     trialMask = (data_TO['pos']>0) & (data_TO['pos']<445)
#     X = pca.fit_transform(C)
#
#     print(X.shape)
#     # skree plots
#     f,axarr = plt.subplots(5,2,figsize=[15,15])
#     axarr[0,0].plot(pca.explained_variance_)
#     axarr[0,0].set_ylabel("% variance",size=28)
#     axarr[0,0].tick_params(labelsize=20)
#
#     axarr[0,1].plot(np.log(pca.singular_values_))
#     axarr[0,1].set_ylabel("log(eigenvalue)",size=28)
#     axarr[0,1].tick_params(labelsize=20)
#
#     axarr[1,0].plot(pca.explained_variance_[:10])
#     axarr[1,0].set_ylabel("% variance",size=28)
#     axarr[1,0].tick_params(labelsize=20)
#
#
#     axarr[1,1].plot(np.log(pca.singular_values_[:10]))
#     axarr[1,1].set_ylabel("log(eigenvalue)",size=28)
#     axarr[1,1].tick_params(labelsize=20)
#
#
#
#
#     s_cxt=axarr[2,0].scatter(X[trialMask[:X.shape[0]],0],X[trialMask[:X.shape[0]],1],
#                                c=data_TO.loc[trialMask,'morph']._values,cmap='cool',s=2)
#     plt.colorbar(s_cxt,ax=axarr[2,0])
#     axarr[2,0].set_title('Colored by Context')
#     axarr[2,0].set_xlabel("PC 1")
#     axarr[2,0].set_ylabel("PC 2")
#
#
#     s_pos=axarr[2,1].scatter(X[trialMask,0],X[trialMask,1],c=data_TO.loc[trialMask,'pos'],cmap='magma',s=2)
#     plt.colorbar(s_pos,ax=axarr[2,1])
#     axarr[2,1].set_title('Colored by Position')
#     axarr[2,1].set_xlabel("PC 1")
#     axarr[2,1].set_ylabel("PC 2")
#
#
#     s_cxt=axarr[3,0].scatter(X[trialMask,1],X[trialMask,2],c=data_TO.loc[trialMask,'morph'],cmap='cool',s=2)
#     axarr[3,0].set_xlabel("PC 2")
#     axarr[3,0].set_ylabel("PC 3")
#
#     s_cxt=axarr[3,1].scatter(X[trialMask,1],X[trialMask,2],c=data_TO.loc[trialMask,'pos'],cmap='magma',s=2)
#     axarr[3,1].set_xlabel("PC 2")
#     axarr[3,1].set_ylabel("PC 3")
#
#
#     s_cxt=axarr[4,0].scatter(X[trialMask,2],X[trialMask,3],c=data_TO.loc[trialMask,'morph'],cmap='cool',s=2)
#     axarr[4,0].set_xlabel("PC 3")
#     axarr[4,0].set_ylabel("PC 4")
#
#     s_cxt=axarr[4,1].scatter(X[trialMask,2],X[trialMask,3],c=data_TO.loc[trialMask,'pos'],cmap='magma',s=2)
#     axarr[4,1].set_xlabel("PC 3")
#     axarr[4,1].set_ylabel("PC 4")
#
#
#     return f,X
#
# def run_placecells(C,data_TO,save=False,fbase = None,filestr=None):
#     morphs = np.sort(np.unique(data_TO['morph']._values))
#     f,ax = plt.subplots(2,int(morphs.shape[0]),figsize=[5*int(morphs.shape[0]),15])
#
#
#     mask0 = data_TO['morph']==0
#     frmap0, occupancy0 = rate_map(C[mask0,:],data_TO.loc[mask0,'pos'])
#     si0 = spatial_info(frmap0,occupancy0)
#     p0 = spatial_info_perm_test(si0,C[mask0,:],data_TO.loc[mask0,'pos'],nperms = 100)
#     maxInds0 = np.argmax(frmap0,axis=0)
#     p0_mask = np.argsort(maxInds0)
#     p0_mask = p0_mask[p0>.95]
#
#
#     mask1 = data_TO['morph']==1
#     frmap1, occupancy1 = rate_map(C[mask1,:],data_TO.loc[mask1,'pos'])
#     si1 = spatial_info(frmap1,occupancy1)
#     p1 = spatial_info_perm_test(si1,C[mask1,:],data_TO.loc[mask1,'pos'],nperms = 100)
#     maxInds1 = np.argmax(frmap1,axis=0)
#     p1_mask = np.argsort(maxInds1)
#     p1_mask = p1_mask[p1>.95]
#
#     fr_dict ={}
#     for i, m in enumerate(morphs):
#         mask = data_TO['morph']==m
#         frmap, occ = rate_map(C[mask,:],data_TO.loc[mask,'pos'])
#         frmap_norm = np.copy(frmap)
#         for j in range(frmap.shape[1]):
#             frmap_norm[:,j]= gaussian_filter1d(frmap[:,j],2)/frmap[:,j].sum()
#         fr_dict[m] = frmap/np.linalg.norm(frmap,ord='fro')
#
#         fr_0 = frmap_norm[:,p0_mask]
#         fr_0_norm = np.copy(fr_0)
#
#
#         ax[0,i].imshow(fr_0_norm.T,aspect='auto',cmap='Greys')#
#
#
#         fr_1 = frmap_norm[:,p1_mask]
#         fr_1_norm = np.copy(fr_1)
#
#         ax[1,i].imshow(fr_1_norm.T,aspect='auto',cmap='Greys')
#
#
#     if save:
#         try:
#             os.makedirs(os.path.join(fbase,'placecells'))
#         except:
#             pass
#
#         f.savefig(os.path.join(fbase,'placecells',filestr+".pdf"),format='pdf')
#         f.savefig(os.path.join(fbase,'placecells',filestr+".svg"),format='svg')
#
#
#     return f, ax, mask0, mask1
#
#
# def run_simmat(C,data_TO,save=False,fbase = None,filestr=None):
#     morphs = np.sort(np.unique(data_TO['morph']._values))
#     j = 0
#     for i,m in enumerate(morphs):
#         mask = data_TO['morph']==m
#         frmap, occ = rate_map(C[mask,:],data_TO.loc[mask,'pos'])
#         frmap_norm = np.zeros(frmap.shape)
#         print(frmap_norm.shape)
#         for j in range(frmap.shape[1]):
#             frmap_norm[:,j] = frmap[:,j]/frmap[:,j].sum()
#         if i == 0:
#             X = frmap.T
#             X_norm = frmap_norm.T
#         else:
#             X = np.hstack((X,frmap.T))
#             X_norm = np.hstack((X_norm,frmap_norm.T))
#
#
#
#     # calculate rate map
#     f,ax = plt.subplots(1,1, figsize=[20,20])
#     # add to array
#
#     simmat = np.matmul(X.T,X)
#     ax.imshow(simmat,aspect='auto',cmap='magma')
#     ax.vlines(np.arange(frmap.shape[0]-1,frmap.shape[0]*(i+1)-1,frmap.shape[0]),0,frmap.shape[0]*(i+1),color='blue')
#     ax.hlines(np.arange(frmap.shape[0]-1,frmap.shape[0]*(i+1)-1,frmap.shape[0]),0,frmap.shape[0]*(i+1),color='blue')
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     #ax.set_xlim([0,frmap.shape[0]*i])
#     #ax.set_ylim([0,frmap.shape[0]*i])
#
#     edges = np.arange(0,frmap.shape[0]*morphs.shape[0]+1,frmap.shape[0])
#     cm_sim = np.zeros([morphs.shape[0],morphs.shape[0]])
#     cm_sim_norm = np.zeros([morphs.shape[0],morphs.shape[0]])
#     for k,(start,stop) in enumerate(zip(edges[:-1],edges[1:])):
#         for l, (sstart,sstop) in enumerate(zip(edges[:-1],edges[1:])):
#             tmp = np.nanmean(simmat[start:stop,sstart:sstop])
#             cm_sim[k,l] = tmp
#             cm_sim_norm[k,l] = tmp
#
#     for z in range(cm_sim.shape[0]):
#         cm_sim_norm[z,:]/=cm_sim_norm[z,z]
#     f_cm, ax_cm = plt.subplots(figsize=[5,5])
#     ax_cm.imshow(cm_sim,aspect='auto',cmap='magma')
#     ax_cm.spines['top'].set_visible(False)
#     ax_cm.spines['right'].set_visible(False)
#
#     f_cm_n, ax_cm_n = plt.subplots(figsize=[5,5])
#     ax_cm_n.imshow(cm_sim_norm,aspect='auto',cmap='magma')
#     ax_cm_n.spines['top'].set_visible(False)
#     ax_cm_n.spines['right'].set_visible(False)
#
#
#
#
#     if save:
#         try:
#             os.makedirs(os.path.join(fbase,'simmat'))
#         except:
#             pass
#
#         f.savefig(os.path.join(fbase,'simmat',filestr+".png"),format='png')
#         f.savefig(os.path.join(fbase,'simmat',filestr+".svg"),format='svg')
#         f_cm.savefig(os.path.join(fbase,'simmat',filestr+"_cm.png"),format='png')
#         f_cm.savefig(os.path.join(fbase,'simmat',filestr+"_cm.svg"),format='svg')
#         f_cm_n.savefig(os.path.join(fbase,'simmat',filestr+"_cm_n.png"),format='png')
#         f_cm_n.savefig(os.path.join(fbase,'simmat',filestr+"_cm_n.svg"),format='svg')
#
#     return simmat, cm_sim, cm_sim_norm
