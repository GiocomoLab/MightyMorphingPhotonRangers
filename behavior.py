import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter1d
import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
from astropy.convolution import convolve, Gaussian1DKernel
import utilities as u



def learning_curve_plots(data,reversal=False):

    if isinstance(data,list):
        N = len(data)
    else:
        N = 1
        data = [data]


    f_pcntcorr,ax_pcntcorr = plt.subplots(figsize=[5,5])
    f_lp,ax_lp = plt.subplots(figsize=[5,5])
    f_sess, ax_sess = plt.subplots(figsize=[5,5])
    pcnt0, pcnt1 = [], []

    #sess_avg_pcnt = np.zeros([u_morphs.shape])
    for i,d in enumerate(data):

        # plot licking behavior
        trial_info, tstart, tend = u.by_trial_info(d)
        if reversal:
            trial_info['pcnt']=1-trial_info['pcnt']
        pcnt_mean = u.avg_by_morph(trial_info['morphs'],trial_info['pcnt'])
        u_morphs = np.sort(np.unique(trial_info['morphs']))

        pcnt0.append(pcnt_mean[0])
        pcnt1.append(pcnt_mean[-1])
        ax_sess.scatter(i*np.ones(pcnt_mean.shape),pcnt_mean,c=np.sort(np.unique(trial_info['morphs'])),cmap='cool')

       # morph_vals = np.arange(0,1.25,.25)
        ax_pcntcorr.plot(np.sort(np.unique(trial_info['morphs'])),pcnt_mean,color=plt.cm.copper(i/float(N)))
        #ax.plot(morph_vals,pcnt_mean_post,color='red')
        ax_pcntcorr.set_ylabel("P(licked at second tower)")
        ax_pcntcorr.set_xlabel("morph")
        ax_pcntcorr.set_ylim([0,1])
        ax_pcntcorr.spines['top'].set_visible(False)
        ax_pcntcorr.spines['right'].set_visible(False)
        #ax_pcntcorr.set_title(mouse)



        licknans = np.isnan(trial_info['pos_lick']) & (trial_info['max_pos']<450) & (trial_info['max_pos']>245)
        trial_info['pos_lick'][licknans]=trial_info['max_pos'][licknans]
         #position of first lick
        pos_lick = u.avg_by_morph(trial_info['morphs'],trial_info['pos_lick'])

        ax_lp.plot(np.sort(np.unique(trial_info['morphs'])),pos_lick,color=plt.cm.copper(i/float(N)))
        #ax_lp.scatter(trial_info['morphs'],trial_info['pos_lick'],color=plt.cm.copper(i/float(N)),s=5)
        ax_lp.set_ylabel("cm")
        ax_lp.set_xlabel("morph")
        ax_lp.set_title("position of first lick")
        ax_lp.spines['top'].set_visible(False)
        ax_lp.spines['right'].set_visible(False)

    ax_sess.plot(np.arange(i+1),pcnt0,color=plt.cm.cool(0.))
    ax_sess.plot(np.arange(i+1),pcnt1,color=plt.cm.cool(1.))
    ax_sess.set_xlabel('Session Number')
    ax_sess.set_ylabel('P(lick at second tower)')
    #ax_sess.set_title(mouse)
    ax_sess.spines['top'].set_visible(False)
    ax_sess.spines['right'].set_visible(False)
    ax_sess.set_ylim([0,1])

    return (f_sess,ax_sess), (f_pcntcorr, ax_pcntcorr), (f_lp, ax_lp)

def lick_plot_task(d,bin_edges,rzone0=(250.,315),rzone1=(350,415),smooth=True,ratio = True, max_pos=None):
    '''standard plot for licking behavior'''
    f = plt.figure(figsize=[15,15])

    gs = gridspec.GridSpec(5,5)


    ax = f.add_subplot(gs[0:-1,0:-1])
    ax.axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(0)),zorder=0)
    ax.axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(1)),zorder=0)
    ax = u.smooth_raster(bin_edges[:-1],d['all'],vals=d['labels'],ax=ax,smooth=smooth,tports=max_pos)
    ax.set_ylabel('Trial',size='xx-large')


    meanlr_ax = f.add_subplot(gs[-1,:-1])
    meanlr_ax.axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(0)),zorder=0)
    meanlr_ax.axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(1)),zorder=0)
    for i, m in enumerate(np.unique(d['labels'])):
        meanlr_ax.plot(bin_edges[:-1],np.nanmean(d[m],axis=0),color=plt.cm.cool(np.float(m)))
    meanlr_ax.set_ylabel('Licks/sec',size='xx-large')
    meanlr_ax.set_xlabel('Position (cm)',size='xx-large')


    if ratio:
        lickrat_ax = f.add_subplot(gs[:-1,-1])
        bin_edges = np.array(bin_edges)
        rzone0_inds = np.where((bin_edges[:-1]>=rzone0[0]) & (bin_edges[:-1] <= rzone0[1]))[0]
        rzone1_inds = np.where((bin_edges[:-1]>=rzone1[0]) & (bin_edges[:-1] <= rzone1[1]))[0]
        rzone_lick_ratio = {}
        for i,m in enumerate(np.unique(d['labels'])):
            zone0_lick_rate = d[m][:,rzone0_inds].mean(axis=1)
            zone1_lick_rate = d[m][:,rzone1_inds].mean(axis=1)
            rzone_lick_ratio[m] = np.divide(zone0_lick_rate,zone0_lick_rate+zone1_lick_rate)
            rzone_lick_ratio[m][np.isinf(rzone_lick_ratio[m])]=np.nan

        for i,m in enumerate(np.unique(d['labels'])):

            trial_index = d['labels'].shape[0] - d['indices'][m]
            lickrat_ax.scatter(rzone_lick_ratio[m],trial_index,
                               c=plt.cm.cool(np.float(m)),s=10)
            k = Gaussian1DKernel(5)
            lickrat_ax.plot(convolve(rzone_lick_ratio[m],k,boundary='extend'),trial_index,c=plt.cm.cool(np.float(m)))
        lickrat_ax.set_yticklabels([])
        lickrat_ax.set_xlabel(r'$\frac{zone_0}{zone_0 + zone_1}  $',size='xx-large')


        for axis in [ax, meanlr_ax, lickrat_ax]:
            for edge in ['top','right']:
                axis.spines[edge].set_visible(False)

        return f, (ax, meanlr_ax, lickrat_ax)
    else:
        for axis in [ax, meanlr_ax]:
            for edge in ['top','right']:
                axis.spines[edge].set_visible(False)

        return f, (ax, meanlr_ax)

def plot_speed_task(x,d,vals,ax=None,f=None,rzone0=(250,315),rzone1=(350,415)):
    '''plot individual trial and average speed as a function of position along the Track
    x = position, d=dictionary output of by_trial_dict'''
    if ax is None:
        f, ax = plt.subplots(1,2,figsize=[10,5])
    for i,m in enumerate(np.unique(vals)):
        for j in range(d[m].shape[0]):
            tmp = ax[0].plot(x,d[m][j,:],color = plt.cm.cool(np.float(m)),alpha=.1)
        tmp = ax[0].plot(x,np.nanmean(d[m],axis=0),color=plt.cm.cool(np.float(m)),zorder=1)
        tmp = ax[1].plot(x,np.nanmean(d[m],axis=0),color=plt.cm.cool(np.float(m)))

    ax[0].axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(0)),zorder=0)
    ax[0].axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(1)),zorder=0)
    ax[1].axvspan(rzone0[0],rzone0[1],alpha=.2,color=plt.cm.cool(np.float(0)),zorder=0)
    ax[1].axvspan(rzone1[0],rzone1[1],alpha=.2,color=plt.cm.cool(np.float(1)),zorder=0)
    for edge in ['top','right']:
        ax[0].spines[edge].set_visible(False)
        ax[1].spines[edge].set_visible(False)

    ax[0].set_xlabel('Position')
    ax[0].set_ylabel('Speed cm/s')
    ax[0].set_ylim([0, 60])
    ax[1].set_ylim([0, 60])
    return f,ax


def behavior_raster_foraging(lick_mat,centers,morphs,reward_pos,smooth=True, max_pos=None):
    '''plot licking behavior when the animal is doing the 'foraging' task '''

    f = plt.figure(figsize=[15,15])
    gs = gridspec.GridSpec(4,6)
    axarr = []

    # lick x pos
    #       colored by morph
    ax = f.add_subplot(gs[:,:2])
    ax = u.smooth_raster(centers,lick_mat,vals=morphs,ax=ax,smooth=smooth,cmap='cool')
    ax.set_ylabel('Trial',size='xx-large')
    axarr.append(ax)


    # sort trials by morph
    ax = f.add_subplot(gs[:,2:4])
    msort = np.argsort(morphs)
    ax = u.smooth_raster(centers,lick_mat[msort,:],vals=morphs[msort],ax=ax,smooth=smooth,cmap='cool')
    axarr.append(ax)


    #       colored by position of reward

    ax = f.add_subplot(gs[:,4:])
    rsort = np.argsort(reward_pos)
    ax = u.smooth_raster(centers,lick_mat[rsort,:],vals=reward_pos[rsort],ax=ax,smooth=smooth,cmap='viridis')
    ax.axvline(200,ymin=0,ymax=lick_mat.shape[0]+1)
    axarr.append(ax)

    for i,a in enumerate(axarr):
        for edge in ['top','right']:
            a.spines[edge].set_visible(False)
        if i>0:
            # a.spines['left'].set_visible(False)
            a.set_yticklabels([])


    # lick x pos centered on entry to reward zone
    #       colored by morph

    #       colored by positon of reward

    return f, axarr




# def plot_speed_foraging(speedmat,centers,morphs,reward_pos):
#     '''plot individual trial and average speed as a function of position along the Track
#     x = position, d=dictionary output of by_trial_dict'''
#
#     f, ax = plt.subplots(2,2,figsize=[10,5])
#     for i,m in enumerate(np.unique(vals)):
#         for j in range(d[m].shape[0]):
#             tmp = ax[0].plot(x,d[m][j,:],color = plt.cm.cool(np.float(m)),alpha=.1)
#         tmp = ax[0].plot(x,np.nanmean(d[m],axis=0),color=plt.cm.cool(np.float(m)),zorder=1)
#         tmp = ax[1].plot(x,np.nanmean(d[m],axis=0),color=plt.cm.cool(np.float(m)))
#
#
#     for edge in ['top','right']:
#         ax[0].spines[edge].set_visible(False)
#         ax[1].spines[edge].set_visible(False)
#
#     ax[0].set_xlabel('Position')
#     ax[0].set_ylabel('Speed cm/s')
#     ax[0].set_ylim([0, 60])
#     ax[1].set_ylim([0, 60])
#     return f,ax
