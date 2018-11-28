# insert code for doing MCN decoding analysis
import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import utilities as u
import preprocessing as pp
import numpy as np
import scipy as sp
import sklearn as sk
import sklearn.linear_model
from sklearn.neighbors.kde import KernelDensity
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter, gaussian_filter1d



class empirical_density:
    '''calculate empirical joint density'''
    def __init__(self,x,y,xbins = np.arange(0,451,5),ybins=np.arange(-2,15,.5),sigma=(2,2)):
        ybins = np.append(ybins,25)
        H,xedge,yedge = np.histogram2d(x,y,bins=[xbins,ybins])
        H_smooth = gaussian_filter(H,sigma,mode='nearest')

        H_smooth = np.divide(H_smooth,np.dot(H_smooth.sum(axis=1)[np.newaxis].T,np.ones([1,H_smooth.shape[1]])))
        # make sure likelihood isn't precisely 0 anywhere
        H_smooth = np.minimum(H_smooth+.0001,.9999)
        H_smooth = np.divide(H_smooth,np.dot(H_smooth.sum(axis=1)[np.newaxis].T,np.ones([1,H_smooth.shape[1]])))

        self.xbins=xbins
        self.ybins=ybins
        self.sigma = sigma
        self.H_smooth= H_smooth

    def Likelihood(self,xi,yi):
        x_inds = np.digitize(xi,self.xbins[:-1],right=False)-1
        x_inds = np.maximum(0,np.minimum(x_inds,self.H_smooth.shape[0]))
        y_inds = np.digitize(yi,self.ybins[:-1],right=False)-1
        y_inds = np.maximum(0,np.minimum(y_inds,self.H_smooth.shape[1]))
        return self.H_smooth[x_inds,y_inds]

def transition_prob_matrix(x,binsize=5):
    '''calculate transition proabibilities'''
    # bin positions in 10 cm
    bin_edges = [0]
    for i in range(binsize,465,binsize):
        bin_edges.append(i)

    x_binned = np.digitize(x,bin_edges,right=True)

    # #transition matrix
    XX = np.zeros([len(bin_edges),len(bin_edges)])

    for b in np.unique(x_binned).tolist():
        inds = np.where(x_binned==b)[0]
        xx = x_binned[(inds+1)%x_binned.shape[0]]
        next_inds = np.unique(xx)
        bcount = np.bincount(xx)
        bcount = bcount[bcount>0]

        XX[next_inds,b] = bcount/bcount.sum()

    XX = np.minimum(XX+.0001,.9999)
    XX_smooth = gaussian_filter1d(XX,4,axis=0)
    for b in np.unique(x_binned).tolist():
        XX_smooth[:,b]/=XX_smooth[:,b].sum()

    return XX, bin_edges

def empirical_decoding_model(L,T,starts, stops, prefix = "E:\\"):

    # allocate for single cell data
    post_ix= np.memmap(os.path.join(prefix,"post_ix.dat"),dtype='float32',mode='r+',shape=tuple(L.shape))#np.zeros(L.shape)
    post_i = np.memmap(os.path.join(prefix,"post_i.dat"),dtype='float32', mode='r+',shape=(L.shape[0],L.shape[2])) #np.zeros([L.shape[0],L.shape[2]])

    # allocate for population data
    pop_post_ix = np.zeros(L.shape[0:2])
    pop_post_i = np.zeros([L.shape[0],])

    # number of spatial bins
    NX = int(T.shape[0]/2)

    for trial,(start,stop)  in enumerate(zip(starts.tolist(),stops.tolist())):
        #print(a.shape)
        if trial%5==0:
            print("processing trial %d" % trial)

        for j in range(stop-start+1):
            if j==0: # if first timepoint, set initial conditions

                #### single cell initial conditions
                # set probability of being in current position to 1
                B = np.ones([T.shape[0],1])
                #B[0,:] = .5
                B/= B.ravel().sum()
                B = np.matlib.repmat(B,1,L.shape[2])

                BB = np.ones([T.shape[0],1])
                #BB[0,:] = .5
                BB/= BB.ravel().sum()


            ######## single cell decoding
            # new posterior - unnormalized
            B = np.multiply(np.dot(T,B),np.squeeze(L[start+j,:,:]))
            # denominator for normaliztion
            d = np.matlib.repmat(B.sum(axis=0),T.shape[0],1)
            B = np.divide(B,d)

            post_ix[start+j,:,:] = B
            post_i[start+j,:] = B[NX:,:].sum(axis=0)
            # ensure no underflow
            B=np.minimum(.0001+B, .9999)
            d = np.matlib.repmat(B.sum(axis=0),T.shape[0],1)
            B = np.divide(B,d)

            # ######## population decoding
            logBB = np.sum(np.log(np.squeeze(L[start+j,:,:])),axis=1)[np.newaxis].T + np.log(np.dot(T,BB))
            # prevent underflow
            numInf = np.isinf(logBB).sum()
            if numInf>0:
                print("number of inf inds: %f" % np.isinf(logBB).sum())
                logBB[np.isinf(logBB)] = -1E10


            logBB -= logBB.max() -1

            BB = np.exp(logBB)
            BB/=BB.ravel().sum()

            pop_post_ix[start+j,:] = BB.ravel()
            pop_post_i[start+j]= BB[NX:].sum()

            # ensure there is no underflow
            BB=np.minimum(BB+.0001,.9999)
            BB/=BB.ravel().sum()


    return {'cell ix':post_ix,
            'cell i':post_i,
            'pop ix': pop_post_ix,
            'pop i': pop_post_i}



#def single_session(sess,save=False,dirbase=None,trainOnCorrect=True,prefix = "E:\\"):
class single_session:
    def __init__(self,sess,save=False,trainOnCorrect=True,prefix = "E:\\"):
        VRDat, C,Cd,S, A = pp.load_scan_sess(sess)
        C_z = sp.stats.zscore(C,axis=0)
        trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)

        # get rewarded trials
        pcnt = u.correct_trial_mask(trial_info['rewards'],tstart_inds,teleport_inds,VRDat.shape[0])

        # get lick positions
        lick_positions = u.lick_positions(VRDat['lick']._values,VRDat['pos']._values)


        if trainOnCorrect:
            inds0 = (VRDat['morph']==0) & (VRDat['pos']>0)  & (pcnt>0)
            inds1 = (VRDat['morph']==1) & (VRDat['pos']>0)  & (pcnt>0)
        else:
            inds0 = (VRDat['morph']==0) & (VRDat['pos']>0)
            inds1 = (VRDat['morph']==1) & (VRDat['pos']>0)

        morphInds = {}
        for m in [0,.25,.5,.75,1]:
            morphInds[m] = (VRDat['morph']==m) & (VRDat['pos']>0)

        XX,xbins = self.state_transition_matrix(VRDat['pos']._values,inds0,inds1)
        pos_binned = np.digitize(VRDat['pos']._values,xbins,right=True)


        self.C_z = C_z
        self.pos = VRDat['pos']._values
        self.XX = XX
        self.tstarts = tstart_inds
        self.teleports = teleport_inds
        self.lick_pos = lick_positions
        self.pos_binned = pos_binned
        self.trial_info = trial_info
        self.inds0 = inds0
        self.inds1 = inds1
        self.pcnt = pcnt
        self.morphInds = morphInds
        self.prefix = prefix
        self.xbins = xbins


    def state_transition_matrix(self,pos,inds0,inds1):

        XX_I0, xbins = transition_prob_matrix(pos[inds0],binsize=5)
        XX_I1, xbins = transition_prob_matrix(pos[inds1],binsize=5)

        XX = np.zeros([2*XX_I0.shape[0],2*XX_I0.shape[1]])
        XX[:XX_I0.shape[0],:XX_I0.shape[1]]=XX_I0
        XX[XX_I0.shape[0]:,XX_I0.shape[1]:]=XX_I1

        xbins = np.array(xbins)
        return XX, xbins

    def likelihood_maps(self):
        # get likelihoods at all timepoints - functionalize
        try:
            L = np.memmap(os.path.join(self.prefix,"L.dat"),dtype='float32',mode='r+',shape = (self.C_z.shape[0],self.xbins.shape[0]*2,self.C_z.shape[1]))
        except:
            L = np.memmap(os.path.join(self.prefix,"L.dat"),dtype='float32',mode='w+',shape = (self.C_z.shape[0],self.xbins.shape[0]*2,self.C_z.shape[1]))
        for c in range(0,self.C_z.shape[1]):
            pdf0 = empirical_density(self.pos[self.inds0],self.C_z[self.inds0,c])
            pdf1 = empirical_density(self.pos[self.inds1],self.C_z[self.inds1,c])
            if c % 50 ==0:
                print(c)

            xx,CC = np.meshgrid(self.xbins,self.C_z[:,c])
            L[:,:self.xbins.shape[0],c]=np.reshape(pdf0.Likelihood(xx.ravel(),CC.ravel()),xx.shape)
            L[:,self.xbins.shape[0]:,c]=np.reshape(pdf1.Likelihood(xx.ravel(),CC.ravel()),xx.shape)
        return L

    def ctxt_likelihood_given_pos(self):
        C_z = self.C_z
        LL = np.zeros([C_z.shape[0],2,C_z.shape[1]],dtype='float32')
        for c in range(0,C_z.shape[1]):
            pdf0 = empirical_density(self.pos[self.inds0],C_z[self.inds0,c])
            pdf1 = empirical_density(self.pos[self.inds1],C_z[self.inds1,c])
            if c % 50 ==0:
                print(c)

            LL[:,0,c]=pdf0.Likelihood(self.pos,C_z[:,c])
            LL[:,1,c]=pdf1.Likelihood(self.pos,C_z[:,c])
        return LL

    def ctxt_LLR_given_pos(self):
        LL = np.log(self.ctxt_likelihood_given_pos())
        LLR = np.squeeze( LL[:,0,:]- LL[:,1,:])
        LLR = gaussian_filter1d(LLR,1,axis=0)
        LLR_pop = LLR.sum(axis=-1)
        return LLR,LLR_pop

    def ctxt_LLR(self,L=None):
        if L is None:
            L=self.likelihood_maps()

        nbins = self.xbins.shape[0]
        fr, px0 = u.rate_map(self.C_z[self.inds0],self.pos[self.inds0],bin_size=5,max_pos=465)
        fr, px1 = u.rate_map(self.C_z[self.inds1],self.pos[self.inds1],bin_size=5,max_pos=465)

        px0 = np.matlib.repmat(px0[np.newaxis],L.shape[0],1)
        px1 = np.matlib.repmat(px1[np.newaxis],L.shape[0],1)
        Z = np.zeros([L.shape[0],2,self.C_z.shape[1]])
        for c in range(self.C_z.shape[1]):
            l0 = np.squeeze(L[:,:nbins,c])
            l1 = np.squeeze(L[:,nbins:,c])

            Z[:,0,c] = np.multiply(l0,px0).sum(axis=1)
            Z[:,1,c] = np.multiply(l1,px1).sum(axis=1)
            if c%50 == 0:
                print(c)

        Z = np.log(Z)
        LLR = np.squeeze(Z[:,1,:]-Z[:,0,:])
        LLR_pop = LLR.sum(axis=-1)
        return LLR,LLR_pop

    def run_decoding(self, L=None):
        if L is None:
            L = self.likelihood_maps()

        return empirical_decoding_model(L,self.XX,self.tstarts,self.teleports)



    def plot_decoding(self,decode_dict,LLR,LLR_pop,rzone0=[250,315],rzone1=[350,415],save=False):
        rzone0 = [i/5 for i in rzone0]
        rzone1 = [i/5 for i in rzone1]
        if save:
            try:
                os.makedirs("%s\\decoding" % self.prefix)
            except:
                print("error making directory")

        for t,(start,stop) in enumerate(zip(self.tstarts.tolist(),self.teleports.tolist())):

            px = decode_dict['pop ix'][start:stop,:].T

            gs = gridspec.GridSpec(10,1)
            f = plt.figure(figsize=[10,10])
            ax = f.add_subplot(gs[0:6,:])
            #f, ax = plt.subplots(2,1,figsize= [15,5],sharex=True)
            ax.imshow(px,aspect = 'auto',cmap='magma',alpha=.4,zorder=2)
            ax.axhline(self.xbins.shape[0],xmin=0,xmax=px.shape[1],color='white',linewidth=5,zorder=10)

            ax.plot(self.pos_binned[start:stop], color = plt.cm.cool(0.),linewidth=2,zorder=0,alpha=.5)
            ax.plot(self.pos_binned[start:stop]+93, color = plt.cm.cool(1.),linewidth=2,zorder=0,alpha=.5)
            ax.fill_between(np.arange(px.shape[1]),rzone0[0],y2 = rzone0[1],color=plt.cm.cool(0.),alpha=.2)
            ax.fill_between(np.arange(px.shape[1]),rzone1[0],y2 = rzone1[1],color=plt.cm.cool(1.),alpha=.2)
            ax.fill_between(np.arange(px.shape[1]),rzone0[0]+93,y2 = rzone0[1]+93,color=plt.cm.cool(0.),alpha=.2)
            ax.fill_between(np.arange(px.shape[1]),rzone1[0]+93,y2 = rzone1[1]+93,color=plt.cm.cool(1.),alpha=.2)
            ax.set_xlim([0,px.shape[1]])

            x = np.arange(px.shape[1])
            ax.scatter(x,self.lick_pos[start:stop]/5,s=50,marker='x',color='blue',alpha=.5,zorder=1)
            ax.scatter(x,self.lick_pos[start:stop]/5+93,s=50,marker='x',color='blue',alpha=.5,zorder=1)
            ax.set_title("trial %d morph %f reward %f" % (t,self.trial_info['morphs'][t],self.trial_info['rewards'][t]))

            aax = f.add_subplot(gs[6,:],sharex=ax)
            aax.scatter(x,-LLR_pop[start:stop],c=plt.cm.cool(decode_dict['pop i'][start:stop]))
            aax.axhline(0,xmin=0,xmax=px.shape[1])
            aax.set_xlim([0,px.shape[1]])
            aax.set_ylim([-70,70])

            aaax= f.add_subplot(gs[7:,:],sharex=ax)
            aaax.imshow(LLR[start:stop,:].T,cmap = 'cool',aspect='auto',vmin=-2.5,vmax=2.5)
            if save:
                f.savefig("%s\\decoding\\trial%d_morph%2f_reward%d.pdf" % (prefix,t,self.trial_info['morphs'][t],int(self.trial_info['rewards'][t])),format='pdf')


    def plot_llr(self,LLR):

        f_pos,ax_pos = plt.subplots(1,5,figsize=[20,5])
        ff_pos,aax_pos = plt.subplots()
        f_time,ax_time = plt.subplots(1,5,figsize=[20,5])
        ff_time,aax_time = plt.subplots()
        for i,(start,stop,m,r)  in enumerate(zip(self.tstarts.tolist(),self.teleports.tolist(),self.trial_info['morphs'].tolist(),self.trial_info['rewards'].tolist())):

            self._single_line_llr_multiax(self.pos[start:stop],LLR[start:stop],m,r,ax_pos)
            ax_pos[-1].set_ylabel('position')

            self._single_line_llr_multiax(np.arange(stop-start)*1./15.46,LLR[start:stop],m,r,ax_time,xlim=[0,250])
            ax_time[-1].set_ylabel('time')

            aax_pos.plot(self.pos[start:stop],LLR[start:stop],color=plt.cm.cool(np.float(m)),alpha=.5)
            aax_time.plot(np.arange(stop-start)*1./15.46,LLR[start:stop],color=plt.cm.cool(np.float(m)),alpha=.5)
        return (f_pos,ax_pos), (ff_pos,aax_pos), (f_time,ax_time), (ff_time,aax_time)
        # edit axes




    def _single_line_llr_multiax(self,x,y,m,r,ax,lw=.5,ylim=[-100,100],xlim=[0,460]):
        if r>0:

            if m == 0:
                ax[0].plot(x,y,color=plt.cm.cool(m),linewidth=lw)
            elif m == .25:
                ax[1].plot(x,y,color=plt.cm.cool(m),linewidth=lw)
            elif m  == .5 :
                ax[2].plot(x,y,color=plt.cm.cool(m),linewidth=lw)
            elif m == .75:
                ax[3].plot(x,y,color=plt.cm.cool(m),linewidth=lw)
            elif m == 1.:
                ax[4].plot(x,y,color=plt.cm.cool(m),linewidth=lw)
        else:
            if m == 0:
                ax[0].plot(x,y,color='black',linewidth=lw,alpha=.6)
            elif m == .25:
                ax[1].plot(x,y,color='black',linewidth=lw,alpha=.6)
            elif m  == .5 :
                ax[2].plot(x,y,color='black',linewidth=lw,alpha=.6)
            elif m == .75:
                ax[3].plot(x,y,color='black',linewidth=lw,alpha=.6)
            elif m == 1.:
                ax[4].plot(x,y,color='black',linewidth=lw,alpha=.6)




# def llr(C_z,VRDat,glm,trial_info,save=False,dir=None):
#     # get likelihood for each point in time for log-likelihood ratio
#     tmp_mu_i0 = glm.predict(pos_morph_design_matrix(VRDat['pos'],np.zeros([VRDat['pos'].shape[0],])))
#     l_i0 = gaussian_pdf(C_z,tmp_mu_i0,1)
#
#     tmp_mu_i1 = glm.predict(pos_morph_design_matrix(VRDat['pos'],np.ones([VRDat['pos'].shape[0],])))
#     l_i1 = gaussian_pdf(C_z,tmp_mu_i1,1)
#     llr = np.log(l_i0).sum(axis=1) - np.log(l_i1).sum(axis=1)
#     print(llr.shape)
#
#     tstarts,tstops = np.where(VRDat['tstart']==1)[0],np.where(VRDat['teleport']==1)[0]
#     trial_llr, trial_pos = [],[]
#     for (start,stop) in zip(tstarts,tstops):
#         trial_llr.append(llr[start:stop])
#         trial_pos.append(VRDat['pos']._values[start:stop])
#
#     f,ax = plt.subplots(1,5,figsize=[20,5])
#     nbins = 100
#     xbins = np.linspace(0,460,nbins)
#     mcount = {0:np.zeros([nbins,]), .25:np.zeros([nbins,]), .5:np.zeros([nbins,]), .75:np.zeros([nbins,]), 1.:np.zeros([nbins,])}
#     avgTrace = {0:np.zeros([nbins,]), .25:np.zeros([nbins,]), .5:np.zeros([nbins,]), .75:np.zeros([nbins,]), 1.:np.zeros([nbins,])}
#     #for key in avgTrace.keys():
#     #    avgTrace[key][:]=np.nan
#
#     for i,(lr,tmppos,m,r)  in enumerate(zip(trial_llr,trial_pos,trial_info['morphs'],trial_info['rewards'])):
#         if r>0:
#             if m == 0:
#                 ax[0].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
#                 ax[0].set_ylim([-300,300])
#
#             elif m == .25:
#                 ax[1].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
#                 ax[1].set_ylim([-300,300])
#             elif m  == .5 :
#                 ax[2].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
#                 ax[2].set_ylim([-300,300])
#             elif m == .75:
#                 ax[3].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
#                 ax[3].set_ylim([-300,300])
#
#             elif m == 1.:
#                 ax[4].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
#                 ax[4].set_ylim([-300,300])
#         else:
#             if m == 0:
#                 ax[0].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)
#
#             elif m == .25:
#                 ax[1].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)
#
#             elif m  == .5 :
#                 ax[2].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)
#
#             elif m == .75:
#                 ax[3].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)
#
#
#             elif m == 1.:
#                 ax[4].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)
#
#
#         if i == 0:
#             for j in range(5):
#                 ax[j].axhline(0,xmin=0,xmax=460,color='black',linewidth=.5)
#
#
#         posbins = np.digitize(tmppos,xbins,right=True)
#         #bcounts = np.bincounts(pos,xbins)
#         emptyTrace = np.zeros([nbins,])
#      #   emptyTrace[:]=np.nan
#         for z in range(nbins):
#             avgTrace[m][z] += lr[posbins==z].sum()
#             mcount[m][z] += sum(posbins==z)
#
#
#     f_mu,ax_mu = plt.subplots()
#     for key in avgTrace.keys():
#         avgTrace[key] = np.divide(avgTrace[key],mcount[key])
#         #print(avgTrace[key])
#         ax_mu.plot(xbins,avgTrace[key],color=plt.cm.cool(np.float(key)))
#         ax_mu.axhline(0,xmin=0,xmax=460,color='black',linewidth=.5)
#
#     if save:
#         f.savefig("%s\\single_trial_llr.pdf" % dir,format='pdf')
#         f_mu.savefig("%s\\trial_avg_llr.pdf" % dir,format='pdf')


# def plot_decoding(decode_dict,trial_info,trial_pos_binned,trial_licks,save = False, dir = None,rzone0=[250,315],rzone1=[350,415]):
#
#     rzone0 = [i/5 for i in rzone0]
#     rzone1 = [i/5 for i in rzone1]
#     if save:
#         try:
#             os.makedirs("%s\\decoding" % dir)
#         except:
#             print("error making directory")
#
#     for t in range(trial_info['morphs'].shape[0]):
#         px = np.hstack((decode_dict['pop i0x_y'][t],decode_dict['pop i1x_y'][t])).T
#         #print(px.sum(axis=0))
#         gs = gridspec.GridSpec(5,1)
#         f = plt.figure(figsize=[10,10])
#         ax = f.add_subplot(gs[0:-1,:])
#         #f, ax = plt.subplots(2,1,figsize= [15,5],sharex=True)
#         ax.imshow(px,aspect = 'auto',cmap='magma',alpha=.4,zorder=2)
#
#         ax.axhline(93,xmin=0,xmax=px.shape[1],color='white',linewidth=5,zorder=10)
#
#         ax.plot(trial_pos_binned[t], color = plt.cm.cool(0.),linewidth=2,zorder=0,alpha=.5)
#         ax.plot(trial_pos_binned[t]+93, color = plt.cm.cool(1.),linewidth=2,zorder=0,alpha=.5)
#         ax.fill_between(np.arange(px.shape[1]),rzone0[0],y2 = rzone0[1],color=plt.cm.cool(0.),alpha=.2)
#         ax.fill_between(np.arange(px.shape[1]),rzone1[0],y2 = rzone1[1],color=plt.cm.cool(1.),alpha=.2)
#         ax.fill_between(np.arange(px.shape[1]),rzone0[0]+93,y2 = rzone0[1]+93,color=plt.cm.cool(0.),alpha=.2)
#         ax.fill_between(np.arange(px.shape[1]),rzone1[0]+93,y2 = rzone1[1]+93,color=plt.cm.cool(1.),alpha=.2)
#         ax.set_xlim([0,px.shape[1]])
#
#         x = np.arange(px.shape[1])
#         licks = trial_licks[t]
#
#         ax.scatter(x,licks/5,s=50,marker='x',color='blue',alpha=.5,zorder=1)
#         ax.scatter(x,licks/5+93,s=50,marker='x',color='blue',alpha=.5,zorder=1)
#         ax.set_title("trial %d morph %f reward %f" % (t,trial_info['morphs'][t],trial_info['rewards'][t]))
#
#         aax = f.add_subplot(gs[-1,:],sharex=ax)
#         aax.scatter(x,decode_dict['pop i0'][t],c=plt.cm.cool(decode_dict['pop i1'][t]))
#         aax.set_ylim([0,1])
#         aax.axhline(.5,xmin=0,xmax=px.shape[1])
#         aax.set_xlim([0,px.shape[1]])
#         if save:
#             f.savefig("%s\\decoding\\trial%d_morph%2f_reward%d.pdf" % (dir,t,trial_info['morphs'][t],int(trial_info['rewards'][t])),format='pdf')
#
# def make_spline_basis(x,knots=np.arange(0,450,50)):
#     '''make cubic spline basis functions'''
#     knotfunc = lambda k: np.power(np.multiply(x-k,(x-k)>0),3)
#     spline_basis_list = [knotfunc(k) for k in knots.tolist()]
#     spline_basis_list += [np.ones(x.shape[0]),x,np.power(x,2)]
#     return np.array(spline_basis_list).T
#
# def pos_morph_design_matrix(x,m,splines=True,knots=np.arange(-50,450,50),speed=None):
#     '''make design matrix for GLM that uses basis functions for position and separate regresssors for each context'''
#     if splines:
#         basis = make_spline_basis(x,knots=knots)
#     else:
#         # add functionality for radial basis functions
#         pass
#
#     M = np.matlib.repmat(m[np.newaxis].T,1,basis.shape[1])
#
#     dmat = np.hstack((basis,np.multiply(M,basis)))
#     if speed is not None:
#
#         dmat= np.hstack((dmat,speed[np.newaxis].T))
#     return dmat
#
# def gaussian_pdf(x,mu,sigma,univariate=True,eps=.01):
#     '''calculate pdf for given mean and covariance'''
#     if univariate:
#         # add a spmall epsilon
#         return 1/(2.*np.pi)**.5 * np.divide(np.exp(-1.*np.divide(np.power(x-mu,2),2*np.power(sigma,2))),sigma)
#     else:
#         # add a multivariate gaussian here
#
#         # check for poor conditioning of covariance matrix
#         pass
