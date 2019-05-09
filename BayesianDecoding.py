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

def transition_prob_matrix(x,binsize=5,sig=4):
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
    XX_smooth = gaussian_filter1d(XX,sig,axis=0)
    for b in np.unique(x_binned).tolist():
        XX_smooth[:,b]/=XX_smooth[:,b].sum()

    return XX, bin_edges

def empirical_decoding_model(L,T,starts, stops, prefix = "E:\\"):

    try:
        # allocate for single cell data
        post_ix= np.memmap(os.path.join(prefix,"post_ix.dat"),dtype='float32',mode='w+',shape=tuple(L.shape))#np.zeros(L.shape)
    except:
        # allocate for single cell data
        post_ix= np.memmap(os.path.join(prefix,"post_ix.dat"),dtype='float32',mode='r+',shape=tuple(L.shape))#np.zeros(L.shape)

    try:
        post_i = np.memmap(os.path.join(prefix,"post_i.dat"),dtype='float32', mode='w+',shape=(L.shape[0],L.shape[2])) #np.zeros([L.shape[0],L.shape[2]])
    except:
        post_i = np.memmap(os.path.join(prefix,"post_i.dat"),dtype='float32', mode='r+',shape=(L.shape[0],L.shape[2])) #np.zeros([L.shape[0],L.shape[2]])

        # allocate for population data
    try:
        pop_post_ix = np.memmap(os.path.join(prefix,"pop_post_ix.dat"),dtype='float32',mode='w+',shape= tuple(L.shape[0:2]))
    except:
        pop_post_ix = np.memmap(os.path.join(prefix,"pop_post_ix.dat"),dtype='float32',mode='r+',shape= tuple(L.shape[0:2]))

    try:
        pop_post_i = np.memmap(os.path.join(prefix,"pop_pos_i.dat"),dtype='float32',mode='w+',shape= (L.shape[0],))
    except:
        # allocate for population data
        pop_post_i = np.memmap(os.path.join(prefix,"pop_pos_i.dat"),dtype='float32',mode='r+',shape= (L.shape[0],))


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
                B = B*np.ones([1,L.shape[2]])
                # B = np.matlib.repmat(B,1,L.shape[2])

                BB = np.ones([T.shape[0],1])
                #BB[0,:] = .5
                BB/= BB.ravel().sum()


            ######## single cell decoding
            # new posterior - unnormalized
            B = np.multiply(np.dot(T,B),np.squeeze(L[start+j,:,:]))
            # denominator for normaliztion
            d = np.ones([T.shape[0],1])*B.sum(axis=0)
            # d = np.matlib.repmat(B.sum(axis=0),T.shape[0],1)
            B = np.divide(B,d)

            post_ix[start+j,:,:] = B
            post_i[start+j,:] = B[NX:,:].sum(axis=0)
            # ensure no underflow
            B=np.minimum(.0001+B, .9999)
            d = B.sum(axis=0)*np.ones([T.shape[0],1])
            # d = np.matlib.repmat(B.sum(axis=0),T.shape[0],1)
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


    post_ix.flush()
    post_i.flush()
    pop_post_ix.flush()
    pop_post_i.flush()

    return {'cell ix':post_ix,
            'cell i':post_i,
            'pop ix': pop_post_ix,
            'pop i': pop_post_i}



class single_session:
    def __init__(self,sess,save=False,trainOnCorrect=True,prefix = "E:\\"):
        try:
            os.makedirs(prefix)
        except:
            pass

        VRDat, C,S, A = pp.load_scan_sess(sess)
        C=u.df(C)
        C_z = sp.stats.zscore(C,axis=0)
        # C_z = S
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

        XX_tmp, xbins = transition_prob_matrix(pos[inds0|inds1],binsize=5)
        n = XX_I0.shape[0]
        XX = np.zeros([XX_tmp.shape[0]*2,XX_tmp.shape[1]*2])
        XX[:XX_tmp.shape[0],:XX_tmp.shape[1]]=XX_tmp
        XX[XX_tmp.shape[0]:,:XX_tmp.shape[1]]=XX_tmp
        XX[:XX_tmp.shape[0],XX_tmp.shape[1]:]=XX_tmp
        XX[XX_tmp.shape[0]:,XX_tmp.shape[1]:]=XX_tmp
        # XX =  np.matlib.repmat(XX_tmp,2,2)

        xbins = np.array(xbins)
        return XX, xbins

    def likelihood_maps(self,mmap = True):
        # get likelihoods at all timepoints - functionalize
        if mmap:
            try:
                L = np.memmap(os.path.join(self.prefix,"L.dat"),dtype='float32',mode='r+',shape = (self.C_z.shape[0],self.xbins.shape[0]*2,self.C_z.shape[1]))
            except:
                L = np.memmap(os.path.join(self.prefix,"L.dat"),dtype='float32',mode='w+',shape = (self.C_z.shape[0],self.xbins.shape[0]*2,self.C_z.shape[1]))
        else:
            L = np.zeros( (self.C_z.shape[0],self.xbins.shape[0]*2,self.C_z.shape[1]))
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

    def ctxt_LLR(self,L=None,save=True):
        if L is None:
            L=self.likelihood_maps()

        nbins = self.xbins.shape[0]
        fr, px0 = u.rate_map(self.C_z[self.inds0],self.pos[self.inds0],bin_size=5,max_pos=465)
        fr, px1 = u.rate_map(self.C_z[self.inds1],self.pos[self.inds1],bin_size=5,max_pos=465)

        px0= np.ones([L.shape[0],1])*px0[np.newaxis]
        # px0 = np.matlib.repmat(px0[np.newaxis],L.shape[0],1)

        # px1 = np.matlib.repmat(px1[np.newaxis],L.shape[0],1)
        px1= np.ones([L.shape[0],1])*px1[np.newaxis]
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
        if save:
            np.savez(os.path.join(self.prefix,'ctxt_LLR.npz'),LLR,LLR_pop)

        return LLR,LLR_pop

    def independent_decoder(self,L=None):
        #not working yet
        if L is None:
            L = self.likelihood_maps()
        try:
            Z = np.memmap(os.path.join(self.prefix,"L.dat"),dtype='float32',mode='r+',shape = (self.C_z.shape[0],self.xbins.shape[0]*2,self.C_z.shape[1]))
        except:
            Z = np.memmap(os.path.join(self.prefix,"L.dat"),dtype='float32',mode='w+',shape = (self.C_z.shape[0],self.xbins.shape[0]*2,self.C_z.shape[1]))
        nbins = self.xbins.shape[0]
        fr, px0 = u.rate_map(self.C_z[self.inds0],self.pos[self.inds0],bin_size=5,max_pos=465)
        fr, px1 = u.rate_map(self.C_z[self.inds1],self.pos[self.inds1],bin_size=5,max_pos=465)
        px = np.append(px0,px1)
        px/=px.sum()
        px = np.ones([Z.shape[0],1])*px[np.newaxis]
        # px = np.matlib.repmat(px[np.newaxis],Z.shape[0],1)

        for c in range(L.shape[2]):
            if c%50==0:
                print(c)
            tmp=np.multiply(np.squeeze(L[:,:,c]),px)
            d = np.dot(tmp.sum(axis=1)[np.newaxis].T,np.ones([1,px.shape[1]]))
            Z[:,:,c]=np.divide(tmp,d)+1E-10

        logZZ = np.log(Z).sum(axis=2)
        ZZ = np.exp(logZZ-np.max(logZZ)-1)
        d = np.dot(ZZ.sum(axis=1)[np.newaxis].T,np.ones([1,px.shape[1]]))
        ZZ=np.divide(ZZ,d)
        return {'cell ix':Z,
                'cell i':np.squeeze(Z[:,self.xbins.shape[0]:,:].sum(axis=1)),
                'log pop ix': logZZ,
                'pop ix': ZZ,
                'pop i': np.squeeze(ZZ[:,self.xbins.shape[0]:].sum(axis=1))}









    def run_decoding(self, L=None):
        if L is None:
            L = self.likelihood_maps()

        return empirical_decoding_model(L,self.XX,self.tstarts,self.teleports,prefix = self.prefix)



    def plot_decoding(self,decode_dict,rzone0=[250,315],rzone1=[350,415],save=False,cellsort=None):
        rzone0 = [i/5 for i in rzone0]
        rzone1 = [i/5 for i in rzone1]
        if save:
            try:
                os.makedirs("%s\\decoding" % self.prefix)
            except:
                print("error making directory")

        if cellsort is None:
            cellsort = np.arange(decode_dict['cell i'].shape[1])

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
            aax.scatter(x,decode_dict['pop i'][start:stop],c=plt.cm.cool(decode_dict['pop i'][start:stop]))
            aax.axhline(0,xmin=0,xmax=px.shape[1])
            aax.set_xlim([0,px.shape[1]])
            aax.set_ylim([-.2,1.2])

            aaax= f.add_subplot(gs[7:,:],sharex=ax)
            aaax.imshow(decode_dict['cell i'][start:stop,cellsort].T,cmap = 'cool',aspect='auto',vmin=0,vmax=1)
            if save:
                f.savefig("%s\\decoding\\trial%d_morph%2f_reward%d.png" % (self.prefix,t,self.trial_info['morphs'][t],int(self.trial_info['rewards'][t])),format='png')


    def plot_llr(self,LLR,save=False):

        keys = np.unique(self.trial_info['morphs'])
        nm = keys.shape[0]
        # pos binned data
        llr_pos,occ,edges,centers = u.make_pos_bin_trial_matrices(LLR,self.pos,self.tstarts,self.teleports)

        # by morph means
        d_pos = u.trial_type_dict(llr_pos,self.trial_info['morphs'])


        # time-binned data
        llr_time = u.make_time_bin_trial_matrices(LLR,self.tstarts,self.teleports)

        # by morph means
        d_time = u.trial_type_dict(llr_time,self.trial_info['morphs'])

        mu_pos = np.zeros([keys.shape[0],llr_pos.shape[1]])
        sem_pos = np.zeros([keys.shape[0],llr_pos.shape[1]])
        mu_time = np.zeros([keys.shape[0],llr_time.shape[1]])
        sem_time = np.zeros([keys.shape[0],llr_time.shape[1]])
        for j,k in enumerate(keys):
            mu_pos[j,:] = np.nanmean(d_pos[k],axis=0)
            sem_pos[j,:] = np.nanstd(d_pos[k],axis=0)
            mu_time[j,:] = np.nanmean(d_time[k],axis=0)
            sem_time[j,:] = np.nanstd(d_time[k],axis=0)

        # actually plot stuff
        f_mp,ax_mp = plt.subplots()
        f_mt,ax_mt = plt.subplots()
        time = np.arange(0,mu_time.shape[1])*1/15.46
        for z in range(nm):
            # for zz in range(5):
                ax_mp.plot(centers,mu_pos[z,:],color=plt.cm.cool(keys[z]))
                ax_mt.plot(time,mu_time[z,:],color=plt.cm.cool(keys[z]))

                ax_mp.fill_between(centers,mu_pos[z,:]+sem_pos[z,:],y2=mu_pos[z,:]-sem_pos[z,:],
                                color=plt.cm.cool(keys[z]),alpha=.4)
                ax_mt.fill_between(time,mu_time[z,:]+sem_time[z,:],y2=mu_time[z,:]-sem_time[z,:],
                                color=plt.cm.cool(keys[z]),alpha=.4)

        ax_mp.set_xlabel('position')
        ax_mp.set_ylabel('LLR')
        ax_mt.set_xlabel('time')
        ax_mt.set_ylabel('LLR')

        # ff_pos,aax_pos = plt.subplots()
        f_pos,ax_pos = plt.subplots(1,nm,figsize=[20,5])
        f_time,ax_time = plt.subplots(1,nm,figsize=[20,5])
        # ff_time,aax_time = plt.subplots()
        for i,(start,stop,m,r)  in enumerate(zip(self.tstarts.tolist(),self.teleports.tolist(),
                self.trial_info['morphs'].tolist(),self.trial_info['rewards'].tolist())):
            # print(start,stop,m,r,wj,bj)
            self._single_line_llr_multiax(self.pos[start:stop],LLR[start:stop],m,r,ax_pos)
            self._single_line_llr_multiax(np.arange(stop-start)*1./15.46,LLR[start:stop],m,r,ax_time,xlim=[0,250])


        ax_pos[0].set_xlabel('position')
        # aax_pos.set_xlabel('position')
        ax_time[0].set_xlabel('time')
        # aax_time.set_xlabel('time')
        for z in [0, -1]:
            for a in range(nm):

                ax_pos[a].fill_between(centers,mu_pos[z,:]+sem_pos[z,:],y2=mu_pos[z,:]-sem_pos[z,:],
                            color=plt.cm.cool(keys[z]),alpha=.4)
                ax_time[a].fill_between(time,mu_time[z,:]+sem_time[z,:],y2=mu_time[z,:]-sem_time[z,:],
                            color=plt.cm.cool(keys[z]),alpha=.4)

        if save:
            try:
                os.makedirs(self.prefix)
            except:
                pass
            f_mp.savefig(os.path.join(self.prefix,"LLR_position.png"),format="png")
            f_mt.savefig(os.path.join(self.prefix,"LLR_time.png"),format="png")
            f_pos.savefig(os.path.join(self.prefix,"LLR_pos_st.png"),format="png")
            f_time.savefig(os.path.join(self.prefix,"LLR_time_st.png"),format="png")

        return (f_mp,ax_mp),(f_mt,ax_mt),(f_pos,ax_pos),(f_time,ax_time) #, (ff_pos,aax_pos), (f_time,ax_time), (ff_time,aax_time)
        # edit axes




    def _single_line_llr_multiax(self,x,y,m,r,ax,lw=.5,
            ylim=[-100,100],xlim=[0,460]):
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
                ax[-1].plot(x,y,color=plt.cm.cool(m),linewidth=lw)
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
                ax[-1].plot(x,y,color='black',linewidth=lw,alpha=.6)

    def confusion_matrix(self,decode_dict,save=False):
        d_trial_mat, tr, edges, centers = u.make_pos_bin_trial_matrices(decode_dict['pop ix'],self.pos,self.tstarts,self.teleports)
        d_m_dict = u.trial_type_dict(d_trial_mat,self.trial_info['morphs'])

        keys = np.unique(self.trial_info['morphs'])
        c = np.zeros([d_trial_mat.shape[-1],d_trial_mat.shape[1]*keys.shape[0]])
        for n,key in enumerate(keys.tolist()):
            c[:,n*d_trial_mat.shape[1]:(n+1)*d_trial_mat.shape[1]]=np.nanmean(d_m_dict[key],axis=0).T

        f,ax = plt.subplots()
        ax.imshow(c,cmap='viridis',vmin=0,vmax=.5)
        ax.set_xlabel('True Label')
        ax.set_ylabel('Decoded Label')
        if save:
            try:
                os.makedirs(self.prefix)
            except:
                pass
            f.savefig(os.path.join(self.prefix,"confusion_matrix.png"),format="png")
        return c, (f,ax)
