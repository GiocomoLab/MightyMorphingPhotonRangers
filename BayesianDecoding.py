# insert code for doing MCN decoding analysis
import os
os.sys.path.append("C:\\Users\mplitt\MightyMorphingPhotonRangers")
import utilities as u
import preprocessing as pp
import numpy as np
import scipy as sp
import sklearn as sk
import sklearn.linear_model
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

def make_spline_basis(x,knots=np.arange(0,450,50)):
    '''make cubic spline basis functions'''
    knotfunc = lambda k: np.power(np.multiply(x-k,(x-k)>0),3)
    spline_basis_list = [knotfunc(k) for k in knots.tolist()]
    spline_basis_list += [np.ones(x.shape[0]),x,np.power(x,2)]
    return np.array(spline_basis_list).T



def pos_morph_design_matrix(x,m,splines=True,knots=np.arange(-50,450,50),speed=None):
    '''make design matrix for GLM that uses basis functions for position and separate regresssors for each context'''
    if splines:
        basis = make_spline_basis(x,knots=knots)
    else:
        # add functionality for radial basis functions
        pass

    M = np.matlib.repmat(m[np.newaxis].T,1,basis.shape[1])

    dmat = np.hstack((basis,np.multiply(M,basis)))
    if speed is not None:

        dmat= np.hstack((dmat,speed[np.newaxis].T))
    return dmat






class empirical_density:
    '''calculate empirical joint density'''
    def _init_(x,y,xknots = np.linspace(0,450), yknots=(-3,15)):
        self.xknots=xknots
        elf.yknots = yknots
        self.d = sp.interpolate.LSQBivariateSpline(x,y,xknots,yknots)
        self.N = d.integral(xknots[0],xknots[-1],yknots[0],yknots[-1])
    def pdf(self,xi,yi):
        return self.d.ev(xi,yi)/self.N

    def condy_x(self,xi,yi):
        return self.pdf(xi,yi)/self.d.integral(xi,xi,self.yknots[0],self.yknots[-1])

    def condx_y(self,xi,yi):
        return self.pdf(xi,yi)/self.d.integral(self.xknots[0],self.xknots[-1],yi,yi)





def gaussian_pdf(x,mu,sigma,univariate=True,eps=.01):
    '''calculate pdf for given mean and covariance'''
    if univariate:
        # add a spmall epsilon
        return 1/(2.*np.pi)**.5 * np.divide(np.exp(-1.*np.divide(np.power(x-mu,2),2*np.power(sigma,2))),sigma)
    else:
        # add a multivariate gaussian here

        # check for poor conditioning of covariance matrix
        pass

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
    return XX, bin_edges

def decoding_model(trial_C_z,XX_I0,XX_I1,mu_i0,mu_i1,morphs):

    # allocate for single cell data
    post_i0x_y, post_i1x_y= [],[]
    post_i0 ,post_i1 = [],[]

    # allocate for population data
    pop_post_i0x_y, pop_post_i1x_y = [], []
    pop_post_i0, pop_post_i1 = [], []

    for trial,I  in enumerate(morphs):

        #print(a.shape)
        if trial%5==0:
            print("processing trial %d" % trial)


        cz = trial_C_z[trial]
        post_trial0,post_trial1 = [],[]
        pop_post_trial0, pop_post_trial1 = [],[]
        for j in range(cz.shape[0]):
            if j==0: # if first timepoint, set initial conditions

                #### single cell initial conditions
                # set probability of being in current position to 1
                onehot = .001*np.ones([XX_I0.shape[1],1])
                onehot[0] = 1.
                onehot = onehot/onehot.ravel().sum()

                # multiply by prior on being in context (.5)
                tmp0 = .5*np.dot(onehot,np.ones([1,cz.shape[1]]))
                tmp1 = .5*np.dot(onehot,np.ones([1,cz.shape[1]]))

                # normalization factor to account for digitization of position
                tmp_denom = tmp0.sum(axis=0)+tmp1.sum(axis=0)
                tmp_denom = np.dot(np.ones([XX_I0.shape[0],1]),tmp_denom[np.newaxis])

                # posterior having observed 0 time points
                Z0_t = np.divide(tmp0,tmp_denom)
                Z1_t = np.divide(tmp1,tmp_denom)


                #### pop decoding initial conditions
                ttmp0 = .5*onehot
                ttmp1 = .5*onehot

                # normalization factor
                ttmp_denom = ttmp0.sum(axis=0)+ttmp1.sum(axis=0)

                # posterior having observed 1 time frame
                ZZ0_t = ttmp0/ttmp_denom
                ZZ1_t = ttmp1/ttmp_denom

            ######## single cell decoding
            XZ0 = np.dot(XX_I0,Z0_t)
            XZ1 = np.dot(XX_I1,Z1_t)

            # make activity into a matrix and means at each position into a matrix in order to calculate likelihoods
            CZX = np.matlib.repmat(cz[j,:],mu_i0.shape[0],1)

            l0 = gaussian_pdf(CZX,mu_i0,1)
            l1 = gaussian_pdf(CZX,mu_i1,1)
            denom = np.matlib.repmat(l0.sum(axis=0) + l1.sum(axis=0),mu_i0.shape[0],1)
            l0 = np.maximum(np.divide(l0,denom),.001)
            l1 = np.maximum(np.divide(l1,denom),.001)

            # numerator of new posterior
            tmpnum0 = np.multiply(XZ0,l0)
            tmpnum1= np.multiply(XZ1,l1)

            # normalization factor for updated posterior
            tmp_denom = tmpnum0.sum(axis=0)+tmpnum1.sum(axis=0)
            tmp_denom = np.dot(np.ones([XX_I0.shape[0],1]),tmp_denom[np.newaxis])

            # new posterior
            Z0_t = np.divide(tmpnum0,tmp_denom)
            Z1_t = np.divide(tmpnum1,tmp_denom)


            # add to list for trial
            post_trial0.append(Z0_t)
            post_trial1.append(Z1_t)


            # ######## population decoding
            XXZZ0 = np.dot(XX_I0,1*ZZ0_t) + np.dot(XX_I1,(1-1)*ZZ1_t)
            XXZZ1 = np.dot(XX_I0,(1-1)*ZZ0_t) + np.dot(XX_I1,1*ZZ1_t)

            # make activity into a matrix and means at each position into a matrix in order to calculate likelihoods
            CCZZXX = np.matlib.repmat(cz[j,:],mu_i0.shape[0],1)

            #calculate likelihoods as a function of binned position
            ll0 = gaussian_pdf(CCZZXX,mu_i0,1)
            ll1 = gaussian_pdf(CCZZXX,mu_i1,1)

            # normalize from binning
            ddenom = np.matlib.repmat(ll0.sum(axis=0) + ll1.sum(axis=0),mu_i0.shape[0],1)
            ll0 = np.divide(ll0,ddenom)
            ll1 = np.divide(ll1,ddenom)


            # population log-likelihood of current activity as a function of position
            log_L0 = np.log(ll0).sum(axis=1)
            log_L1 = np.log(ll1).sum(axis=1)

            # numerator of new posterior
            # first calculate in log space
            log_tmpnum0 = log_L0 + np.squeeze(np.log(XXZZ0))
            # bring back to values that won't overflow
            log_tmpnum0 -= log_tmpnum0.max()-1
            # back to normal space
            ttmpnum0 = np.exp(log_tmpnum0)

            # repeat
            log_tmpnum1 = log_L1 + np.squeeze(np.log(XXZZ1))
            log_tmpnum1 -= log_tmpnum1.max()-1
            ttmpnum1 = np.exp(log_tmpnum1)

            # normalization factor for updated posterior
            ttmp_denom = ttmpnum0.sum(axis=0)+ttmpnum1.sum(axis=0)


            # new posterior
            ZZ0_t = ttmpnum0/ttmp_denom
            ZZ1_t = ttmpnum1/ttmp_denom

            # add to list for trial
            pop_post_trial0.append(ZZ0_t)
            pop_post_trial1.append(ZZ1_t)




        # append trials posterior to list
        post_i0x_y.append(np.array(post_trial0))
        post_i1x_y.append(np.array(post_trial1))
        # sum across positions to get posterior of context
        post_i1.append(np.squeeze(np.array(post_trial1).sum(axis=1)))
        post_i0.append(np.squeeze(np.array(post_trial0).sum(axis=1)))


        # append trials population posterior to list
        pop_post_i0x_y.append(np.array(pop_post_trial0))
        pop_post_i1x_y.append(np.array(pop_post_trial1))
        # marginalize across position to get posterior of context
        pop_post_i1.append(np.squeeze(np.array(pop_post_trial1).sum(axis=1)))
        pop_post_i0.append(np.squeeze(np.array(pop_post_trial0).sum(axis=1)))


    return {'i0x_y':post_i0x_y,
            'i1x_y':post_i1x_y,
            'i0': post_i0,
            'i1':post_i1,
            'pop i0x_y': pop_post_i0x_y,
            'pop i1x_y':pop_post_i1x_y,
            'pop i0': pop_post_i0,
            'pop i1': pop_post_i1}



def single_session(sess,save=False,dirbase=None):
    VRDat, C,Cd,S, A = pp.load_scan_sess(sess)
    C_z = sp.stats.zscore(C,axis=0)
    trial_info, tstart_inds, teleport_inds = u.by_trial_info(VRDat)
    C_z_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C_z,VRDat['pos']._values,VRDat['tstart']._values,VRDat['teleport']._values)

    # find position transition probabilities
    morph0inds = VRDat['morph']==0
    XX_I0, xbins = transition_prob_matrix(VRDat['pos']._values[morph0inds],binsize=5)

    morph1inds = VRDat['morph']==1
    XX_I1, xbins = transition_prob_matrix(VRDat['pos']._values[morph1inds],binsize=5)
    xbins = np.array(xbins)
    pos_binned = np.digitize(VRDat['pos']._values,xbins,right=True)

    # get some more by trial info without averaging in spatial bins
    pcnt = np.zeros([VRDat.shape[0],])
    trial_pos, trial_C_z  = [], []
    trial_pos_binned = []
    trial_licks = []
    for i,(start,stop) in enumerate(zip(tstart_inds,teleport_inds)):
        pos = VRDat['pos']._values[start:stop]
        trial_pos.append(pos)
        trial_C_z.append(C_z[start:stop,:]) # restrict to well fit cells
        trial_pos_binned.append(pos_binned[start:stop])
        lick_inds = np.where(VRDat['lick']._values[start:stop]>0)[0]
        licks = np.zeros(pos.shape)
        licks[:]=np.nan
        licks[lick_inds]=pos[lick_inds]
        trial_licks.append(licks)
        pcnt[start:stop] = int(trial_info['rewards'][i]>0)

    #  set up encoding models
    dmat = pos_morph_design_matrix(VRDat['pos']._values,VRDat['morph']._values)
    train_inds = (((VRDat['morph']==1) | (VRDat['morph']==0)) & (VRDat['pos']>0))  # & pcnt>0
    dmat_extreme = dmat[train_inds,:]
    C_extreme = C_z[train_inds,:]

    glm_base = sk.linear_model.LinearRegression()

    # for cells in session
    glm_base.fit(dmat_extreme,C_extreme)
    mu_extreme_hat = glm_base.predict(dmat_extreme)

    # get data for estimating all likelihoods
    mu_i0 = glm_base.predict(pos_morph_design_matrix(xbins,np.zeros([xbins.shape[0],])))
    mu_i1 = glm_base.predict(pos_morph_design_matrix(xbins,np.ones([xbins.shape[0],])))


    decode_dict= decoding_model(trial_C_z,XX_I0,XX_I1,mu_i0,mu_i1,trial_info['morphs'])
    plot_cellsVtime(decode_dict,trial_info,save=save,dir=dirbase)
    llr(C_z,VRDat,glm_base,trial_info,save=save,dir=dirbase)
    plot_decoding(decode_dict,trial_info,trial_pos_binned,trial_licks,save=save,dir=dirbase)
    return decode_dict

def plot_cellsVtime(decode_dict,trial_info,save = False, dir=None):
    if save:
        try:
            os.makedirs(dir+"\\cellvtime")
        except:
            print("path exists")

    for i, (tmppost,m) in enumerate(zip(decode_dict['i1'],trial_info['morphs'])):
        #if m>0 and m<1:
        f,ax = plt.subplots()
        ax.imshow(tmppost.T,aspect='auto',cmap='cool',vmin=0,vmax=1)
        ax.set_title(m)
        if save:

            f.savefig("%s\\cellvtime\\trial%d_morph%2f.pdf" % (dir,i,m),format='pdf')


def llr(C_z,VRDat,glm,trial_info,save=False,dir=None):
    # get likelihood for each point in time for log-likelihood ratio
    tmp_mu_i0 = glm.predict(pos_morph_design_matrix(VRDat['pos'],np.zeros([VRDat['pos'].shape[0],])))
    l_i0 = gaussian_pdf(C_z,tmp_mu_i0,1)

    tmp_mu_i1 = glm.predict(pos_morph_design_matrix(VRDat['pos'],np.ones([VRDat['pos'].shape[0],])))
    l_i1 = gaussian_pdf(C_z,tmp_mu_i1,1)
    llr = np.log(l_i0).sum(axis=1) - np.log(l_i1).sum(axis=1)
    print(llr.shape)

    tstarts,tstops = np.where(VRDat['tstart']==1)[0],np.where(VRDat['teleport']==1)[0]
    trial_llr, trial_pos = [],[]
    for (start,stop) in zip(tstarts,tstops):
        trial_llr.append(llr[start:stop])
        trial_pos.append(VRDat['pos']._values[start:stop])

    f,ax = plt.subplots(1,5,figsize=[20,5])
    nbins = 100
    xbins = np.linspace(0,460,nbins)
    mcount = {0:np.zeros([nbins,]), .25:np.zeros([nbins,]), .5:np.zeros([nbins,]), .75:np.zeros([nbins,]), 1.:np.zeros([nbins,])}
    avgTrace = {0:np.zeros([nbins,]), .25:np.zeros([nbins,]), .5:np.zeros([nbins,]), .75:np.zeros([nbins,]), 1.:np.zeros([nbins,])}
    #for key in avgTrace.keys():
    #    avgTrace[key][:]=np.nan

    for i,(lr,tmppos,m,r)  in enumerate(zip(trial_llr,trial_pos,trial_info['morphs'],trial_info['rewards'])):
        if r>0:
            if m == 0:
                ax[0].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
                ax[0].set_ylim([-300,300])

            elif m == .25:
                ax[1].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
                ax[1].set_ylim([-300,300])
            elif m  == .5 :
                ax[2].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
                ax[2].set_ylim([-300,300])
            elif m == .75:
                ax[3].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
                ax[3].set_ylim([-300,300])

            elif m == 1.:
                ax[4].plot(tmppos[:],lr[:],color=plt.cm.cool(m),linewidth=.3)
                ax[4].set_ylim([-300,300])
        else:
            if m == 0:
                ax[0].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)

            elif m == .25:
                ax[1].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)

            elif m  == .5 :
                ax[2].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)

            elif m == .75:
                ax[3].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)


            elif m == 1.:
                ax[4].plot(tmppos[:],lr[:],color='black',linewidth=.3,alpha=.6)


        if i == 0:
            for j in range(5):
                ax[j].axhline(0,xmin=0,xmax=460,color='black',linewidth=.5)


        posbins = np.digitize(tmppos,xbins,right=True)
        #bcounts = np.bincounts(pos,xbins)
        emptyTrace = np.zeros([nbins,])
     #   emptyTrace[:]=np.nan
        for z in range(nbins):
            avgTrace[m][z] += lr[posbins==z].sum()
            mcount[m][z] += sum(posbins==z)


    f_mu,ax_mu = plt.subplots()
    for key in avgTrace.keys():
        avgTrace[key] = np.divide(avgTrace[key],mcount[key])
        #print(avgTrace[key])
        ax_mu.plot(xbins,avgTrace[key],color=plt.cm.cool(np.float(key)))
        ax_mu.axhline(0,xmin=0,xmax=460,color='black',linewidth=.5)

    if save:
        f.savefig("%s\\single_trial_llr.pdf" % dir,format='pdf')
        f_mu.savefig("%s\\trial_avg_llr.pdf" % dir,format='pdf')


def plot_decoding(decode_dict,trial_info,trial_pos_binned,trial_licks,save = False, dir = None,rzone0=[250,315],rzone1=[350,415]):

    rzone0 = [i/5 for i in rzone0]
    rzone1 = [i/5 for i in rzone1]
    if save:
        try:
            os.makedirs("%s\\decoding" % dir)
        except:
            print("error making directory")

    for t in range(trial_info['morphs'].shape[0]):
        px = np.hstack((decode_dict['pop i0x_y'][t],decode_dict['pop i1x_y'][t])).T
        #print(px.sum(axis=0))
        gs = gridspec.GridSpec(5,1)
        f = plt.figure(figsize=[10,10])
        ax = f.add_subplot(gs[0:-1,:])
        #f, ax = plt.subplots(2,1,figsize= [15,5],sharex=True)
        ax.imshow(px,aspect = 'auto',cmap='magma',alpha=.4,zorder=2)

        ax.axhline(93,xmin=0,xmax=px.shape[1],color='white',linewidth=5,zorder=10)

        ax.plot(trial_pos_binned[t], color = plt.cm.cool(0.),linewidth=2,zorder=0,alpha=.5)
        ax.plot(trial_pos_binned[t]+93, color = plt.cm.cool(1.),linewidth=2,zorder=0,alpha=.5)
        ax.fill_between(np.arange(px.shape[1]),rzone0[0],y2 = rzone0[1],color=plt.cm.cool(0.),alpha=.2)
        ax.fill_between(np.arange(px.shape[1]),rzone1[0],y2 = rzone1[1],color=plt.cm.cool(1.),alpha=.2)
        ax.fill_between(np.arange(px.shape[1]),rzone0[0]+93,y2 = rzone0[1]+93,color=plt.cm.cool(0.),alpha=.2)
        ax.fill_between(np.arange(px.shape[1]),rzone1[0]+93,y2 = rzone1[1]+93,color=plt.cm.cool(1.),alpha=.2)
        ax.set_xlim([0,px.shape[1]])

        x = np.arange(px.shape[1])
        licks = trial_licks[t]

        ax.scatter(x,licks/5,s=50,marker='x',color='blue',alpha=.5,zorder=1)
        ax.scatter(x,licks/5+93,s=50,marker='x',color='blue',alpha=.5,zorder=1)
        ax.set_title("trial %d morph %f reward %f" % (t,trial_info['morphs'][t],trial_info['rewards'][t]))

        aax = f.add_subplot(gs[-1,:],sharex=ax)
        aax.scatter(x,decode_dict['pop i0'][t],c=plt.cm.cool(decode_dict['pop i1'][t]))
        aax.set_ylim([0,1])
        aax.axhline(.5,xmin=0,xmax=px.shape[1])
        aax.set_xlim([0,px.shape[1]])
        if save:
            f.savefig("%s\\decoding\\trial%d_morph%2f_reward%d.pdf" % (dir,t,trial_info['morphs'][t],int(trial_info['rewards'][t])),format='pdf')
