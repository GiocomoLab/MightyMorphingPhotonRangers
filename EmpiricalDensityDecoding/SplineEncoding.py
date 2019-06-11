import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.linear_model import RidgeCV, LinearRegression
import os

os.sys.path.append("C:\\Users\\mplitt\\MightyMorphingPhotonRangers")
import utilities as u


class NBDecodingModel:
    def __init__(self,ncells,ops={},nx=45,nm=5,nc=5):
        # nx,nm,nc = 45,10,10
        self.n_cells = ncells
        self.nx,self.nm,self.nc=nx,nm,nc
        self._x = np.linspace(450/nx,450,num=nx)
        self._m = np.linspace(1/nm,1,num=nm)
        self._c = np.linspace(100/nc,100,num=nc)
        # self._x = np.linspace(0.001,449.999,num=nx)
        # self._m = np.linspace(0.001,.999,num=nm)
        # self._c = np.linspace(0.001,99.99,num=nc)
        # self._X = np.matmul(self._x.reshape([-1,1]),np.ones([1,nm]))
        # self._M = np.matmul(np.ones([nx,1]),self._m.reshape([1,-1]))
        try:
            self.J = np.memmap("E:\\J.dat",dtype='float16',mode='w+',
                shape=(nx,nm,nc,ncells))
        except:
            self.J = np.memmap("E:\\J.dat",dtype='float16',mode='r+',
                shape=(nx,nm,nc,ncells))
        print(self.J.shape)
        #np.meshgrid(self._x,self._c)
        self._set_ops(ops)
        # self._cells = {}

    def _set_ops(self,ops_in):
        ops_out = {}
        for k,v in ops_in.items():
            ops_out[k]=v

        self.ops = ops_out

    def fit_cells(self,pos,morph,C):
        # self._fit_priors(pos,morph)
        assert pos.shape==morph.shape, "position and morph vectors need to be of same length"
        assert pos.shape[0] == C.shape[0], "position, morph and activity vectors need to be of the same length"

        # self.n_cells = C.shape[1]
        # dummymdl = EncodingModel(ops=self.ops)
        FEATS = np.zeros([pos.shape[0],3])
        FEATS[:,0],FEATS[:,1]=pos,morph


        for c in range(C.shape[1]):
            FEATS[:,2] = C[:,c]
            mdl = EncodingModel(ops=self.ops)
            xbins,mbins,cbins = np.zeros([self.nx+1,]),np.zeros([self.nm+1,]),np.zeros([self.nc+1,])
            xbins[1:],mbins[1:],cbins[1:] = self._x,self._m,self._c
            if c ==0:
                Count, edges = np.histogramdd(FEATS,bins=[xbins,mbins,cbins])
                centers= []
                for ed in edges:
                    centers.append(ed[:-1]+.5*(ed[1:]-ed[:-1]))

                self._centers = centers
                print(centers)
                P,M,Z = np.meshgrid(centers[0],centers[1],centers[2],indexing='ij')


                spl_basis = mdl.make_design_matrix(P.ravel(),M.ravel(),Z.ravel())
                # self._cells[c]=EncodingModel(ops=self.ops)
                # spl_basis = self._cells[c].make_design_matrix(P.ravel(),M.ravel(),Z.ravel())
            # else:

                # self._cells[c] = EncodingModel(ops=self.ops)

            Count, tmp = np.histogramdd(FEATS,bins=[xbins,mbins,cbins])
            Count+=1000
            mdl.fit_linear(spl_basis,Count.ravel())

            self.J[:,:,:,c]= mdl.predict_linear(spl_basis).reshape(P.shape)

    def get_likelihood(self,C):

        L = np.zeros([self.nx,self.nm,C.shape[0]])
        C_d = np.digitize(C,self._c,right=True)
        print(np.amax(C),self._c[-1],np.amax(C_d),np.amin(C_d))
        for t in range(C.shape[0]):
            if t%1000==0:
                print(t)
            J_t = np.array([self.J[:,:,d,ind] for ind,d in enumerate(C_d[t,:].tolist())])
            if (J_t<=0).sum()>0:
                print((J_t<=0).sum())
            logJ_t = np.log(J_t).sum(axis=0)
            logJ_t-=np.amax(logJ_t)
            P_XM_t = np.exp(logJ_t)
            P_XM_t/=P_XM_t.ravel().sum()
            #
            L[:,:,t]=P_XM_t

        return L
        #
        # P_XM = np.zeros([self._x.shape[0],self._m.shape[0],C.shape[0]])
        #
        # for t in range(C.shape[0]):
        #     J_t = self._get_likelihood_1timepoint(C[t,:])
        #     logJ_t = np.log(J_t).sum(axis=-1)
        #     logJ_t-=np.amax(logJ_t)
        #     P_XM_t = np.exp(logJ_t)
        #     P_XM_t/=P_XM.ravel().sum()
        #
        #     P_XM[:,:,t]=P_XM_t
        #
        # return P_XM

    def _get_likelihood_1cell(self,c,cell):
        J = np.zeros([self._x.shape[0],self._m.shape[0],c.shape[0]])
        for ind,m in enumerate(self._m.tolist()):
            print(ind)
            X,c_vec = np.meshgrid(self._x,c)
            M  = m*np.ones(X.shape)

            dm_denom = self._cells[cell].make_design_matrix(X.ravel(),M.ravel(),c_vec.ravel())
            J_denom = self._cells[cell].predict_linear(dm_denom)

            J[:,ind,:]=J_denom.reshape([self._x.shape[0],c.shape[0]])

        return J



    def _get_likelihood_1timepoint(self,C):

        J = np.zeros([self._x.shape[0],self._m.shape[0],self.n_cells])
        for cell in range(C.shape[0]):

            # c = C[cell]

            # dm_num = self._cells[cell].make_design_matrix(np.array([x]),np.array([m]),np.array([c]))
            # J_num = self._cells[cell].predict(dm_num)

            c_vec = C[cell]*np.ones(self._X.shape)
            # print(c_vec.shape)
            dm_denom = self._cells[cell].make_design_matrix(self._X.ravel(),self._M.ravel(),c_vec.ravel())
            J_denom = self._cells[cell].predict_linear(dm_denom)
            J_cell = J_denom/J_denom.sum()
            J[:,:,cell]=J_cell.reshape([self._x.shape[0],self._m.shape[0]])

        return J


        # dummyEM = EncodingModel(ops = ops)
        # decodingDM = dummyEM.make_design_matrix(self._X.ravel(),self._C.ravel())


    def _fit_priors(self,pos,morph):
        kd = KernelDensity()
        kd.fit(pos.reshape([-1,1]))
        self.p_x = np.exp(kd.score_samples(self._x.reshape([-1,1]))).reshape([-1,1])
        self.p_x/=self.p_x.sum()
        # p_x,edges = np.histogram(pos,bins=50,density=True)
        # self.p_x = p_x.reshape([50,1])
        kd.fit(morph.reshape([-1,1]))
        self.p_m = np.exp(kd.score_samples(self._m.reshape([-1,1]))).reshape([1,-1])
        self.p_m/=self.p_m.sum()
        # p_c,edges = np.histogram(morph,bins=50,density=True)
        # self.p_c = p_c.reshape([1,50])






class EncodingModel:
    def __init__(self,ops={}):
        self._set_ops(ops)
        self._set_ctrl_pts()
        s= self.ops['s']
        self.S = np.array([[-s, 2-s, s-2, s],
                    [2*s, s-3, 3-2*s, -s],
                    [-s, 0, s, 0,],
                    [0, 1, 0, 0]])
        # print(self.pos_ctrl_pts.shape)
        #self.coefs = np.zeros([self._n_coefs,])

    def _set_ops(self,ops_in):

        ops_out={'key':0,
        'n_ctrl_pts_pos':3,
        'n_ctrl_pts_morph':3,
        'n_ctrl_pts_activity':3,
        'n_pos_bins':45,
        'n_morph_bins':10,
        'n_activity_bins':10,
        'max_pos':450.001,
        'RidgeCV':True,
        's':.5}
        for k,v in ops_in.items():
            #print(k,v)
            ops_out[k]=v

        self.ops = ops_out
        #print(self.ops)
        self._n_coefs = (self.ops['n_ctrl_pts_pos']+2)*(self.ops['n_ctrl_pts_morph']+2)*(self.ops['n_ctrl_pts_activity']+2)

    def _set_ctrl_pts(self):
        # need to pad original ctrl point vector
        self.pos_ctrl_pts = np.zeros([self.ops['n_ctrl_pts_pos']+2,])
        pos_ctrl_pts = np.linspace(0,self.ops['max_pos'],num=self.ops['n_ctrl_pts_pos'])
        dpos = pos_ctrl_pts[1]-pos_ctrl_pts[0]
        self.pos_ctrl_pts[0] = pos_ctrl_pts[0]-dpos
        self.pos_ctrl_pts[1:-1]=pos_ctrl_pts
        self.pos_ctrl_pts[-1]=pos_ctrl_pts[-1]+dpos

        self.morph_ctrl_pts = np.zeros([self.ops['n_ctrl_pts_morph']+2,])
        morph_ctrl_pts = np.linspace(0,1.001,num=self.ops['n_ctrl_pts_morph'])
        dmorph = morph_ctrl_pts[1]-morph_ctrl_pts[0]
        self.morph_ctrl_pts[0]=morph_ctrl_pts[0]-dmorph
        self.morph_ctrl_pts[1:-1]=morph_ctrl_pts
        self.morph_ctrl_pts[-1]=morph_ctrl_pts[-1]+dmorph

        self.activity_ctrl_pts = np.zeros([self.ops['n_ctrl_pts_activity']+2,])
        activity_ctrl_pts = np.linspace(0,100.001,num=self.ops['n_ctrl_pts_activity'])
        dact = activity_ctrl_pts[1]-activity_ctrl_pts[0]
        self.activity_ctrl_pts[0] = activity_ctrl_pts[0]-dact
        self.activity_ctrl_pts[1:-1]=activity_ctrl_pts
        self.activity_ctrl_pts[-1]=activity_ctrl_pts[-1]+dact


    def make_spline(self,pos,morph,C):
        # print(pos.shape,morph.shape,C.shape)
        assert pos.shape==morph.shape, "position and morph vectors need to be of same length"
        assert pos.shape[0] == C.shape[0], "position, morph and activity vectors need to be of the same length"

        splbasis= np.zeros([pos.shape[0],self._n_coefs])
        for i in range(pos.shape[0]):
            p,m,c = pos[i],morph[i],C[i]
            # print(p,m,c)
            # print(self.pos_ctrl_pts,self.)
            x_p = self._1d_spline_coeffs(self.pos_ctrl_pts,p)
            x_m = self._1d_spline_coeffs(self.morph_ctrl_pts,m)
            # print(self.activity_ctrl_pts.shape,c)
            x_c = self._1d_spline_coeffs(self.activity_ctrl_pts,c)
            #print(x_p.shape,x_m.shape)
            # loop through activity control points
            x_pmc = np.kron(x_c.reshape([-1,1]),np.matmul(x_p.reshape([-1,1]),x_m.reshape([1,-1])))
            splbasis[i,:]=x_pmc.ravel()

        return splbasis

    def _1d_spline_coeffs(self,ctrl,v):

        x = np.zeros(ctrl.shape)

        # nearest ctrl pt
        ctrl_i = (ctrl<v).sum()-1
        pre_ctrl_pt = ctrl[ctrl_i]

        # next ctrl pt
        post_ctrl_pt = ctrl[ctrl_i+1]

        alpha = (v-pre_ctrl_pt)/(post_ctrl_pt-pre_ctrl_pt)
        u = np.array([alpha**3, alpha**2, alpha, 1]).reshape([1,-1])
        # p =
        # print(v,ctrl_i,np.matmul(u,self.S).shape)
        x[ctrl_i-1:ctrl_i+3] = np.matmul(u,self.S)
        return x


    def make_design_matrix(self,pos,morph,C):
        return self.make_spline(pos,morph,C)


    def fit_linear(self,X,y):
        if self.ops['RidgeCV']:
            # mdl = LinearRegression(fit_intercept=False)
            mdl = RidgeCV(fit_intercept=False)

            mdl.fit(X,y)


            # print(mdl.alpha_)

        else:
            mdl = LinearRegression(fit_intercept=False)
            mdl.fit(X,y)

        self.coef_ = mdl.coef_
        self.mdl_ = mdl

        # self.alpha_ = mdl.alpha_

    def predict_linear(self,X):
        return self.mdl_.predict(X)


    def fit_model(self,pos,morph,C):
        assert pos.shape==morph.shape, "position and morph vectors need to be of same length"
        assert pos.shape[0] == C.shape[0], "position, morph and activity vectors need to be of the same length"

        self._FEATS = np.zeros([pos.shape[0],3])
        self._FEATS[:,0],self._FEATS[:,1],self._FEATS[:,2] = pos,morph,C

        self._Count, edges = np.histogramdd(FEATS,bins=[self.ops['n_pos_bins'],
                                                    self.ops['n_morph_bins'],
                                                    self.ops['n_activity_bins']])

        self._centers = []
        for ed in edges:
            self._centers.append(ed[:-1]+ (ed[1:]-ed[:-1]).mean())

        P,M,Z = np.meshgrid(self._centers[0],self._centers[1],self._centers[2],indexing='ij')
        DM = self.make_design_matrix(P.ravel(),M.ravel(),Z.ravel())
        self.fit_linear(DM,self._Count.ravel())
