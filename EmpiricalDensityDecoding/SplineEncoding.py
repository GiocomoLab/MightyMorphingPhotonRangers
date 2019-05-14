import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.linear_model import RidgeCV, LinearRegression
import os

os.sys.path.append("C:\\Users\\mplitt\\MightyMorphingPhotonRangers")
import utilities as u


class NBDecodingModel:
    def __init__(self,ops={}):
        self._x = np.linspace(0.001,449.999,num=50)
        self._m = np.linspace(0.001,.999,num=50)
        self._c = np.linspace(0.001,99.99,num=50)

        self._set_ops(ops)
        self._cells = {}



        dummyEM = EncodingModel(ops = ops)
        decodingDM = dummyEM.make_design_matrix(self._X.ravel(),self._C.ravel())
        self.decodingDM = decodingDM



    def _set_ops(self,ops_in):
        ops_out = {}
        for k,v in ops_in.items():
            ops_out[k]=v

        self.ops = ops_out

    def fit_population(self,pos,morph,S):
        self._fit_priors(pos,morph)
        self.n_cells = S.shape[1]
        splmdl = EncodingModel(ops=self.ops)
        for c in range(S.shape[1]):
            X = splmdl.make_design_matrix(pos,morph,S[:,c])
            self._cells[c] = EncodingModel(ops=self.ops)
            self._cells[c].fit_linear(X,S[:,c])

    def _fit_priors(self,pos,morph):
        kd = KernelDensity()
        kd.fit(pos.reshape([-1,1]))
        self.p_x = np.exp(kd.score_samples(self._x.reshape([-1,1]))).reshape([-1,1])
        self.p_x/=self.p_x.sum()
        # p_x,edges = np.histogram(pos,bins=50,density=True)
        # self.p_x = p_x.reshape([50,1])
        kd.fit(morph.reshape([-1,1]))
        self.p_c = np.exp(kd.score_samples(self._c.reshape([-1,1]))).reshape([1,-1])
        self.p_c/=self.p_c.sum()
        # p_c,edges = np.histogram(morph,bins=50,density=True)
        # self.p_c = p_c.reshape([1,50])


    def poisson_predict_rate(self,cells = None):
        if cells is None:
            cells = np.ones([self.n_cells,])
            cells = cells>0
        Lam = np.zeros([self.decodingDM.shape[0],self.n_cells])
        Lam[:]=np.nan
        for cell,model in self._cells.items():
            if cells[cell]:
                Lam[:,cell]=model.predict_poisson(self.decodingDM).ravel()

        self.Lam=Lam

    def poisson_predict_likelihood_1timepoint(self,S,cells=None):
        if cells is None:
            cells = np.ones([self.n_cells,])
            cells = cells>0
        # try:
        #     L_allcells= np.memmap(os.path.join("E:\\L_allcells.dat"),dtype='float32',
        #         mode='r+',shape=(self.Lam.shape[0],self.n_cells,s.shape[0]))
        # except:
        #     L_allcells= np.memmap(os.path.join("E:\\L_allcells.dat"),dtype='float32',
        #         mode='w+',shape=(self.Lam.shape[0],self.n_cells,S.shape[0]))
        # L_allcells = np.zeros([self.Lam.shape[0],self.n_cells])

        s_v = np.matmul(np.ones([self.Lam.shape[0],1]),S.reshape([1,-1]))
        L_allcells = _gamma_pdf(s_v.ravel(),self.Lam.ravel()).reshape(self.Lam.shape)


        # lam_v = np.matmul(lam.reshape([-1,1]),np.ones([1,s.shape[0]]))
        #
        # for cell in self._cells.keys():
        #     if cells[cell]:
        #         s = S[cell]
        #         lam = self.Lam[:,cell]
        #         s_v = np.matmul(np.ones([lam.shape[0],1]),s.reshape([1,-1]))
        #         lam_v = np.matmul(lam.reshape([-1,1]),np.ones([1,s.shape[0]]))
        #         # s_v,lam_v = np.meshgrid(s,lam)
        #
        #         # L_allcells[:,cell,:] = _gamma_pdf(s_v.ravel(),lam_v.ravel()).reshape(s_v.shape)
        #         L_allcells[:,cell,:] = _gamma_pdf(s_v.ravel(),lam_v.ravel()).reshape(s_v.shape)
        #
        return L_allcells

    def poisson_decode(self,S,cells = None):
        P_XC = np.zeros([S.shape[0],self._x.shape[0],self._c.shape[0]])

        for t in range(S.shape[0]):

            L_allcells = self.poisson_predict_likelihood_1timepoint(S[t,:])+1E-5
            a = np.isnan(L_allcells)
            if a.sum()>0:
                print(a.sum())
            L = np.log(L_allcells).sum(axis=1).reshape(self._C.shape) + np.log(self.p_x) + np.log(self.p_c)

            A = np.amax(L)

            log_denom = A + np.log(np.exp(L.ravel()-A).sum())

            P_XC[t,:,:] = np.exp(L-log_denom)

        return P_XC

    def single_cell_decoding(self,p_xcy):
        p_xc = p_xcy.reshape([self._X.shape])*self.p_x*self.p_c
        p_xc/=p_xc.ravel().sum()
        return p_xc.sum(axis=0), p_xc.sum(axis=0)



class EncodingModel:
    def __init__(self,ops={}):
        self._set_ops(ops)
        self._set_ctrl_pts()
        s= self.ops['s']
        self.S = np.array([[-s, 2-s, s-2, s],
                    [2*s, s-3, 3-2*s, -s],
                    [-s, 0, s, 0,],
                    [0, 1, 0, 0]])
        print(self.pos_ctrl_pts.shape)
        #self.coefs = np.zeros([self._n_coefs,])

    def _set_ops(self,ops_in):

        ops_out={'key':0,
        'n_ctrl_pts_pos':5,
        'n_ctrl_pts_morph':3,
        'n_ctrl_pts_activity':3,
        'max_pos':450,
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
        morph_ctrl_pts = np.linspace(0,1,num=self.ops['n_ctrl_pts_morph'])
        dmorph = morph_ctrl_pts[1]-morph_ctrl_pts[0]
        self.morph_ctrl_pts[0]=morph_ctrl_pts[0]-dmorph
        self.morph_ctrl_pts[1:-1]=morph_ctrl_pts
        self.morph_ctrl_pts[-1]=morph_ctrl_pts[-1]+dmorph

        self.activity_ctrl_pts = np.zeros([self.ops['n_ctrl_pts_activity']+2,])
        activity_ctrl_pts = np.linspace(0,100,num=self.ops['n_ctrl_pts_activity'])
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

        # spl_basis = self.pos_morph_spline(pos,morph)
        #
        # X = np.zeros([spl_basis.shape[0],spl_basis.shape[1]+1])
        # X[:,:-1]=spl_basis
        # # may need to take out intercept
        # X[:,-1]=.5
        #
        # return X

    def fit_linear(self,X,y):
        if self.ops['RidgeCV']:
            # mdl = LinearRegression(fit_intercept=False)
            mdl = RidgeCV(fit_intercept=False)
            mdl.fit(X,y)

        else:
            mdl = LinearRegression(fit_intercept=False)
            mdl.fit(X,y)

        self.coef_ = mdl.coef_
        self.mdl_ = mdl
        # self.alpha_ = mdl.alpha_

    def predict_linear(self,X):
        return self.mdl_.predict(X)
