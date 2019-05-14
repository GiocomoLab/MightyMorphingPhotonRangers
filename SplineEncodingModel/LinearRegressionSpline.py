import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neighbors import KernelDensity
import os
import copy
os.sys.path.append("C:\\Users\\mplitt\\MightyMorphingPhotonRangers")
import utilities as u


class NBDecodingModel:
    def __init__(self,ops={}):
        self._x = np.linspace(0.001,449.999,num=50)
        self._c = np.linspace(0.001,.999,num=50)
        self._X = np.matmul(self._x.reshape([-1,1]),np.ones([1,50]))
        self._C = np.matmul(np.ones([50,1]),self._c.reshape([1,50]))
        #np.meshgrid(self._x,self._c)
        self._set_ops(ops)
        self._cells = {}
        self.Lam=None


        dummyEM = EncodingModel(ops = ops)
        decodingDM = dummyEM.make_design_matrix(self._X.ravel(),self._C.ravel())
        self.decodingDM = decodingDM



    def _set_ops(self,ops_in):
        ops_out = {}
        for k,v in ops_in.items():
            ops_out[k]=v

        self.ops = ops_out

    def poisson_fit(self,pos,morph,S):
        self._fit_priors(pos,morph)
        self.n_cells = S.shape[1]
        splmdl = EncodingModel(ops=self.ops)
        X = splmdl.make_design_matrix(pos,morph)
        for c in range(S.shape[1]):
            self._cells[c] = EncodingModel(ops=self.ops)
            self._cells[c].fit_poisson(X,S[:,c])

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


def _gamma_pdf(y,lam):
    y = np.maximum(y,1E-3)
    return np.power(y,lam-1)*np.exp(-y)/sp.special.gamma(lam)

class EncodingModel:
    def __init__(self,ops={}):
        self._set_ops(ops)
        self._set_ctrl_pts()
        s= self.ops['s']
        self.S = np.array([[-s, 2-s, s-2, s],
                    [2*s, s-3, 3-2*s, -s],
                    [-s, 0, s, 0,],
                    [0, 1, 0, 0]])

        #self.coefs = np.zeros([self._n_coefs,])

    def _set_ops(self,ops_in):

        ops_out={'n_ctrl_pts_pos':3,
        'n_ctrl_pts_morph':3,
        'max_pos':450,
        'RidgeCV':True,
        's':.5}
        for k,v in ops_in.items():
            #print(k,v)
            ops_out[k]=v

        self.ops = ops_out
        #print(self.ops)
        self._n_coefs = (self.ops['n_ctrl_pts_pos']+2)*(self.ops['n_ctrl_pts_morph']+2)

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


    def pos_morph_spline(self,pos,morph):
        assert pos.shape==morph.shape, "position and morph vectors need to be of same length"

        splbasis= np.zeros([pos.shape[0],self._n_coefs])
        for i in range(pos.shape[0]):
            p,m = pos[i],morph[i]
            # print(p,m)
            x_p = self._1d_spline_coeffs(self.pos_ctrl_pts,p)
            x_m = self._1d_spline_coeffs(self.morph_ctrl_pts,m)
            #print(x_p.shape,x_m.shape)
            x_pm = np.matmul(x_p.reshape([-1,1]),x_m.reshape([1,-1]))
            splbasis[i,:]=x_pm.ravel()

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


    def make_design_matrix(self,pos,morph):
        spl_basis = self.pos_morph_spline(pos,morph)
        return spl_basis
        # X = np.zeros([spl_basis.shape[0],spl_basis.shape[1]+1])
        # X[:,:-1]=spl_basis
        # X[:,-1]=.5

        # return X

    def fit_linear(self,X,y):
        if self.ops['RidgeCV']:
            # mdl = LinearRegression(fit_intercept=False)
            mdl = RidgeCV(fit_intercept=False)
            mdl.fit(X,y)

        else:
            pass

        self.coef_ = mdl.coef_
        # self.alpha_ = mdl.alpha_

    def predict_linear(self,X):
        #print(self.coef_.shape,X.shape)
        return np.matmul(X,self.coef_.T)

    def fit_poisson(self,X,y,alpha=.001):
        coefs0 = np.random.rand(X.shape[1],1)
        coef_opt= sp.optimize.fmin_ncg(_f_poisson,coefs0,_grad_poisson,fhess=_hessian_poisson,args=(X,y,alpha),disp=0)
        self.coef_ = coef_opt.reshape([-1,1])

        self.alpha_=alpha

    def predict_poisson(self,X):
        return np.exp(np.matmul(X,self.coef_))

    def prob_poisson(self,y,X):
        '''assume y is a scalar'''
        return np.array([sp.stats.gamma.pdf(y,lam) for lam in self.predict_poisson(X).tolist()])





def _f_poisson(coefs,X,y,alpha):
    u = np.matmul(X,coefs)
    rate = np.exp(u)

    f_l2 = .5*alpha*np.linalg.norm(coefs[:-1].ravel(),2)
    return (rate-np.multiply(y,np.log(rate))).sum()  + f_l2

def _grad_poisson(coefs,X,y,alpha):
    u = np.matmul(X,coefs)
    rate = np.exp(u)

    return np.matmul(X.T,rate-y) + alpha*coefs

def _hessian_poisson(coefs,X,y,alpha):
    u = np.matmul(X,coefs)
    rate = np.exp(u)

    rX = rate[:,np.newaxis]*X
    return np.matmul(rX.T,X) + alpha*np.eye(coefs.size)
#
