import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.linear_model import RidgeCV
import os

os.sys.path.append("C:\\Users\\mplitt\\MightyMorphingPhotonRangers")
import utilities as u

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

        ops_out={'key':0,
        'n_ctrl_pts_pos':10,
        'n_ctrl_pts_morph':5,
        'max_pos':450,
        'RidgeCV':True,
        's':.5}
        for k,v in ops_in.items():
            ops_out[k]=v

        self.ops = ops_out
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
        #print(ctrl_i,np.matmul(u,self.S).shape)
        x[ctrl_i-1:ctrl_i+3] = np.matmul(u,self.S)
        return x


    def make_design_matrix(self,pos,morph):
        spl_basis = self.pos_morph_spline(pos,morph)

        X = np.zeros([spl_basis.shape[0],spl_basis.shape[1]+1])
        X[:,:-1]=spl_basis
        X[:,-1]=1.

        return X

    def fit_linear(self,X,y):
        if self.ops['RidgeCV']:
            mdl = RidgeCV(fit_intercept=False)
            mdl.fit(X,y)
        else:
            pass

        self.coef_ = mdl.coef_
        self.alpha_ = mdl.alpha_

    def predict_linear(self,X):
        print(self.coef_.shape,X.shape)
        return np.matmul(X,self.coef_.T)

    def fit_poisson(self,X,y,alpha=.1):
        coefs0 = np.random.rand([X.shape[1],1])
        res = sp.optimize.fmin_ncg(self._f_poisson,coefs0,self._grad_poisson,self._hessian_poisson,args=(X,y,alpha))
        self.coef_ = res[0]
        self.alpha_=alpha

    def _f_poisson(coefs,X,y,alpha):
        u = np.matmul(X,coefs)
        rate = np.exp(u)

        f_l2 = .5*alpha*np.linalg.norm(coefs.ravel(),2)
        return (rate-np.multiply(y,np.log(rate))).sum()  + f_l2

    def _grad_poisson(coefs,X,y,alpha):
        u = np.matmul(X,coefs)
        rate = np.exp(u)

        return np.matmul(X.T,rate-y) + alpha*coefs

    def _hessian_poisson(coefs,X,y,alpha):
        u = np.matmul(X,coefs)
        rate = np.exp(u)

        rX = rate*X
        return np.matmul(rX.T,X) + alpha*np.eye(coefs.size)
#

    def predict_poisson(self,X):
        return np.exp(np.matmul(X,self.coef_.T))
        
