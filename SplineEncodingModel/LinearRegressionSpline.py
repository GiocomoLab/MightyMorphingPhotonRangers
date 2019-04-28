import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.linear_model import RidgeCV
import os

os.sys.path.append("C:\\Users\\mplitt\\MightyMorphingPhotonRangers")
import utilities as u

class EncodingModel:
    def __init__(self,ops={}}):
        self._set_ops(ops)
        self._set_ctrl_pts()
        s= self.ops['s']
        self.S = np.array([[-s, 2-s, s-2, s],
                    [2*s, s-3, 3-2*s, -s],
                    [-s, 0, s, 0,],
                    [0 1 0 0]])
        self.coefs = np.zeros([n_coefs,])

    def _set_ops(ops_in):

        ops_out={'key':0,
        'n_ctrl_pts_pos':10,
        'n_ctrl_pts_morph':5,
        'max_pos':450,
        'RidgeCV':True}
        for k,v in ops_in.items():
            ops_out[k]=v

        self.ops = ops_out
        self._n_coefs = (self.ops['n_ctrl_pts_pos']+2)*(self.ops['n_ctrl_pts_morph']+2)

    def _set_ctrl_pts():
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


    def pos_morph_spline(pos,morph):
        assert pos.shape==morph.shape, "position and morph vectors need to be of same length"

        splbasis= np.zeros([pos.shape[0],self._n_coefs])
        for i in range(pos.shape[0]):
            p,m = pos[i],morph[i]
            x_p = self._1d_spline_coeffs(self.pos_ctrl_pts,p)
            x_m = self._1d_spline_coeffs(self.morph_ctrl_pts,m)

            x_pm = np.matmul(x_p.reshape([-1,1]),x_m.reshape([1,-1]))
            splbasis[i,:]=x_pm.ravel()

        return splbasis

    def _1d_spline_coeffs(ctrl,v):

        x = np.zeros(ctrl.shape)

        # nearest ctrl pt
        ctrl_i = (ctrl<v).sum()-1
        pre_ctrl_pt = ctrl[ctrl_i]

        # next ctrl pt
        post_ctrl_pt = ctrl[ctrl_i+1]

        alpha = (v-pre_ctrl_pt)/(post_ctrl_pt-pre_ctrl_pt)
        u = np.array([alpha**3, alpha**2, alpha, 1]).reshape([1,-1])
        # p =
        x[ctrl_i-1:ctrl_i+2] = np.matmul(u,self.S)
        return x


    def make_design_matrix(pos,morph):
        spl_basis = self.pos_morph_spline(pos,morph)

        X = np.zeros([spl_basis.shape[0],spl_basis.shape[1]+1])
        X[:,:-1]=spl_basis
        X[:,-1]=1.

        return X

    def fit_linear(X,y):
        if self.ops['RidgeCV']:
            mdl = RidgeCV(fit_intercept=False)
            mdl.fit(X,y)
        else:
            pass

        self.coef_ = mdl.coef_
        self.lambda_ = mdl.alpha_

    def predict_linear(X):
        return np.matmul(X,self.coef_)

    def fit_poisson(X,y):


        pass

    def predict_poisson(X):
        pass
