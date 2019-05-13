import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.linear_model import RidgeCV, LinearRegression
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
        print(self.pos_ctrl_pts.shape)
        #self.coefs = np.zeros([self._n_coefs,])

    def _set_ops(self,ops_in):

        ops_out={'key':0,
        'n_ctrl_pts_pos':5,
        'n_ctrl_pts_morph':3,
        'n_ctrl_pts_activity':5
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


    def make_spline(self,pos,morph,C,cell):
        assert pos.shape==morph.shape, "position and morph vectors need to be of same length"
        assert pos.shape == C.shape[0], "position, morph and activity vectors need to be of the same length"

        splbasis= np.zeros([pos.shape[0],self._n_coefs])
        for i in range(pos.shape[0]):
            p,m,c = pos[i],morph[i],C[i,cell]
            # print(p,m)
            x_p = self._1d_spline_coeffs(self.pos_ctrl_pts,p)
            x_m = self._1d_spline_coeffs(self.morph_ctrl_pts,m)
            x_c = self._1d_spline_coeffs(self.activity_ctrl_pts,c)
            #print(x_p.shape,x_m.shape)
            # loop through activity control points
            x_pmc = np.kron(x_c.reshape([-1,1]),np.matmul(x_p.reshape([-1,1]),x_m.reshape([1,-1])))
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
        return self.make_spline(pos,morph,C,cell)

        spl_basis = self.pos_morph_spline(pos,morph)

        X = np.zeros([spl_basis.shape[0],spl_basis.shape[1]+1])
        X[:,:-1]=spl_basis
        # may need to take out intercept
        X[:,-1]=.5

        return X

    def fit_linear(self,X,y):
        if self.ops['RidgeCV']:
            # mdl = LinearRegression(fit_intercept=False)
            mdl = RidgeCV(fit_intercept=True)
            mdl.fit(X,y)

        else:
            mdl = LinearRegression(fit_intercept=False)
            mdl.fit(X,y)

        self.coef_ = mdl.coef_
        # self.alpha_ = mdl.alpha_
