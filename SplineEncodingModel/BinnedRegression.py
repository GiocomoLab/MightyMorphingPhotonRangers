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


    def _set_ops(self,ops_in):

        ops_out={'key':0,
        'n_pts_pos':10,
        'n_pts_morph':5,
        'max_pos':450,
        'RidgeCV':True}
        for k,v in ops_in.items():
            ops_out[k]=v

        self.ops = ops_out
        self._n_coefs = (self.ops['n_ctrl_pts_pos']+2)*(self.ops['n_ctrl_pts_morph']+2)

    def _1d_bin(self,vec,nbins):
        edges = np.linspace(0,np.amax(vec),num=nbins+1):



    def make_design_matrix(self,pos,morph):
        spl_basis = self.pos_morph_spline(pos,morph)

        X = np.zeros([spl_basis.shape[0],spl_basis.shape[1]+1])
        X[:,:-1]=spl_basis
        X[:,-1]=.5

        return X

    def fit_linear(self,X,y):
        if self.ops['RidgeCV']:
            # mdl = LinearRegression(fit_intercept=False)
            mdl = RidgeCV(fit_intercept=False)
            mdl.fit(X,y)

        else:
            pass

        self.coef_ = mdl.coef_
        self.alpha_ = mdl.alpha_

    def predict_linear(self,X):
        #print(self.coef_.shape,X.shape)
        return np.matmul(X,self.coef_.T)

    def fit_poisson(self,X,y,alpha=.1):
        coefs0 = np.random.rand(X.shape[1],1)
        coef_opt= sp.optimize.fmin_ncg(_f_poisson,coefs0,_grad_poisson,fhess=_hessian_poisson,args=(X,y,alpha))
        self.coef_ = coef_opt.reshape([-1,1])

        self.alpha_=alpha

    def predict_poisson(self,X):
        return np.exp(np.matmul(X,self.coef_))


def _f_poisson(coefs,X,y,alpha,beta):
    u = np.matmul(X,coefs)
    rate = np.exp(u)

    f_l2 = .5*alpha*np.linalg.norm(coefs[:-1].ravel(),2)
    f_ddcoef = .5*beta
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
