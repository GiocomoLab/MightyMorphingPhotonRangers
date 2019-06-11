import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools

avec = np.linspace(0,1,num=20)
C = np.array([(x,y) for x,y in itertools.product(avec.tolist(),avec.tolist())])
dispersion = .01
beta = np.ones([C.shape[0],])
rbf = lambda x : np.exp(-np.dot(x-C,x-C)/dispersion)
K = lambda X: np.array(map(rbf,X.tolist()))
mu = lambda X,b: np.exp(np.dot(K(X),b))

S = np.random.randn([1000,])

# def llh(S,X,b):
#     return np.exp(S*np.log(mu(X,b)) - mu(X,b))

def objective(S,X,b,alpha=.1):
    return -(S*mu(X,b) - np.exp(mu(X,b))).sum() + alpha*np.linalg.norm(b,ord=2)

def gradient(S,X,b,alpha=.1):
    # return gradient vector
    return -(S-np.exp(mu(X,b)))*K(X) +2*alpha*b


def hessian(S,X,b):
    H = np.zeros([b.size,b.size])
    D = 0
    for i in range(S.shape[0]):
        u = np.exp(mu(X[i,:],b))*K(X[i,;])
        v = np.copy(K(X[:,i]))
    # return hessian matrix
    pass



xx  = np.linspace(0,1,num=1000)
f,ax = plt.subplots()
for a in avec.tolist():
    z = [rbf(x,a) for x in xx.tolist()]
    ax.plot(z)

  
