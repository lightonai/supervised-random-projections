import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

from numba import jit
from lightonml.projections.sklearn import OPUMap


@jit(nopython=True)
def delta_matrix(y):
    # dumbest way to do it, but I have numba
    L = []
    for a in y:
        tmp = []
        for b in y:
            tmp.append(a == b)
        L.append(tmp)
    return np.array(L)


class RPEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, expansion_factor=10):
        self.expansion_factor = expansion_factor
        self.threshold = None
        self.mean = self.std = None

    def fit(self, X, y=None):
        self.mean = X.mean(0)
        self.std = X.std()
        self.proj = torch.nn.Linear(X.shape[1], self.expansion_factor*X.shape[1]).cuda()
        return self

    def transform(self, X):
        X = X - self.mean
        with torch.no_grad():
            Y = self.proj(torch.cuda.FloatTensor(X)).cpu().numpy()
        Z = np.logical_and(Y > 0, Y < self.std)
        return Z


def scatter_plot(X, y, ax=None, s=5, title=None, xlabel=None, ylabel=None, fontsize=12, axis_off=True):
    if ax is None:
        fig, ax = plt.subplots()
    for c in np.unique(y):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=s)
    if title:
        ax.set_title(title, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if axis_off:
        ax.set_axis_off()


class SPCA(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, eps=1e-3):
        super().__init__()
        self.k = k
        self.eps = eps
        self.U = None

    def fit(self, X, y):
        L = delta_matrix(y)
        Lc = L - L.mean(0)
        Q = X.T @ Lc @ X
        v, U = np.linalg.eigh(Q + self.eps*np.eye(len(Q)))
        self.U = U[:, ::-1]
        return self

    def transform(self, X):
        reduced = X @ self.U[:, :self.k]
        return reduced


class KSPCA(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, eps=1e-3, metric='rbf', gamma=1):
        super().__init__()
        self.k = k
        self.eps = eps
        self.metric = metric
        self.gamma = gamma
        self.orig_data, self.V = (None, None)

    def fit(self, X, y):
        self.train_data = X
        L = delta_matrix(y)
        K = pairwise_kernels(X, metric=self.metric, gamma=self.gamma)
        Lc = L - L.mean(0)
        Q = Lc @ K
        w, V = np.linalg.eig(Q + self.eps * np.eye(len(Q)))
        idx = np.argsort(w)
        V = V[:, idx].real
        self.V = V[:, ::-1]
        return self

    def transform(self, X):
        K = pairwise_kernels(X, self.train_data, metric=self.metric, gamma=self.gamma)
        reduced = K @ self.V[:, :self.k]
        return reduced


class SRP(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, y_gamma=100, opu=None, postproc=None):
        super().__init__()
        self.k = k
        self.opu = opu
        self.postproc = postproc
        if opu is None:
            self.y_features = RBFSampler(gamma=y_gamma, n_components=self.k)
        else:
            self.y_features = OPUMap(n_components=k, opu=opu)
        self.U = None

    def fit(self, X, y, center=True):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        psi = self.y_features.fit_transform(y).astype('float32')
        if self.postproc:
            psi = self.postproc(psi)
        if center:
            X = X - X.mean(0)
        self.U = psi.T @ X
        return self

    def transform(self, X):
        return X @ self.U.T


class KSRP(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, gamma=1, n_components=500, metric='rbf', y_gamma=100, opu=None,
                 x_opu=False, y_opu=False, x_postproc=None, y_postproc=None,
                 feature_encoder=None):
        super().__init__()
        self.k = k
        self.opu = opu
        self.gamma = gamma
        if opu is None:
            self.metric = metric
            self.x_features = RBFSampler(gamma=gamma, n_components=n_components)
            self.y_features = RBFSampler(gamma=y_gamma, n_components=self.k)
        else:
            if x_opu:
                self.x_features = make_pipeline(
                    feature_encoder,
                    OPUMap(n_components=n_components, opu=opu, verbose_level=0)
                )
            else:
                self.x_features = RBFSampler(gamma=gamma, n_components=n_components)
            if y_opu:
                # n_components should be k but since it's the same OPU object it would mess with x_features
                self.y_features = OPUMap(n_components=n_components, opu=opu, verbose_level=0)
            else:
                self.y_features = RBFSampler(gamma=y_gamma, n_components=self.k)

        self.x_postproc = x_postproc
        self.y_postproc = y_postproc
        self.U = None
        self.psi = None
        self.train_data = None

    def fit(self, X, y, center=True):
        self.train_data = X
        phi = self.x_features.fit_transform(X).astype('float32')
        if self.x_postproc:
            phi = self.x_postproc(phi)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        psi = self.y_features.fit_transform(y).astype('float32')
        if psi.shape[1] > self.k:
            psi = psi[:, :self.k]
        if self.y_postproc:
            psi = self.y_postproc(psi)

        if center:
            phi = phi - phi.mean(0)
        self.psi = psi
        self.U = psi.T @ phi
        return self

    def transform(self, X, exact=False):
        # KSRP can be KSPCA with only the SPCA part approximated by SRP
        if exact:
            K = pairwise_kernels(X, self.train_data, metric=self.metric, gamma=self.gamma)
            return K @ self.psi
        # or both steps can be approximated with random features
        else:
            phi = self.x_features.transform(X).astype('float32')
            if self.x_postproc:
                phi = self.x_postproc(phi)
            return phi @ self.U.T
