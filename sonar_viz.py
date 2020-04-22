import matplotlib.pyplot as plt
import numpy as np

import sklearn.decomposition
import sklearn.model_selection

from lightonopu import OPU

import load_data
import dimreduc


gamma = 1
X, y = load_data.get_uci('sonar')
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
# here the labels are already binary, so no encoding is required to use them as input to the OPU

opu = OPU()
opu.open()

pca = sklearn.decomposition.PCA(n_components=2)
kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
spca = dimreduc.SPCA(k=2)
kspca = dimreduc.KSPCA(k=2, gamma=gamma)
srp = dimreduc.SRP(k=2)
ksrp = dimreduc.KSRP(k=2, n_components=1000, gamma=gamma)
srp_opu = dimreduc.SRP(k=2, opu=opu, postproc=np.cos)
ksrp_opu = dimreduc.KSRP(k=2, opu=opu, n_components=1000, feature_encoder=dimreduc.RPEncoder(),
                         x_opu=True, y_opu=True, x_postproc=np.cos, y_postproc=np.cos)

res = {}
res['PCA'] = (pca.fit_transform(X_train, y_train), pca.transform(X_test))
res['KPCA'] = (kpca.fit_transform(X_train, y_train), kpca.transform(X_test))
res['SPCA'] = (spca.fit_transform(X_train, y_train), spca.transform(X_test))
res['KSPCA'] = (kspca.fit_transform(X_train, y_train), kspca.transform(X_test))
res['SRP'] = (srp.fit_transform(X_train, y_train), srp.transform(X_test))
res['KSRP'] = (ksrp.fit_transform(X_train, y_train), ksrp.transform(X_test))
res['SRP-OPU'] = (srp_opu.fit_transform(X_train, y_train), srp_opu.transform(X_test))
res['KSRP-OPU'] = (ksrp_opu.fit_transform(X_train, y_train), ksrp_opu.transform(X_test))


def single_plot(ax, method, title):
    mtrain = y_train == 0
    mtest = y_test == 0
    reduced_train, reduced_test = res[title]
    ax.scatter(reduced_train[mtrain, 0], reduced_train[mtrain, 1],
               color='C0', label='class 1 - train')
    ax.scatter(reduced_test[mtest, 0], reduced_test[mtest, 1],
               facecolor='none', edgecolor='C0', label='class 1 - test')
    ax.scatter(reduced_train[~mtrain, 0], reduced_train[~mtrain, 1],
               color='C1', label='class 2 - train')
    ax.scatter(reduced_test[~mtest, 0], reduced_test[~mtest, 1],
               facecolor='none', edgecolor='C1', label='class 2 - test')
    ax.set_title(title)
    ax.set_axis_off()


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
single_plot(axes[0, 0], res, 'SPCA')
single_plot(axes[1, 0], res, 'KSPCA')
single_plot(axes[0, 1], res, 'SRP')
single_plot(axes[1, 1], res, 'KSRP')
single_plot(axes[0, 2], res, 'SRP-OPU')
single_plot(axes[1, 2], res, 'KSRP-OPU')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.62, 0.45), fontsize=12)
fig.tight_layout()
plt.show()
