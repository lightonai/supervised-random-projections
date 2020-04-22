import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def add_noise_columns(df, n_cols=8, sigma=2):
    for i in range(n_cols):
        df['noise_{}'.format(i)] = np.random.randn(len(df)) * sigma
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def get_xor(n_samples=500, scale=10, sigma=1):
    centers = np.array([(0, 0), (0, 1), (1, 0), (1, 1)]) * scale
    centers = centers - scale/2

    X, y = make_blobs(n_samples, n_features=2, centers=centers, cluster_std=sigma)
    mask = np.logical_or(y==0,y==3)
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'label': mask.astype(np.uint8)
    })
    df = add_noise_columns(df)

    X = df.drop('label', axis=1).values
    y = df.label.values
    return X, y


def get_spirals(n_samples=500, noise_level=0.5, max_dist=20):
    t = np.linspace(0, max_dist, n_samples/2)
    noise = np.random.randn(len(t)) * noise_level
    x = np.cos(t) * t + noise
    y = np.sin(t) * t + noise
    df = pd.DataFrame({
        'x': np.hstack((x, -x)),
        'y': np.hstack((y, -y)),
        'label': [0]*len(t) + [1]*len(t)
    })
    df = add_noise_columns(df)

    X = df.drop('label', axis=1).values
    y = df.label.values
    return X, y


def get_uci(dataset='sonar'):

    if dataset == 'sonar':
        filename = 'sonar.all-data'
        positive = 'M'
    elif dataset == 'ionosphere':
        filename = 'ionosphere.data'
        positive = 'g'
    else:
        raise ValueError('Dataset must be sonar or ionosphere')

    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    X, y = [], []
    for line in lines:
        line = line.split(',')
        row = [float(i) for i in line[:-1]]
        label = line[-1]
        X.append(row)
        y.append(label == positive)

    X = np.array(X)
    y = np.array(y)
    return X, y
