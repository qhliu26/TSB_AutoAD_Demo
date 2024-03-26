import numpy as np
import math
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.vus.metrics import get_metrics
from TSB_UAD.utils.slidingWindows import find_length_rank
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from functools import wraps
import time
import os
import logging

# import sys
# sys.path.append('..')
from TSB_UAD.models.IForest import IForest
from TSB_UAD.models.PCA import PCA
# from TSB_UAD.models.LOF import LOF
# from TSB_UAD.models.MatrixProfile import MatrixProfile
# from TSB_UAD.models.POLY import POLY
# from TSB_UAD.models.NormA import NORMA
# from TSB_UAD.models.OCSVM import OCSVM
# from TSB_UAD.models.HBOS import HBOS
# from TSB_UAD.models.AE import AutoEncoder
# from TSB_UAD.models.LSTM import LSTM
# from TSB_UAD.models.CNN import CNN

def run_iforest_dev(data, periodicity, n_estimators):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_pca_dev(data, periodicity, n_components):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow=slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score