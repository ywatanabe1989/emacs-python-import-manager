;;; -*- coding: utf-8; lexical-binding: t -*-
;;; Author: ywatanabe
;;; Timestamp: <2025-04-29 12:25:44>
;;; File: /home/ywatanabe/.emacs.d/lisp/python-import-manager/predefined-packages/pim-imports-data-science.el

;;; Copyright (C) 2025 Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)


(defvar pim-imports-data-science
  '(
    ;; Data Science Core
    ("np" . "import numpy as np")
    ("pd" . "import pandas as pd")
    ("xr" . "import xarray as xr")
    ("scipy" . "import scipy")
    ("stats" . "from scipy import stats")
    ("dask" . "import dask.dataframe as dd")
    ("vaex" . "import vaex")
    ("polars" . "import polars as pl")
    ("cudf" . "import cudf")
    ("pa" . "import pyarrow as pa")

    ;; Imbalanced Learning
    ("RandomUnderSampler"
     . "from imblearn.under_sampling import RandomUnderSampler")
    ("NearMiss" . "from imblearn.under_sampling import NearMiss")
    ("TomekLinks" . "from imblearn.under_sampling import TomekLinks")
    ("ClusterCentroids"
     . "from imblearn.under_sampling import ClusterCentroids")
    ("SMOTE" . "from imblearn.over_sampling import SMOTE")
    ("ADASYN" . "from imblearn.over_sampling import ADASYN")
    ("SMOTETomek" . "from imblearn.combine import SMOTETomek")
    ("SMOTEENN" . "from imblearn.combine import SMOTEENN")

    ;; PyTorch Ecosystem
    ("torch" . "import torch")
    ("nn" . "import torch.nn as nn")
    ("F" . "import torch.nn.functional as F")
    ("Dataset" . "from torch.utils.data import Dataset")
    ("DataLoader" . "from torch.utils.data import DataLoader")
    ("optim" . "import torch.optim as optim")
    ("transforms" . "import torchvision.transforms as transforms")
    ("models" . "import torchvision.models as models")
    ("summary" . "from torchsummary import summary")
    ("DDP" . "from torch.nn.parallel import DistributedDataParallel")
    ("distributed" . "import torch.distributed as dist")
    ("amp" . "from torch.cuda.amp import autocast, GradScaler")
    ("TensorBoard"
     . "from torch.utils.tensorboard import SummaryWriter")
    ("torchmetrics" . "import torchmetrics")
    ("timm" . "import timm")
    ("einops" . "import einops")
    ("kornia" . "import kornia")
    ("catalyst" . "import catalyst")
    ("ignite" . "import ignite")
    ("ViT" . "from pytorch_pretrained_vit import ViT")

    ;; Scikit-learn - Linear Models
    ("LinearRegression"
     . "from sklearn.linear_model import LinearRegression")
    ("LogisticRegression"
     . "from sklearn.linear_model import LogisticRegression")
    ("Ridge" . "from sklearn.linear_model import Ridge")
    ("Lasso" . "from sklearn.linear_model import Lasso")
    ("ElasticNet" . "from sklearn.linear_model import ElasticNet")

    ;; Scikit-learn - Ensemble Methods
    ("RandomForestClassifier"
     . "from sklearn.ensemble import RandomForestClassifier")
    ("GradientBoostingClassifier"
     . "from sklearn.ensemble import GradientBoostingClassifier")
    ("IsolationForest"
     . "from sklearn.ensemble import IsolationForest")

    ;; Scikit-learn - Preprocessing
    ("StandardScaler"
     . "from sklearn.preprocessing import StandardScaler")
    ("MinMaxScaler" . "from sklearn.preprocessing import MinMaxScaler")
    ("RobustScaler" . "from sklearn.preprocessing import RobustScaler")
    ("LabelEncoder" . "from sklearn.preprocessing import LabelEncoder")
    ("OneHotEncoder"
     . "from sklearn.preprocessing import OneHotEncoder")
    ("PolynomialFeatures"
     . "from sklearn.preprocessing import PolynomialFeatures")
    ("FunctionTransformer"
     . "from sklearn.preprocessing import FunctionTransformer")

    ;; Scikit-learn - Feature Selection
    ("SelectKBest"
     . "from sklearn.feature_selection import SelectKBest")
    ("mutual_info_classif"
     . "from sklearn.feature_selection import mutual_info_classif")
    ("f_classif" . "from sklearn.feature_selection import f_classif")

    ;; Scikit-learn - Dimensionality Reduction
    ("PCA" . "from sklearn.decomposition import PCA")
    ("TSNE" . "from sklearn.manifold import TSNE")
    ("UMAP" . "from umap import UMAP")

    ;; Scikit-learn - Clustering
    ("KMeans" . "from sklearn.cluster import KMeans")
    ("DBSCAN" . "from sklearn.cluster import DBSCAN")

    ;; Scikit-learn - Other Algorithms
    ("SVC" . "from sklearn.svm import SVC")
    ("KNeighborsClassifier"
     . "from sklearn.neighbors import KNeighborsClassifier")
    ("DecisionTreeClassifier"
     . "from sklearn.tree import DecisionTreeClassifier")
    ("LocalOutlierFactor"
     . "from sklearn.neighbors import LocalOutlierFactor")

    ;; Scikit-learn - Pipeline
    ("make_pipeline" . "from sklearn.pipeline import make_pipeline")
    ("Pipeline" . "from sklearn.pipeline import Pipeline")
    ("make_column_transformer"
     . "from sklearn.compose import make_column_transformer")
    ("ColumnTransformer"
     . "from sklearn.compose import ColumnTransformer")

    ;; Scikit-learn - Model Selection
    ("GridSearchCV"
     . "from sklearn.model_selection import GridSearchCV")
    ("RandomizedSearchCV"
     . "from sklearn.model_selection import RandomizedSearchCV")
    ("cross_val_score"
     . "from sklearn.model_selection import cross_val_score")
    ("train_test_split"
     . "from sklearn.model_selection import train_test_split")
    ("KFold" . "from sklearn.model_selection import KFold")
    ("StratifiedKFold"
     . "from sklearn.model_selection import StratifiedKFold")
    ("TimeSeriesSplit"
     . "from sklearn.model_selection import TimeSeriesSplit")
    ("GroupKFold" . "from sklearn.model_selection import GroupKFold")
    ("learning_curve"
     . "from sklearn.model_selection import learning_curve")
    ("validation_curve"
     . "from sklearn.model_selection import validation_curve")
    ("permutation_test_score"
     . "from sklearn.model_selection import permutation_test_score")

    ;; Regression Metrics
    ("r2_score" . "from sklearn.metrics import r2_score")
    ("mean_absolute_error"
     . "from sklearn.metrics import mean_absolute_error")
    ("mean_squared_error"
     . "from sklearn.metrics import mean_squared_error")
    ("mean_squared_log_error"
     . "from sklearn.metrics import mean_squared_log_error")
    ("median_absolute_error"
     . "from sklearn.metrics import median_absolute_error")
    ("explained_variance_score"
     . "from sklearn.metrics import explained_variance_score")
    ("max_error" . "from sklearn.metrics import max_error")
    ("mean_absolute_percentage_error"
     . "from sklearn.metrics import mean_absolute_percentage_error")
    ("mean_poisson_deviance"
     . "from sklearn.metrics import mean_poisson_deviance")
    ("mean_gamma_deviance"
     . "from sklearn.metrics import mean_gamma_deviance")
    ("mean_tweedie_deviance"
     . "from sklearn.metrics import mean_tweedie_deviance")
    ("d2_absolute_error_score"
     . "from sklearn.metrics import d2_absolute_error_score")
    ("d2_pinball_score"
     . "from sklearn.metrics import d2_pinball_score")
    ("d2_tweedie_score"
     . "from sklearn.metrics import d2_tweedie_score")

    ;; Classification Metrics
    ("accuracy_score" . "from sklearn.metrics import accuracy_score")
    ("precision_score" . "from sklearn.metrics import precision_score")
    ("recall_score" . "from sklearn.metrics import recall_score")
    ("f1_score" . "from sklearn.metrics import f1_score")
    ("matthews_corrcoef"
     . "from sklearn.metrics import matthews_corrcoef")
    ("hamming_loss" . "from sklearn.metrics import hamming_loss")
    ("jaccard_score" . "from sklearn.metrics import jaccard_score")
    ("log_loss" . "from sklearn.metrics import log_loss")
    ("zero_one_loss" . "from sklearn.metrics import zero_one_loss")
    ("brier_score_loss"
     . "from sklearn.metrics import brier_score_loss")

    ;; ROC and PR Curves
    ("roc_curve" . "from sklearn.metrics import roc_curve")
    ("auc" . "from sklearn.metrics import auc")
    ("average_precision_score"
     . "from sklearn.metrics import average_precision_score")
    ("det_curve" . "from sklearn.metrics import det_curve")

    ;; Multilabel Metrics
    ("multilabel_confusion_matrix"
     . "from sklearn.metrics import multilabel_confusion_matrix")
    ("coverage_error" . "from sklearn.metrics import coverage_error")
    ("label_ranking_average_precision_score"
     . "from sklearn.metrics import label_ranking_average_precision_score")
    ("label_ranking_loss"
     . "from sklearn.metrics import label_ranking_loss")

    ;; Scikit-learn - Utils
    ("bootstrap" . "from sklearn.utils import bootstrap")
    ("resample" . "from sklearn.utils import resample")
    ("shuffle" . "from sklearn.utils import shuffle")

    ;; Scikit-learn - Base
    ("BaseEstimator" . "from sklearn.base import BaseEstimator")
    ("TransformerMixin" . "from sklearn.base import TransformerMixin")

    ;; Scikit-learn - Imputation
    ("SimpleImputer" . "from sklearn.impute import SimpleImputer")
    ("KNNImputer" . "from sklearn.impute import KNNImputer")
    ("IterativeImputer"
     . "from sklearn.impute import IterativeImputer")

    ;; Scikit-learn - Cross-validation Variations
    ("LeaveOneOut" . "from sklearn.model_selection import LeaveOneOut")
    ("LeavePOut" . "from sklearn.model_selection import LeavePOut")
    ("RepeatedKFold"
     . "from sklearn.model_selection import RepeatedKFold")
    ("RepeatedStratifiedKFold"
     . "from sklearn.model_selection import RepeatedStratifiedKFold")
    ("ShuffleSplit"
     . "from sklearn.model_selection import ShuffleSplit")
    ("StratifiedShuffleSplit"
     . "from sklearn.model_selection import StratifiedShuffleSplit")
    ("GroupShuffleSplit"
     . "from sklearn.model_selection import GroupShuffleSplit")
    ("LeaveOneGroupOut"
     . "from sklearn.model_selection import LeaveOneGroupOut")
    ("LeavePGroupsOut"
     . "from sklearn.model_selection import LeavePGroupsOut")
    ("PredefinedSplit"
     . "from sklearn.model_selection import PredefinedSplit")

    ;; Transformers & NLP
    ("AutoTokenizer" . "from transformers import AutoTokenizer")
    ("AutoModel" . "from transformers import AutoModel")
    ("AutoModelForSequenceClassification"
     . "from transformers import AutoModelForSequenceClassification")
    ("pipeline" . "from transformers import pipeline")
    ("BertModel" . "from transformers import BertModel")
    ("BertTokenizer" . "from transformers import BertTokenizer")
    ("GPT2Model" . "from transformers import GPT2Model")
    ("GPT2Tokenizer" . "from transformers import GPT2Tokenizer")

    ;; Deep Learning Extensions
    ("Trainer" . "from transformers import Trainer")
    ("TrainingArguments"
     . "from transformers import TrainingArguments")
    ("DataCollatorWithPadding"
     . "from transformers import DataCollatorWithPadding")
    ("EarlyStopping"
     . "from pytorch_lightning.callbacks import EarlyStopping")
    ("ModelCheckpoint"
     . "from pytorch_lightning.callbacks import ModelCheckpoint")

    ;; Machine Learning Extensions
    ("LGBMClassifier" . "from lightgbm import LGBMClassifier")
    ("LGBMRegressor" . "from lightgbm import LGBMRegressor")
    ("XGBClassifier" . "from xgboost import XGBClassifier")
    ("XGBRegressor" . "from xgboost import XGBRegressor")
    ("CatBoostClassifier" . "from catboost import CatBoostClassifier")
    ("CatBoostRegressor" . "from catboost import CatBoostRegressor")

    ;; Advanced Visualization
    ("venn2" "from matplotlib_venn import venn2")
    ("go" . "import plotly.graph_objects as go")
    ("ff" . "import plotly.figure_factory as ff")
    ("make_subplots" . "from plotly.subplots import make_subplots")
    ("colorcet" . "import colorcet as cc")

    ;; Vision Transformers
    ("ViTFeatureExtractor"
     . "from transformers import ViTFeatureExtractor")
    ("ViTImageProcessor"
     . "from transformers import ViTImageProcessor")
    ("ViTModel" . "from transformers import ViTModel")

    ;; Deep Learning & AI
    ("tf" . "import tensorflow as tf")
    ("keras" . "from tensorflow import keras")
    ("transformers" . "import transformers")
    ("datasets" . "from datasets import load_dataset")
    ("accelerate" . "import accelerate")
    ("wandb" . "import wandb")
    ("lightning" . "import pytorch_lightning as pl")

    ;; ML Libraries
    ("lgb" . "import lightgbm as lgb")
    ("xgb" . "import xgboost as xgb")
    ("optuna" . "import optuna")
    ("shap" . "import shap")

    ;; Data Processing
    ("cv2" . "import cv2")
    ("albumentations" . "import albumentations as A")
    ("tqdm" . "from tqdm import tqdm")
    ("json" . "import json")
    ("yaml" . "import yaml")
    ("pickle" . "import pickle")
    ("joblib" . "import joblib")

    ;; Scientific Computing
    ("spacy" . "import spacy")
    ("nltk" . "import nltk")
    ("scipy.signal" . "from scipy import signal")
    ("scipy.optimize" . "from scipy import optimize")

    ;; Database
    ("sqlite3" . "import sqlite3")
    ("sqlalchemy" . "import sqlalchemy")
    ("psycopg2" . "import psycopg2")
    ("mysql" . "import mysql.connector")
    ("mongodb" . "from pymongo import MongoClient")

    ;; Visualization
    ("matplotlib" . "import matplotlib")
    ("plt" . "import matplotlib.pyplot as plt")
    ("DateFormatter" . "from matplotlib.dates import DateFormatter")
    ("Axes3D" . "from mpl_toolkits.mplot3d import Axes3D")
    ("mdates" . "import matplotlib.dates as mdates")
    ("sns" . "import seaborn as sns")
    ("px" . "import plotly.express as px")
    ("bokeh" . "from bokeh.plotting import figure")
    ("hv" . "import holoviews as hv")
    ("pn" . "import panel as pn")
    ("ray" . "import ray")
    ("nx" . "import networkx as nx"))

  "Data Science related Python imports.")


(provide 'pim-imports-data-science)

(when
    (not load-file-name)
  (message "pim-imports-data-science.el loaded."
           (file-name-nondirectory
            (or load-file-name buffer-file-name))))