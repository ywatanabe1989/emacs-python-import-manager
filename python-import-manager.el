;;; -*- lexical-binding: t -*-
;;; Author: ywatanabe
;;; Time-stamp: <2024-11-03 03:46:06 (ywatanabe)>
;;; File: ./python-import-manager/python-import-manager.el


;; Author: Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)
;; Version: 1.0.0
;; Package-Requires: ((emacs "26.1"))
;; Keywords: languages, tools, python
;; URL: https://github.com/ywatanabe/python-import-manager

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <https://www.gnu.org/licenses/>.

;;; Commentary:

;; Python Import Manager (PIM) automatically manages Python imports in your code.
;; It can detect unused imports and add missing ones based on actual usage.

;;; Code:
(require 'async)
(require 'json)
(unless (fboundp 'json-read)
  (require 'json-mode))
(require 'seq)
(require 'python)
(require 'python-isort)

(defgroup python-import-manager nil
  "Management of Python imports."
  :group 'tools
  :prefix "pim-")

(define-minor-mode pim-auto-mode
  "Minor mode to automatically fix imports on save."
  :lighter " PIM"
  (if pim-auto-mode
      (add-hook 'before-save-hook #'pim-fix-imports nil t)
    (remove-hook 'before-save-hook #'pim-fix-imports t)))

(defvar pim--script-dir
  (file-name-directory (or load-file-name buffer-file-name))
  "Directory containing Python Import Manager scripts.")

(defcustom pim-python-path
  (or python-shell-interpreter
      (executable-find "python3"))
  "Path to python executable."
  :type 'string
  :group 'python-import-manager)

(defcustom pim-flake8-path
  (executable-find "flake8")
  "Path to flake8 executable."
  :type 'string
  :group 'python-import-manager)

(defcustom pim-flake8-args
  '("--max-line-length=100" "--select=F401,F821" "--isolated")
  "Arguments to pass to flake8."
  :type '(repeat string)
  :group 'python-import-manager)

(defcustom pim-isort-path
  (executable-find "isort")
  "Path to isort executable."
  :type 'string
  :group 'python-import-manager)

(defcustom pim-isort-args
  '("--profile" "black" "--line-length" "100")
  "Arguments to pass to isort."
  :type '(repeat string)
  :group 'python-import-manager)

(defcustom pim-import-list
  '(
    ("inspect" . "import inspect")
    ("re" . "import re")
    ("os" . "import os")
    ("glob" . "from glob import glob")
    ("json" . "import json")
    ("time" . "import time")
    ("datetime" . "from datetime import datetime")
    ("itertools" . "import itertools")
    ("combinations" . "from itertools import combinations")
    ("cycle" . "from itertools import cycle")
    ("product" . "from itertools import product")
    ("functools" . "import functools")
    ("partial" . "from functools import partial")
    ("math" . "import math")
    ("sys" . "import sys")
    ("random" . "import random")
    ("Path" . "from pathlib import Path")
    ("csv" . "import csv")
    ("pickle" . "import pickle")
    ("asyncio" . "import asyncio")
    ("Decimal" . "from decimal import Decimal")
    ("subprocess" . "import subprocess")
    ("importlib" . "import importlib")
    ("ic" . "from icecream import ic")
    ("pprint" . "from pprint import pprint")
    ("logging" . "import logging")
    ("warnings" . "import warnings")
    ("tqdm" . "from tqdm import tqdm")
    ("Tuple" . "from typing import Tuple")
    ("Dict" . "from typing import Dict")
    ("Any" . "from typing import Any")
    ("Union" . "from typing import Union")
    ("Sequence" . "from typing import Sequence")
    ("Literal" . "from typing import Literal")
    ("Optional" . "from typing import Optional")
    ("Iterable" . "from typing import Iterable")

    ;; Debugging
    ("ipdb" . "import ipdb")
    ("get_ipython" . "from IPython import get_ipython")

    ;; Collections
    ("defaultdict" . "from collections import defaultdict")
    ("Counter" . "from collections import Counter")
    ("deque" . "from collections import deque")
    ("namedtuple" . "from collections import namedtuple")
    ("OrderedDict" . "from collections import OrderedDict")

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

    ;; Visualization
    ("plt" . "import matplotlib.pyplot as plt")
    ("sns" . "import seaborn as sns")
    ("px" . "import plotly.express as px")
    ("bokeh" . "from bokeh.plotting import figure")
    ("hv" . "import holoviews as hv")
    ("pn" . "import panel as pn")
    ("Axes3D" . "from mpl_toolkits.mplot3d import Axes3D")

    ;; Machine Learning
    ("sklearn" . "import sklearn")
    ("train_test_split" . "from sklearn.model_selection import train_test_split")
    ("linear_model" . "from sklearn import linear_model")
    ("svm" . "from sklearn import svm")
    ("tree" . "from sklearn import tree")
    ("ensemble" . "from sklearn import ensemble")
    ("metrics" . "from sklearn import metrics")
    ("linearRegression" . "from sklearn.linear_model import LinearRegression")
    ("KMeans" . "from sklearn.cluster import KMeans")
    ("GridSearchCV" . "from sklearn.model_selection import GridSearchCV")
    ("PCA" . "from sklearn.decomposition import PCA")
    ("KNeighborsClassifier" . "from sklearn.neighbors import KNeighborsClassifier")
    ("DummyClassifier" . "from sklearn.dummy import DummyClassifier")
    ("SimpleImputer" . "from sklearn.impute import SimpleImputer")
    ("MinMaxScaler" . "from sklearn.preprocessing import MinMaxScaler")
    ("StandardScaler" . "from sklearn.preprocessing import StandardScaler")
    ("logistic_regression" . "from sklearn.linear_model import LogisticRegression")
    ("xgboost" . "import xgboost as xgb")
    ("lightgbm" . "import lightgbm as lgb")
    ("catboost" . "from catboost import CatBoostRegressor")

    ;; Deep Learning
    ("tf" . "import tensorflow as tf")
    ("torch" . "import torch")
    ("nn" . "import torch.nn as nn")
    ("F" . "import torch.nn.functional as F")
    ("keras" . "from keras.models import Sequential")
    ("keras_layers" . "from keras.layers import Dense")
    ("tensorflow_probability" . "import tensorflow_probability as tfp")
    ("PyTorchGeometric" . "from torch_geometric.data import Data")
    ("torchvision" . "import torchvision")
    ("fastai" . "from fastai.vision.all import *")
    ("pytorch_lightning" . "from pytorch_lightning import Trainer")

    ;; NLP
    ("transformers" . "from transformers import pipeline")
    ("nltk" . "import nltk")
    ("spacy" . "import spacy")
    ("fairseq" . "from fairseq.models import BlenderbotModel")

    ;; Statistics
    ("statsmodels" . "import statsmodels")
    ("sm" . "import statsmodels.api as sm")
    ("stats" . "from scipy import stats")
    ("pg" . "import pingouin as pg")


    ;; Signal Processing
    ("signal" . "from scipy import signal")
    ("pydsp" . "import pydsp as dsp")
    ("sf" . "import soundfile as sf")
    ("find_peaks" . "from scipy.signal import find_peaks")
    ("mne" . "import mne")

    ;; Image & Video Processing
    ("PIL" . "from PIL import Image")
    ("skimage" . "import skimage")
    ("cv2" . "import cv2")

    ;; Audio Processing
    ("AudioSegment" . "from pydub import AudioSegment")
    ("librosa" . "import librosa")
    ("sr" . "import speech_recognition as sr")

    ;; Web Development & APIs
    ("Flask" . "from flask import Flask")
    ("requests" . "import requests")
    ("websocket" . "import websocket")
    ("uvicorn" . "import uvicorn")

    ;; Database
    ("sqlite3" . "import sqlite3")
    ("sqlalchemy" . "from sqlalchemy import create_engine")
    ("pymongo" . "from pymongo import MongoClient")

    ;; GUI
    ("tkinter" . "import tkinter")
    ("tk" . "import tkinter as tk")

    ;; Testing
    ("pytest" . "import pytest")
    ("hypothesis" . "from hypothesis import given")

    ;; ML Ops & Experiment Management
    ("mlflow" . "import mlflow")
    ("wandb" . "import wandb")
    ("tensorboard" . "import tensorboard")
    ("optuna" . "import optuna")
    ("hydra" . "import hydra")
    ("airflow" . "from airflow import DAG")
    ("prefect" . "from prefect import flow")
    ("dagster" . "from dagster import job")
    ("kedro" . "import kedro")
    ("dvc" . "import dvc.api")

    ;; High Performance Computing
    ("numba" . "from numba import jit")
    ("cupy" . "import cupy as cp")
    ("jax" . "import jax.numpy as jnp")
    ("pyspark" . "from pyspark.sql import SparkSession")
    ("ray" . "import ray")

    ;; Web Scraping
    ("BeautifulSoup" . "from bs4 import BeautifulSoup")
    ("xml" . "import xml.etree.ElementTree as ET")

    ;; Custom Libraries
    ("mngs" . "import mngs")
    ("printc" . "import mngs.str import printc")    
    ;; ("load" . "from mngs.io import load")
    ;; ("save" . "from mngs.io import save")    
    ("torch_fn" . "from mngs.decorators import torch_fn")
    ("numpy_fn" . "from mngs.decorators import numpy_fn")
    ("pandas_fn" . "from mngs.decorators import pandas_fn")
    ("deprecated" . "from mngs.decorators import deprecated")
    ("split" . "from mngs.path import split")
    ("utils" . "from scripts import utils")
    ("ArrayLike" . "from mngs.typing import ArrayLike"))
  "A list of package names to automatically insert when missed."
  :type '(alist :key-type string :value-type string)
  :group 'python-import-manager)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Flake8
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun pim--find-flake8 ()
  "Find flake8 executable."
  (or pim-flake8-path
      (executable-find "flake8")
      (user-error "Cannot find flake8. Please install it or set pim-flake8-path")))

(defun pim--copy-contents-as-temp-file ()
  "Copy current buffer to temp file and return the filename."
  (let ((temp-file (make-temp-file "pim-")))
    (write-region (point-min) (point-max) temp-file)
    temp-file))

(defun pim--get-flake8-output (temp-file &optional args)
  "Run flake8 on TEMP-FILE with optional ARGS and return output."
  (let ((flake8-path (pim--find-flake8)))
    (with-temp-buffer
      (apply #'call-process flake8-path nil t nil
             (append (or args pim-flake8-args) (list temp-file)))
      (buffer-string))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Deletion
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun pim--find-unused-modules ()
  "Find unused modules from flake8 output."
  (let* ((temp-file (pim--copy-contents-as-temp-file))
         (output (pim--get-flake8-output temp-file '("--select=F401")))
         modules)
    (with-temp-buffer
      (insert output)
      (goto-char (point-min))
      (while (re-search-forward "F401 '\\([^']+\\)' imported but unused" nil t)
        (let ((module (match-string 1)))
          (push (if (string-match "\\([^.]+\\)$" module)
                    (match-string 1 module)
                  module)
                modules))))
    (delete-file temp-file)
    modules))

(defun pim--remove-module (module)
  "Remove specific MODULE from import lines."
  (save-excursion
    (goto-char (point-min))
    ;; Remove 'import module' lines
    (while (re-search-forward "^import .*$" nil t)
      (when (string-match-p (format "\\b%s\\b" module)
                           (match-string 0))
        (kill-whole-line)))
    ;; Remove 'from ... import' lines
    (goto-char (point-min))
    (while (re-search-forward "^from .* import.*$" nil t)
      (let* ((line (buffer-substring-no-properties
                    (line-beginning-position) (line-end-position)))
             (imports (when (string-match "import \\(.+\\)" line)
                       (split-string (match-string 1 line) "," t "[\s\t]+")))
             (new-imports (remove module imports)))
        (when (and imports (not (equal imports new-imports)))
          (delete-region (line-beginning-position) (line-end-position))
          (if new-imports
              (insert (format "from %s import %s"
                            (progn (string-match "from \\([^ ]+\\) import" line)
                                   (match-string 1 line))
                            (string-join new-imports ", ")))
            (kill-whole-line)))))))

(defun pim--cleanup-imports ()
  "Remove empty import lines."
  (save-excursion
    (goto-char (point-min))
    (while (re-search-forward "^from .* import *$" nil t)
      (kill-whole-line))))

;;;###autoload
(defun pim-delete-unused ()
  "Remove unused imports."
  (interactive)
  (let ((unused-modules (pim--find-unused-modules)))
    (dolist (module unused-modules)
      (pim--remove-module module))
    (pim--cleanup-imports)))

;;;###autoload
(defun pim-delete-duplicated ()
  "Remove duplicate import statements."
  (interactive)
  (save-excursion
    (let ((imports (make-hash-table :test 'equal)))
      (goto-char (point-min))
      (while (re-search-forward "^\\(from .* import .*\\|import .*\\)$" nil t)
        (let ((line (match-string 0)))
          (if (gethash line imports)
              (kill-whole-line)
            (puthash line t imports)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Insertion
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun pim--find-undefined ()
  "Find undefined names from flake8 output."
  (when (= (point-min) (point-max))
    (user-error "Buffer is empty"))
  (let* ((temp-file (pim--copy-contents-as-temp-file))
         (undefined-list '())
         (output (pim--get-flake8-output temp-file '("--select=F821"))))
    (with-temp-buffer
      (insert output)
      (goto-char (point-min))
      (while (re-search-forward "F821 undefined name '\\([^']+\\)'" nil t)
        (push (match-string 1) undefined-list)))
    (delete-file temp-file)
    undefined-list))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; About the place to insert missed packages

;; if the found missed pacakge is located upper than the main guard...
;; 1. find existing importing lines
;; 2. if not 1, the last of the header
;; 3. if not 1 nor 2. the top of the file

;; elseif missed pacakge is under than the main guard...
;; 1'. Find existing importing lines under the main guard
;; 2'. if not 1', the last of the hader
;; 3', if not 1' nor 2', jsut under the main guard

;; Implementing a dedicated function to pim--define-inserting-position would be practical

;; "header " means consective commenting lines, like ^# ...\n^#...\n\n

;; Note that insertion itself will update the position; so, main-guard position should be considered after insertion of missed packages 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; ;; no indent for main-gudard
;; ;;;###autoload
;; (defun pim--find-import-position (name current-offset)
;;   "Find proper position to insert import for NAME, considering CURRENT-OFFSET."
;;   (save-excursion
;;     (let* ((main-guard-pos (save-excursion
;;                             (re-search-forward "^if __name__ == \"__main__\":" nil t)))
;;            (adjusted-main-guard-pos (when main-guard-pos 
;;                                     (+ main-guard-pos current-offset)))
;;            (name-pos (save-excursion
;;                       (re-search-forward (regexp-quote name) nil t)))
;;            (adjusted-name-pos (when name-pos 
;;                               (+ name-pos current-offset)))
;;            (header-end (pim--find-header-end)))
;;       (if (and adjusted-main-guard-pos 
;;                adjusted-name-pos 
;;                (> adjusted-name-pos adjusted-main-guard-pos))
;;           (progn
;;             (goto-char adjusted-main-guard-pos)
;;             (+ (cond
;;                 ((re-search-forward "^import\\|^from" nil t)
;;                  (line-beginning-position))
;;                 (t
;;                  (goto-char adjusted-main-guard-pos)
;;                  (forward-line 1)
;;                  (point)))
;;                current-offset))
;;         (goto-char (point-min))
;;         (+ (cond
;;             ((and main-guard-pos
;;                   (re-search-forward "^import\\|^from" main-guard-pos t))
;;              (line-beginning-position))
;;             ((> header-end (point-min))
;;              header-end)
;;             (t (point-min)))
;;            current-offset)))))

;; ;; no indent for main guard
;; ;;;###autoload
;; (defun pim-insert-missed ()
;;   "Insert missing imports from predefined list based on undefined names."
;;   (interactive)
;;   (let ((undefined-names (pim--find-undefined))
;;         (import-positions (make-hash-table :test 'equal))
;;         (current-offset 0))
;;     (when undefined-names
;;       (dolist (name undefined-names)
;;         (let* ((import-line (cdr (assoc name pim-import-list)))
;;                (pos (pim--find-import-position name current-offset)))
;;           (when (and import-line pos)
;;             (push import-line (gethash pos import-positions))
;;             (setq current-offset (+ current-offset 
;;                                   (1+ (length import-line)))))))
      
;;       (save-excursion
;;         (maphash (lambda (pos lines)
;;                   (goto-char pos)
;;                   (dolist (line (reverse lines))
;;                     (insert line "\n")))
;;                 import-positions)
        
;;         (goto-char (point-min))
;;         (while (re-search-forward "\n\n\n+" nil t)
;;           (replace-match "\n\n"))))))


;; Implement pim--find-header-end


;;;###autoload
(defun pim--find-header-end ()
  "Find the end position of the header comment section in a Python file.
Header is defined as consecutive comment lines starting from the beginning."
  (save-excursion
    (goto-char (point-min))
    (let ((end-pos (point-min)))
      (while (and (not (eobp))
                  (looking-at "^#"))
        (setq end-pos (line-end-position))
        (forward-line 1))
      ;; Skip empty lines after comments
      (while (and (not (eobp))
                  (looking-at "^$"))
        (setq end-pos (line-end-position))
        (forward-line 1))
      (1+ end-pos))))

;;;###autoload
(defun pim--find-import-position (name current-offset)
  "Find proper position to insert import for NAME, considering CURRENT-OFFSET."
  (save-excursion
    (let* ((main-guard-pos (save-excursion
                            (re-search-forward "^if __name__ == \"__main__\":" nil t)))
           (adjusted-main-guard-pos (when main-guard-pos 
                                    (+ main-guard-pos current-offset)))
           (name-pos (save-excursion
                      (re-search-forward (regexp-quote name) nil t)))
           (adjusted-name-pos (when name-pos 
                              (+ name-pos current-offset)))
           (header-end (pim--find-header-end))
           (base-indent (when main-guard-pos
                         (save-excursion
                           (goto-char (line-beginning-position))
                           (forward-line 1)
                           (current-indentation)))))
      (if (and adjusted-main-guard-pos 
               adjusted-name-pos 
               (> adjusted-name-pos adjusted-main-guard-pos))
          (progn
            (goto-char adjusted-main-guard-pos)
            (forward-line 1)
            (+ (point) current-offset))
        (goto-char (point-min))
        (+ (cond
            ((and main-guard-pos
                  (re-search-forward "^import\\|^from" main-guard-pos t))
             (line-beginning-position))
            ((> header-end (point-min))
             header-end)
            (t (point-min)))
           current-offset)))))

;;;###autoload
(defun pim-insert-missed ()
  "Insert missing imports from predefined list based on undefined names."
  (interactive)
  (let ((undefined-names (pim--find-undefined))
        (import-positions (make-hash-table :test 'equal))
        (current-offset 0))
    (when undefined-names
      (dolist (name undefined-names)
        (let* ((import-line (cdr (assoc name pim-import-list)))
               (pos (pim--find-import-position name current-offset))
               (main-guard-pos (save-excursion
                               (goto-char pos)
                               (re-search-backward "^if __name__ == \"__main__\":" nil t))))
          (when (and import-line pos)
            (let ((indented-line 
                   (if (and main-guard-pos (> pos main-guard-pos))
                       (concat "    " import-line)
                     import-line)))
              (push indented-line (gethash pos import-positions))
              (setq current-offset (+ current-offset 
                                    (1+ (length indented-line))))))))
      
      (save-excursion
        (maphash (lambda (pos lines)
                  (goto-char pos)
                  (dolist (line (reverse lines))
                    (insert line "\n")))
                import-positions)
        
        (goto-char (point-min))
        (while (re-search-forward "\n\n\n+" nil t)
          (replace-match "\n\n"))))))

;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; ;; isort
;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; (defun pim--find-isort ()
;;   "Find isort executable."
;;   (or pim-isort-path
;;       (executable-find "isort")
;;       (user-error "Cannot find isort. Please install it or set pim-isort-path")))

;; (defun pim--get-isort-output (temp-file &optional args)
;;   "Run isort on TEMP-FILE with optional ARGS and return output."
;;   (let ((isort-path (pim--find-isort)))
;;     (with-temp-buffer
;;       (apply #'call-process isort-path nil t nil
;;              (append (or args pim-isort-args) (list temp-file)))
;;       (buffer-string))))

;; (defun pim-isort ()
;;   "Sort imports using isort."
;;   (interactive)
;;   (let* ((temp-file (pim--copy-contents-as-temp-file)))
;;     (pim--get-isort-output temp-file)
;;     (erase-buffer)
;;     (insert-file-contents temp-file)
;;     (delete-file temp-file)))

;;;###autoload
(defun pim-fix-imports ()
  "Fix imports in current buffer."
  (interactive)
  (let ((original-point (point)))
    (pim-delete-unused)
    (pim-insert-missed)
    (pim-delete-duplicated)
    (python-isort-buffer)
    (goto-char original-point)))

;;;###autoload
(defalias 'pim 'pim-fix-imports)

(provide 'python-import-manager)

;;; python-import-manager.el ends here


(message "%s was loaded." (file-name-nondirectory (or load-file-name buffer-file-name)))
