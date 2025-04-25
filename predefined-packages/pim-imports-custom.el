;;; -*- coding: utf-8; lexical-binding: t -*-
;;; Author: 2025-01-22 21:49:24
;;; Timestamp: <2025-01-22 21:49:24>
;;; File: /home/ywatanabe/.emacs.d/lisp/python-import-manager/predefined-packages/pim-imports-custom.el


(defvar pim-imports-custom
  '(
    ("mngs" . "import mngs")
    ("load_configs" . "from mngs.io import load_configs")
    ("printc" . "from mngs.str import printc")
    ("cache_mem" . "from mngs.decorators import cache_mem")
    ("cache_disk" . "from mngs.decorators import cache_disk")
    ("torch_fn" . "from mngs.decorators import torch_fn")
    ("numpy_fn" . "from mngs.decorators import numpy_fn")
    ("pandas_fn" . "from mngs.decorators import pandas_fn")
    ("deprecated" . "from mngs.decorators import deprecated")
    ("split" . "from mngs.path import split")
    ("utils" . "from scripts import utils")
    ("ArrayLike" . "from mngs.types import ArrayLike"))
  "Custom library imports.")

(provide 'pim-imports-custom)


(when (not load-file-name)
  (message "%s was loaded." (file-name-nondirectory (or load-file-name buffer-file-name))))