;;; -*- coding: utf-8; lexical-binding: t -*-
;;; Author: ywatanabe
;;; Timestamp: <2025-04-23 09:08:16>
;;; File: /home/ywatanabe/.emacs.d/lisp/python-import-manager/predefined-packages/pim-imports-standard.el

;;; Copyright (C) 2025 Yusuke Watanabe (ywatanabe@alumni.u-tokyo.ac.jp)

(defvar pim-imports-standard
  '(
    ("argparse" . "import argparse")
    ("warnings" . "import warnings")
    ("inspect" . "import inspect")
    ("re" . "import re")
    ("os" . "import os")
    ("glob" . "from glob import glob")
    ("json" . "import json")
    ("time" . "import time")
    ("datetime" . "from datetime import datetime")
    ("timedelta" . "from datetime import timedelta")
    ("itertools" . "import itertools")
    ("combinations" . "from itertools import combinations")
    ("cycle" . "from itertools import cycle")
    ("product" . "from itertools import product")
    ("functools" . "import functools")
    ("partial" . "from functools import partial")
    ("wraps" . "from functools import wraps")
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
    ("natsort" . "import natsort")
    ("natsorted" . "from natsort import natsorted")

    ;; ABC
    ("abc" . "from collections import abc")
    ("ABC" . "from abc import ABC")
    ("abstractmethod" . "from abc import abstractmethod")
    ("abstractproperty" . "from abc import abstractproperty")
    ("abstractclassmethod" . "from abc import abstractclassmethod")
    ("abstractstaticmethod" . "from abc import abstractstaticmethod")

    ;; Typing
    ("Tuple" . "from typing import Tuple")
    ("Dict" . "from typing import Dict")
    ("Any" . "from typing import Any")
    ("_Any" . "from typing import Any as _Any")
    ("Union" . "from typing import Union")
    ("Sequence" . "from typing import Sequence")
    ("Literal" . "from typing import Literal")
    ("Optional" . "from typing import Optional")
    ("Iterable" . "from typing import Iterable")
    ("Callable" . "from typing import Callable")
    ("List" . "from typing import List")
    ("TypeVar" . "from typing import TypeVar")
    ("Protocol" . "from typing import Protocol")
    ("Generic" . "from typing import Generic")
    ("cast" . "from typing import cast")
    ("Type" . "from typing import Type")
    ("Mapping" . "from typing import Mapping")
    ("Iterator" . "from typing import Iterator")
    ("Generator" . "from typing import Generator")
    ("NamedTuple" . "from typing import NamedTuple")
    ("TextIO" . "from typing import TextIO")

    ;; Context Managers
    ("contextlib" . "import contextlib")
    ("contextmanager" . "from contextlib import contextmanager")
    ("ContextManager" . "from typing import ContextManager")
    ("AsyncContextManager" . "from typing import AsyncContextManager")
    ("AbstractContextManager"
     . "from contextlib import AbstractContextManager")
    ("AbstractAsyncContextManager"
     . "from contextlib import AbstractAsyncContextManager")
    ("suppress" . "from contextlib import suppress")
    ("closing" . "from contextlib import closing")
    ("ExitStack" . "from contextlib import ExitStack")
    ("nullcontext" . "from contextlib import nullcontext")

    ;; Debugging
    ("ipdb" . "import ipdb")
    ("get_ipython" . "from IPython import get_ipython")

    ;; Collections
    ("defaultdict" . "from collections import defaultdict")
    ("Counter" . "from collections import Counter")
    ("deque" . "from collections import deque")
    ("namedtuple" . "from collections import namedtuple")
    ("OrderedDict" . "from collections import OrderedDict")

    ;; Parallel
    ("multiprocessing" . "import multiprocessing")
    ("ThreadPoolExecutor"
     . "from concurrent.futures import ThreadPoolExecutor")
    ("ProcessPoolExecutor"
     . "from concurrent.futures import ProcessPoolExecutor")
    ("as_completed" . "from concurrent.futures import as_completed")
    ("Parallel" . "from joblib import Parallel")
    ("delayed" . "from joblib import delayed")
    ("shared_memory" . "from multiprocessing import shared_memory")
    ("Manager" . "from multiprocessing import Manager")
    ("Pool" . "from multiprocessing import Pool")
    ("Lock" . "from threading import Lock")
    ("Event" . "from threading import Event")
    ("Thread" . "from threading import Thread")

    ;; GUI
    ("tkinter" . "import tkinter")
    ("tk" . "import tkinter as tk")

    ;; Testing
    ("pytest" . "import pytest")
    ("hypothesis" . "from hypothesis import given")

    ;; for __init__.py
    ("isfunction" . "from inspect import isfunction")
    ("isclass" . "from inspect import isclass")
    ("getmembers" . "from inspect import getmembers")

    ("shutil" . "import shutil")
    ("tempfile" . "import tempfile"))

  "Standard Python imports.")

(provide 'pim-imports-standard)

(when
    (not load-file-name)
  (message "pim-imports-standard.el loaded."
           (file-name-nondirectory
            (or load-file-name buffer-file-name))))