;;; -*- coding: utf-8; lexical-binding: t -*-
;;; Author: 2025-01-22 21:49:36
;;; Timestamp: <2025-01-22 21:49:36>
;;; File: /home/ywatanabe/.emacs.d/lisp/python-import-manager/predefined-packages/pim-imports-web.el


(defvar pim-imports-web
  '(
    ;; Web Development & APIs
    ("Flask" . "from flask import Flask")
    ("requests" . "import requests")
    ("websocket" . "import websocket")
    ("uvicorn" . "import uvicorn")

    ;; HTTP/Async
    ("aiohttp" . "import aiohttp")
    ("ClientSession" . "from aiohttp import ClientSession")
    ("requests" . "import requests")

    ;; XML Processing
    ("ET" . "import xml.etree.ElementTree as ET")
    ("xml" . "import xml")
    ("fromstring" . "from xml.etree.ElementTree import fromstring")

    ;; Web Scraping
    ("BeautifulSoup" . "from bs4 import BeautifulSoup")

    ;; Django Core
    ("settings" . "from django.conf import settings")
    ("HttpResponse" . "from django.http import HttpResponse")
    ("JsonResponse" . "from django.http import JsonResponse")
    ("HttpResponseRedirect" . "from django.http import HttpResponseRedirect")
    ("render" . "from django.shortcuts import render")
    ("redirect" . "from django.shortcuts import redirect")
    ("get_object_or_404" . "from django.shortcuts import get_object_or_404")
    ("reverse" . "from django.urls import reverse")
    ("path" . "from django.urls import path")
    ("include" . "from django.urls import include")
    ("models" . "from django.db import models")
    ("forms" . "from django import forms")
    ("admin" . "from django.contrib import admin")
    ("login_required" . "from django.contrib.auth.decorators import login_required")
    ("messages" . "from django.contrib import messages")
    ("authenticate" . "from django.contrib.auth import authenticate")
    ("login" . "from django.contrib.auth import login")
    ("logout" . "from django.contrib.auth import logout")
    ("User" . "from django.contrib.auth.models import User")
    ("Q" . "from django.db.models import Q")
    ("transaction" . "from django.db import transaction")
    ("ValidationError" . "from django.core.exceptions import ValidationError")

    ;; Django Forms
    ("ModelForm" . "from django.forms import ModelForm")
    ("Form" . "from django.forms import Form")
    ("CharField" . "from django.forms import CharField")
    ("EmailField" . "from django.forms import EmailField")
    ("PasswordInput" . "from django.forms import PasswordInput")

    ;; Django Models
    ("Model" . "from django.db.models import Model")
    ("CharField" . "from django.db.models import CharField")
    ("TextField" . "from django.db.models import TextField")
    ("IntegerField" . "from django.db.models import IntegerField")
    ("DateTimeField" . "from django.db.models import DateTimeField")
    ("ForeignKey" . "from django.db.models import ForeignKey")
    ("ManyToManyField" . "from django.db.models import ManyToManyField")
    ("CASCADE" . "from django.db.models import CASCADE")
    ("Sum" . "from django.db.models import Sum")
    ("Count" . "from django.db.models import Count")
    ("Avg" . "from django.db.models import Avg")

    ;; Django Views
    ("View" . "from django.views import View")
    ("TemplateView" . "from django.views.generic import TemplateView")
    ("ListView" . "from django.views.generic import ListView")
    ("DetailView" . "from django.views.generic import DetailView")
    ("CreateView" . "from django.views.generic import CreateView")
    ("UpdateView" . "from django.views.generic import UpdateView")
    ("DeleteView" . "from django.views.generic import DeleteView")

    ;; Django Auth
    ("LoginView" . "from django.contrib.auth.views import LoginView")
    ("LogoutView" . "from django.contrib.auth.views import LogoutView")
    ("PasswordResetView" . "from django.contrib.auth.views import PasswordResetView")
    ("AbstractUser" . "from django.contrib.auth.models import AbstractUser")
    ("Group" . "from django.contrib.auth.models import Group")
    ("Permission" . "from django.contrib.auth.models import Permission"))
  "Web development and database related Python imports.")

(provide 'pim-imports-web)


(when (not load-file-name)
  (message "%s was loaded." (file-name-nondirectory (or load-file-name buffer-file-name))))