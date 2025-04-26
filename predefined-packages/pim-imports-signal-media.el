;;; -*- coding: utf-8; lexical-binding: t -*-
;;; Author: 2025-01-22 21:49:30
;;; Timestamp: <2025-01-22 21:49:30>
;;; File: /home/ywatanabe/.emacs.d/lisp/python-import-manager/predefined-packages/pim-imports-signal-media.el


(defvar pim-imports-signal-media
  '(
    ;; Signal Processing
    ("signal" . "from scipy import signal")
    ("dsp" . "import pydsp as dsp")
    ("sf" . "import soundfile as sf")
    ("find_peaks" . "from scipy.signal import find_peaks")
    ("mne" . "import mne")
    ("welch" . "from scipy.signal import welch")
    ("butter" . "from scipy.signal import butter, filtfilt")
    ("hilbert" . "from scipy.signal import hilbert")
    ("spectrogram" . "from scipy.signal import spectrogram")
    ("stft" . "from scipy.signal import stft")
    ("cwt" . "from scipy.signal import cwt")
    ("coherence" . "from scipy.signal import coherence")
    ("detrend" . "from scipy.signal import detrend")
    ("periodogram" . "from scipy.signal import periodogram")

    ;; scipy.ndimage functions
    ("zoom" . "from scipy.ndimage import zoom")
    ("gaussian_filter" . "from scipy.ndimage import gaussian_filter")
    ("median_filter" . "from scipy.ndimage import median_filter")
    ("rotate" . "from scipy.ndimage import rotate")
    ("shift" . "from scipy.ndimage import shift")
    ("binary_erosion" . "from scipy.ndimage import binary_erosion")
    ("binary_dilation" . "from scipy.ndimage import binary_dilation")
    ("label" . "from scipy.ndimage import label")
    ("fourier_gaussian" . "from scipy.ndimage import fourier_gaussian")
    ("measurements" . "from scipy.ndimage import measurements")
    ;; ("convolve" . "from scipy.ndimage import convolve")
    ;; ("correlate" . "from scipy.ndimage import correlate")
    ("sobel" . "from scipy.ndimage import sobel")
    ("laplace" . "from scipy.ndimage import laplace")

    ;; Image Processing
    ("Image" . "from PIL import Image")
    ("ImageOps" . "from PIL import ImageOps")
    ("ImageEnhance" . "from PIL import ImageEnhance")
    ("ImageDraw" . "from PIL import ImageDraw")
    ("ImageFont" . "from PIL import ImageFont")
    ("ImageFilter" . "from PIL import ImageFilter")
    ("cv2" . "import cv2")
    ("imageio" . "import imageio")
    ("rearrange" . "from einops import rearrange")
    ("skimage" . "import skimage")
    ("TiffFile" . "from tifffile import TiffFile")
    ("imsave" . "from imageio import imsave")
    ("imread" . "from imageio import imread")
    ("io" . "from skimage import io")
    ("color" . "from skimage import color")
    ("filters" . "from skimage import filters")

    ;; Audio Processing
    ("AudioSegment" . "from pydub import AudioSegment")
    ("librosa" . "import librosa")
    ("sr" . "import speech_recognition as sr")
    ("load" . "from librosa import load")
    ("feature" . "from librosa import feature")
    ("effects" . "from librosa import effects")
    ("display" . "from librosa import display")
    ("beat" . "from librosa.beat import beat_track")
    ("onset" . "from librosa.onset import onset_detect")
    ("sox" . "import sox")
    ("pyworld" . "import pyworld")
    ("sounddevice" . "import sounddevice as sd")
    ("wave" . "import wave"))

    ;; ;; Audio Processing
    ;; ("AudioSegment" . "from pydub import AudioSegment")
    ;; ("librosa" . "import librosa")
    ;; ("sr" . "import speech_recognition as sr"))


  "Signal processing and media related Python imports.")

(provide 'pim-imports-signal-media)


(when (not load-file-name)
  (message "%s was loaded." (file-name-nondirectory (or load-file-name buffer-file-name))))