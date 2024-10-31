# Python import manager for Emacs

An Emacs package that automatically manages Python imports

## Usage

| Command | Description |
|---|---|
| `M-x pim-fix-imports` or `M-x pim` | Remove unused imports, add missing imports, and remove duplicate imports |
| `M-x pim-insert-missing` | Add missing imports |
| `M-x pim-delete-duplicates` | Remove duplicate imports |
| `M-x pim-auto-mode` | Toggle automatic import fixing on save |

## Requirements

- Emacs 26.1 or later
- flake8 (Python package)

## Installation

1. Install flake8:
```bash
pip install flake8
```

2. Clone the repository:
```bash
git clone git@github.com:ywatanabe1989/python-import-manager.git ~/.emacs.d/lisp/python-import-manager
```

3. Add to your `load-path`:
```elisp
(add-to-list 'load-path (expand-file-name "~/.emacs.d/lisp/python-import-manager/"))
```

## Customization

```elisp
;;; -*- lexical-binding: t -*-
;;; Author: ywatanabe
;;; Time-stamp: <2024-11-01 01:49:39 (ywatanabe)>
;;; File: ./.dotfiles/.emacs.d/inits/300-python/300-python-pim.el

(add-to-list 'load-path (expand-file-name "~/.emacs.d/lisp/python-import-manager/"))
(require 'python-import-manager)

(setq pim-python-path (expand-file-name "~/.env-3.11/bin/python3"))
(setq pim-flake8-path (expand-file-name "~/.env-3.11/bin/flake8"))
(setq pim-isort-path (expand-file-name "~/.env-3.11/bin/isort"))
(setq pim-import-aliases
      (append pim-import-aliases
             '(("numpy" . "np")
               ("pandas" . "pd")
               ("matplotlib.pyplot" . "plt")
               ("seaborn" . "sns")
               ("mngs" . "mngs")
               ("scripts.utils" . "utils"))))

(add-hook 'python-mode-hook
          (lambda ()
          (pim-auto-mode 1)))

(message "%s was loaded." (file-name-nondirectory (or load-file-name buffer-file-name)))
```

## License

GPL-3.0 or later


