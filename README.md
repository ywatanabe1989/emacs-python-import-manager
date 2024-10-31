# Python import manager for Emacs

An Emacs package that automatically manages Python imports: removes unused imports, adds missing ones, and eliminates duplicates.

## Usage

- `M-x pim` (alias for `pim-fix-imports`): Fix all imports
- `pim` (or `pim-fix-imports`) includes:
  - `M-x pim-delete-unused`: Remove unused imports
  - `M-x pim-insert-missing`: Add missing imports
  - `M-x pim-delete-duplicates`: Remove duplicate imports
- `M-x pim-auto-mode`: Toggle automatic import fixing on save


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
;; Custom flake8 path
(setq pim-flake8-path "/path/to/flake8")

;; Custom flake8 arguments
(setq pim-flake8-args '("--max-line-length=100" "--select=F401,F821" "--isolated"))

;; Custom import aliases
(setq pim-import-aliases
      '(("numpy" . "np")
        ("pandas" . "pd")
        ("matplotlib.pyplot" . "plt")
        ("seaborn" . "sns")))
```

## License

GPL-3.0 or later


