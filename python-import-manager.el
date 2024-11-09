;;; -*- lexical-binding: t -*-
;;; Author: ywatanabe
;;; Time-stamp: <2024-11-09 14:01:35 (ywatanabe)>
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

(add-to-list 'load-path (concat pim--script-dir "predefined-packages"))
(require 'pim-imports-loader)

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
;; ;; working except for type hint
;; (defun pim--find-undefined ()
;;   "Find undefined names from flake8 output."
;;   (when (= (point-min) (point-max))
;;     (user-error "Buffer is empty"))
;;   (let* ((temp-file (pim--copy-contents-as-temp-file))
;;          (undefined-list '())
;;          (output (pim--get-flake8-output temp-file '("--select=F821"))))
;;     (with-temp-buffer
;;       (insert output)
;;       (goto-char (point-min))
;;       (while (re-search-forward "F821 undefined name '\\([^']+\\)'" nil t)
;;         (push (match-string 1) undefined-list)))
;;     (delete-file temp-file)
;;     undefined-list))

(defun pim--find-undefined ()
  "Find undefined names from flake8 output."
  (interactive)
  (when (= (point-min) (point-max))
    (user-error "Buffer is empty"))
  (let* ((temp-file (pim--copy-contents-as-temp-file))
         (undefined-list '())
         (output (pim--get-flake8-output temp-file '("--select=F821"))))
    (with-temp-buffer
      (insert output)
      (goto-char (point-min))
      (while (re-search-forward "F821 undefined name '\\([^']+\\)'" nil t)
        (let* ((name (match-string 1))
               (types (progn
                       (with-temp-buffer
                         (insert name)
                         (goto-char (point-min))
                         (let ((matches '()))
                           (while (re-search-forward "\\([A-Z][A-Za-z0-9_]*\\)[[(]?" nil t)
                             (push (match-string 1) matches))
                           matches)))))
          (setq undefined-list (append types undefined-list)))))
    (delete-file temp-file)
    (delete-dups undefined-list)))


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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Collection
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun pim--collect-all-imports ()
  "Collect all import statements from the buffer."
  (let (imports)
    (save-excursion
      (goto-char (point-min))
      (while (re-search-forward "^\\(from\\|import\\) .*$" nil t)
        (push (match-string-no-properties 0) imports)
        (kill-whole-line)))
    (nreverse imports)))

;; (defun pim--insert-imports-at-top (imports)
;;   "Insert IMPORTS at the appropriate position near the top of the file."
;;   (save-excursion
;;     (goto-char (pim--find-header-end))
;;     (insert (string-join imports "\n") "\n\n")))

(defun pim--find-imports-marker ()
  "Find the position after '\"\"\"Imports\"\"\"' line."
  (save-excursion
    (goto-char (point-min))
    (if (re-search-forward "^\"\"\"Imports\"\"\"$" nil t)
        (progn (forward-line 1) (point))
      (pim--find-header-end))))

(defun pim--insert-imports-at-top (imports)
  "Insert IMPORTS after the Imports marker or at header end."
  (save-excursion
    (goto-char (pim--find-imports-marker))
    (insert (string-join imports "\n") "\n\n")))

;;;###autoload
(defun pim-consolidate-imports ()
  "Move all import statements after '\"\"\"Imports\"\"\"' line."
  (interactive)
  (let ((imports (pim--collect-all-imports)))
    (pim--insert-imports-at-top imports)))

;; (defun pim-consolidate-imports ()
;;   "Move all import statements to the top of the file."
;;   (interactive)
;;   (let ((imports (pim--collect-all-imports)))
;;     (pim--insert-imports-at-top imports)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; All-in-One function
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun pim-fix-imports ()
  "Fix imports in current buffer."
  (interactive)
  (let ((original-point (point)))
    (pim-consolidate-imports)
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
