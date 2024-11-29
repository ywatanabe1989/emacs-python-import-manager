;;; -*- lexical-binding: t -*-
;;; Author: ywatanabe
;;; Time-stamp: <2024-11-17 12:20:13 (ywatanabe)>
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
(require 'blacken)

(defcustom pim-auto-mode nil
  "When non-nil, automatically run PIM on save in Python buffers."
  :type 'boolean
  :group 'python-import-manager)

(defun pim-enable-auto-mode ()
  "Enable automatic PIM on save for Python buffers."
  (when (eq major-mode 'python-mode)
    (add-hook 'before-save-hook #'pim-fix-imports nil t)))  ; Changed from #'pim to #'pim-fix-imports

(defun pim-disable-auto-mode ()
  "Disable automatic PIM on save for Python buffers."
  (remove-hook 'before-save-hook #'pim t))

(define-minor-mode pim-auto-mode
  "Toggle automatic PIM on save."
  :global t
  :group 'python-import-manager
  (if pim-auto-mode
      (progn
        (add-hook 'python-mode-hook #'pim-enable-auto-mode)
        ;; Apply to existing Python buffers
        (dolist (buf (buffer-list))
          (with-current-buffer buf
            (when (eq major-mode 'python-mode)
              (pim-enable-auto-mode)))))
    (progn
      (remove-hook 'python-mode-hook #'pim-enable-auto-mode)
      (dolist (buf (buffer-list))
        (with-current-buffer buf
          (when (eq major-mode 'python-mode)
            (pim-disable-auto-mode)))))))

(defgroup python-import-manager nil
  "Management of Python imports."
  :group 'tools
  :prefix "pim-")

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
  (or (bound-and-true-p python-isort-command)
      (executable-find "isort"))
  "Path to isort executable."
  :type 'string
  :group 'python-import-manager)

(defcustom pim-isort-args
  '("--profile" "black" "--line-length" "100")
  "Arguments to pass to isort."
  :type '(repeat string)
  :group 'python-import-manager)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; pim-import-list
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(add-to-list 'load-path (concat pim--script-dir "predefined-packages"))

(require 'pim-imports-custom)
(require 'pim-imports-data-science)
(require 'pim-imports-signal-media)
(require 'pim-imports-standard)
(require 'pim-imports-web)

;; Merge all imports into one list
(defvar pim-import-list nil "All predefined Python imports combined.")
(setq pim-import-list
      (append pim-imports-custom
              pim-imports-data-science
              pim-imports-signal-media
              pim-imports-standard
              pim-imports-web))


(defun pim--update-import-list ()
  "Update the combined import list from all predefined package lists."
  (setq pim-import-list
        (delete-dups
         (append pim-imports-custom
                 pim-imports-data-science
                 pim-imports-signal-media
                 pim-imports-standard
                 pim-imports-web))))


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
  (interactive)
  (when (= (point-min) (point-max))
    (user-error "Buffer is empty"))

  (let* ((temp-file (pim--copy-contents-as-temp-file))
         (undefined-list '())
         (output (progn
                  (message "Running flake8 on temp file: %s" temp-file)
                  (pim--get-flake8-output temp-file '("--select=F821")))))

    (message "Flake8 output: %s" output)

    (with-temp-buffer
      (insert output)
      (message "Buffer content after insert: %s" (buffer-string))

      (goto-char (point-min))
      (while (re-search-forward "F821 undefined name '\\([^']+\\)'" nil t)
        (let ((name (match-string 1)))
          (message "Found undefined name: %s" name)
          (when (assoc name pim-import-list)
            (message "Name %s found in pim-import-list" name)
            (push name undefined-list)))))

    (message "Temporary file deletion status: %s"
             (condition-case err
                 (progn (delete-file temp-file) "success")
               (error (format "failed: %s" err))))

    (message "Final undefined-list: %s" undefined-list)
    (delete-dups undefined-list)))

;; (defun pim--find-undefined ()
;;   "Find undefined names from flake8 output."
;;   (interactive)
;;   (message "Starting pim--find-undefined...")
;;   (when (= (point-min) (point-max))
;;     (user-error "Buffer is empty"))
;;   (let* ((temp-file (pim--copy-contents-as-temp-file))
;;          (undefined-list '())
;;          (output (pim--get-flake8-output temp-file '("--select=F821"))))
;;     (message "Flake8 output: %s" output)
;;     (save-excursion
;;       (goto-char (point-min))
;;       (while (re-search-forward "\\b\\([a-zA-Z_][a-zA-Z0-9_]*\\)(" nil t)
;;         (let ((name (match-string 1)))
;;           (when (assoc name pim-import-list)
;;             (message "Found function call: %s" name)
;;             (push name undefined-list)))))
;;     (with-temp-buffer
;;       (insert output)
;;       (goto-char (point-min))
;;       (while (re-search-forward "F821 undefined name '\\([^']+\\)'" nil t)
;;         (let ((name (match-string 1)))
;;           (when (assoc name pim-import-list)
;;             (message "Found undefined name: %s" name)
;;             (push name undefined-list)))))
;;     (delete-file temp-file)
;;     (let ((result (delete-dups undefined-list)))
;;       (message "Final undefined list: %s" result)
;;       result)))


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

;; ;; ;; working
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
;;            (header-end (pim--find-header-end))
;;            (base-indent (when main-guard-pos
;;                          (save-excursion
;;                            (goto-char (line-beginning-position))
;;                            (forward-line 1)
;;                            (current-indentation)))))
;;       (if (and adjusted-main-guard-pos
;;                adjusted-name-pos
;;                (> adjusted-name-pos adjusted-main-guard-pos))
;;           (progn
;;             (goto-char adjusted-main-guard-pos)
;;             (forward-line 1)
;;             (+ (point) current-offset))
;;         (goto-char (point-min))
;;         (+ (cond
;;             ((and main-guard-pos
;;                   (re-search-forward "^import\\|^from" main-guard-pos t))
;;              (line-beginning-position))
;;             ((> header-end (point-min))
;;              header-end)
;;             (t (point-min)))
;;            current-offset)))))

(defun pim--find-import-position (name current-offset)
  "Find proper position to insert import for NAME, considering CURRENT-OFFSET."
  (message "Finding import position for %s (offset: %d)" name current-offset)
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
      (message "Main guard: %s, Name pos: %s, Header end: %s"
               main-guard-pos name-pos header-end)
      (let ((final-pos
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
                  current-offset))))
        (message "Selected position: %d" final-pos)
        final-pos))))

;; (defun pim-insert-missed ()
;;   "Insert missing imports from predefined list based on undefined names."
;;   (interactive)
;;   (message "Starting pim-insert-missed...")
;;   (let ((undefined-names (pim--find-undefined))
;;         (import-positions (make-hash-table :test 'equal))
;;         (current-offset 0))
;;     (message "Found undefined names: %s" undefined-names)
;;     (when undefined-names
;;       (dolist (name undefined-names)
;;         (let* ((import-line (cdr (assoc name pim-import-list)))
;;                (pos (pim--find-import-position name current-offset))
;;                (main-guard-pos (save-excursion
;;                                (goto-char pos)
;;                                (re-search-backward "^if __name__ == \"__main__\":" nil t))))
;;           (message "Processing %s: line '%s' at pos %d" name import-line pos)
;;           (when (and import-line pos)
;;             (let ((indented-line
;;                    (if (and main-guard-pos (> pos main-guard-pos))
;;                        (concat "    " import-line)
;;                      import-line)))
;;               (push indented-line (gethash pos import-positions))
;;               (setq current-offset (+ current-offset
;;                                     (1+ (length indented-line))))))))
;;       (message "Final import positions: %s" import-positions)
;;       (save-excursion
;;         (maphash (lambda (pos lines)
;;                   (goto-char pos)
;;                   (dolist (line (reverse lines))
;;                     (message "Inserting at %d: %s" pos line)
;;                     (insert line "\n")))
;;                 import-positions)
;;         (goto-char (point-min))
;;         (while (re-search-forward "\n\n\n+" nil t)
;;           (replace-match "\n\n"))))))

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
;;                (pos (pim--find-import-position name current-offset))
;;                (main-guard-pos (save-excursion
;;                                (goto-char pos)
;;                                (re-search-backward "^if __name__ == \"__main__\":" nil t))))
;;           (when (and import-line pos)
;;             (let ((indented-line
;;                    (if (and main-guard-pos (> pos main-guard-pos))
;;                        (concat "    " import-line)
;;                      import-line)))
;;               (push indented-line (gethash pos import-positions))
;;               (setq current-offset (+ current-offset
;;                                     (1+ (length indented-line))))))))

;;       (save-excursion
;;         (maphash (lambda (pos lines)
;;                   (goto-char pos)
;;                   (dolist (line (reverse lines))
;;                     (insert line "\n")))
;;                 import-positions)

;;         (goto-char (point-min))
;;         (while (re-search-forward "\n\n\n+" nil t)
;;           (replace-match "\n\n"))))))


;;;###autoload
(defun pim-insert-missed ()
  "Insert missing imports from predefined list based on undefined names."
  (interactive)
  (let ((undefined-names (progn
                          (message "Finding undefined names...")
                          (pim--find-undefined)))
        (import-positions (make-hash-table :test 'equal))
        (current-offset 0))

    (message "Undefined names found: %s" undefined-names)

    (when undefined-names
      (dolist (name undefined-names)
        (message "Processing name: %s" name)
        (let* ((import-line (cdr (assoc name pim-import-list)))
               (pos (progn
                     (message "Finding import position for %s with offset %d" name current-offset)
                     (pim--find-import-position name current-offset)))
               (main-guard-pos (save-excursion
                               (goto-char pos)
                               (re-search-backward "^if __name__ == \"__main__\":" nil t))))

          (message "Import line: %s, Position: %s, Main guard pos: %s"
                  import-line pos main-guard-pos)

          (when (and import-line pos)
            (let ((indented-line
                   (if (and main-guard-pos (> pos main-guard-pos))
                       (progn
                         (message "Indenting line due to main guard")
                         (concat "    " import-line))
                     import-line)))
              (message "Adding line to position %d: %s" pos indented-line)
              (push indented-line (gethash pos import-positions))
              (setq current-offset (+ current-offset
                                    (1+ (length indented-line))))))))

      (message "Final import positions hash table: %s"
              (let ((contents '()))
                (maphash (lambda (k v) (push (cons k v) contents))
                        import-positions)
                contents))

      (save-excursion
        (maphash (lambda (pos lines)
                  (message "Inserting at position %d: %s" pos lines)
                  (goto-char pos)
                  (dolist (line (reverse lines))
                    (insert line "\n")))
                import-positions)

        (goto-char (point-min))
        (message "Cleaning up multiple newlines")
        (while (re-search-forward "\n\n\n+" nil t)
          (replace-match "\n\n"))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Split for solid processing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; (defun pim--split-imports ()
;;   "Split 'from x import (A, B)' statements into separate lines.
;; Returns list of split import statements."
;;   (save-excursion
;;     (let (imports-list)
;;       (goto-char (point-min))
;;       (while (re-search-forward "^from \\([^ ]+\\) import (\\([^)]+\\))" nil t)
;;         (let* ((module (match-string 1))
;;                (imports (split-string (match-string 2) "," t "[ \t\n]+"))
;;                (split-imports (mapcar (lambda (imp)
;;                                       (format "from %s import %s" module imp))
;;                                     imports)))
;;           (kill-whole-line)
;;           (setq imports-list (append imports-list split-imports))))
;;       (when imports-list
;;         (goto-char (point-min))
;;         (insert (string-join (delete-dups imports-list) "\n") "\n")))))

;; ;; ;; not working sometims
;; (defun pim--split-imports ()
;;   "Split multi-line import statements properly."
;;   (interactive)
;;   (let ((original-point (point)))
;;     (save-excursion
;;       (goto-char (point-min))
;;       (message "Searching for imports...")
;;       (while (re-search-forward "^from \\([^ ]+\\) import *(\n?\\([^)]+\\))" nil t)
;;         (let* ((module (match-string 1))
;;                (imports-text (match-string 2))
;;                (imports (split-string imports-text "[,\n[:space:]]+" t))
;;                (start (match-beginning 0))
;;                (end (match-end 0))
;;                (original-lines (count-lines start end)))
;;           (message "Found module: %s" module)
;;           (message "Imports text: %s" imports-text)
;;           (message "Split imports: %s" imports)
;;           (message "Region: %d to %d" start end)
;;           (message "Original lines: %d" original-lines)
;;           ;; Store the split lines
;;           (let ((new-imports
;;                  (mapconcat
;;                   (lambda (imp)
;;                     (format "from %s import %s" module imp))
;;                   (sort imports #'string<)
;;                   "\n")))
;;             (message "New imports:\n%s" new-imports)
;;             ;; Remove original
;;             (delete-region start end)
;;             ;; Insert new format
;;             (goto-char start)
;;             (insert new-imports))))
;;       (goto-char original-point))))




;; (defun pim--split-imports ()
;;   "Split multi-line import statements into individual lines."
;;   (interactive)
;;   (let ((original-content (buffer-string))
;;         (temp-buffer (generate-new-buffer "*pim-temp*")))
;;     (with-current-buffer temp-buffer
;;       (insert original-content)
;;       (goto-char (point-min))
;;       (while (re-search-forward "^\\(from \\([^ ]+\\) import *(\n?\\([^)]+\\))\n?\\)" nil t)
;;         (let* ((module (match-string 2))
;;                (imports-text (match-string 3))
;;                (imports (split-string imports-text "[,\n[:space:]]+" t))
;;                (start (match-beginning 0))
;;                (end (match-end 0)))
;;           (delete-region start end)
;;           (goto-char start)
;;           (dolist (imp imports)
;;             (insert (format "from %s import %s\n"
;;                           module (string-trim imp))))))
;;       (let ((new-content (buffer-string)))
;;         (with-current-buffer (current-buffer)
;;           (erase-buffer)
;;           (insert new-content))))
;;     (kill-buffer temp-buffer)))
;; ;; Let's break down into pieces

;; 1. pim--find-multiple-imports-one-line (note the comma)
;; 2. pim--find-multiple-imports-multiple-line (using parentheses and commas)
;; 3. pim--find-multiple-imports-to-temp-buffer



;; for single line: "^from .*, .*[^()]$"
(defun pim--find-multiple-imports-one-line ()
  "Find and split imports on a single line separated by commas."
  (interactive)
  (let ((temp-buffer (generate-new-buffer "*pim-single-line-imports*")))
    (save-excursion
      (goto-char (point-min))
      (while (re-search-forward "^from [^(\n]+ import +[^(\n,]+ *, *[^(\n]+$" nil t)
        (let ((matched-text (buffer-substring-no-properties
                            (line-beginning-position)
                            (line-end-position))))
          (with-current-buffer temp-buffer
            (insert matched-text "\n")))))
    ;; (display-buffer temp-buffer)
    ))

(defun pim--split-imports-one-line ()
  "Split multiple imports into separate lines."
  (interactive)
  (save-excursion
    (goto-char (point-min))
    (while (re-search-forward "^from \\([^(\n]+\\) import +\\([^(\n,]+\\) *, *\\([^(\n]+\\)$" nil t)
      (let* ((module (match-string 1))
             (import1 (match-string 2))
             (import2 (match-string 3))
             (replacement (format "from %s import %s\nfrom %s import %s"
                                module import1 module import2)))
        (replace-match replacement)))))

;; for multiple lines: "^from [^\n]+ import[[:space:]]*(\n?[[:space:]]*[^()]+,?\n?)*)"
(defun pim--find-multiple-imports-to-temp-buffer ()
  "Find multiple imports and write them to a temporary buffer."
  (interactive)
  (let ((temp-buffer (generate-new-buffer "*pim-multiple-imports*")))
    (save-excursion
      (goto-char (point-min))
      (while (re-search-forward "^from [^\n]+ import[[:space:]]*(\n?[[:space:]]*[^()]+,?\n?)*)" nil t)
        (let* ((start (match-beginning 0))
               (end (match-end 0))
               (matched-text (buffer-substring-no-properties start end)))
          (with-current-buffer temp-buffer
            (insert matched-text "\n")))))
    ;; (display-buffer temp-buffer)
    ))

(defun pim--extract-module-and-imports (import-statement)
  "Extract module and imports from a multiline import statement."
  (when (string-match "^from \\([^(\n]+\\) import *( *\n?\\([^)]+\\)" import-statement)
    (let ((module (match-string 1 import-statement))
          (imports (match-string 2 import-statement)))
      (cons module
            (split-string imports ",[\n \t]*" t "[ \t\n]+")))))

(defun pim--format-imports (module-imports)
  "Format module and imports into separate lines."
  (let ((module (car module-imports))
        (imports (cdr module-imports)))
    (mapconcat
     (lambda (imp) (format "from %s import %s" module imp))
     imports "\n")))

(defun pim--split-imports-multiline ()
  "Split multiline imports into separate lines."
  (interactive)
  (let ((imports (pim--find-multiple-imports-to-temp-buffer))
        (count 0)
        (limit 100))
    (save-excursion
      (goto-char (point-min))
      (while (and (< count limit)
                  (re-search-forward "^from \\([^(\n]+\\) import *( *\n?\\([^)]+\\)[[:space:]]*)" nil t))
        (message "Processing import #%d at position %d" count (point))
        (let ((start (match-beginning 0))
              (end (match-end 0)))
          (when-let* ((module-imports (pim--extract-module-and-imports
                                     (match-string 0)))
                      (replacement (pim--format-imports module-imports)))
            (delete-region start end)
            (goto-char start)
            (insert replacement)
            (goto-char (+ start (length replacement)))))
        (setq count (1+ count))))))

(defun pim--split-imports ()
  (interactive)
  (pim--split-imports-one-line)
  (pim--split-imports-multiline)
  )

;; (defun pim--find-multiple-imports-to-temp-buffer ()
;;   "Find multiple imports and write them to a temporary buffer."
;;   (interactive)
;;   (let ((temp-buffer (generate-new-buffer "*pim-multiple-imports*")))
;;     (save-excursion
;;       (goto-char (point-min))
;;       (while (re-search-forward "^from [^(\n]+ import (\n\\([^)]+\\)\n" nil t)
;;         (let* ((start (match-beginning 0))
;;                (end (save-excursion
;;                      (goto-char (+ (match-end 0) 1))
;;                      (re-search-forward ")" nil t)
;;                      (point))
;;                     )
;;                (matched-text (buffer-substring-no-properties start end)))
;;           (with-current-buffer temp-buffer
;;             (insert matched-text "\n"))
;;           (goto-char start)
;;           (insert "# ")
;;           (while (< (point) end)
;;             (forward-line)
;;             (when (< (point) end)
;;               (insert "# ")))))
;;         ))
;;     (display-buffer temp-buffer)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Collection
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; (defun pim--collect-all-imports ()
;;   "Collect all import statements from the buffer."
;;   (let (imports)
;;     (save-excursion
;;       (goto-char (point-min))
;;       (while (re-search-forward "^\\(from\\|import\\) .*$" nil t)
;;         (push (match-string-no-properties 0) imports)
;;         (kill-whole-line)))
;;     (nreverse imports)))

(defun pim--collect-all-imports ()
  "Collect all import statements from the buffer."
  (let (imports)
    (save-excursion
      (goto-char (point-min))
      (while (re-search-forward "^\\(from\\|import\\) .*$" nil t)
        (push (match-string-no-properties 0) imports)))
    (nreverse imports)))

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; All-in-One function
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun pim-fix-imports ()
  "Fix imports in current buffer."
  (interactive)
  (message "pim-fix-imports called")
  (let ((original-point (point)))
    (pim--update-import-list)
    (pim--split-imports)
    (pim-delete-unused)
    (pim-delete-duplicated)
    (pim-insert-missed)
    (python-isort-buffer)
    (blacken-buffer)
    (goto-char original-point)
    ))


;;;###autoload
(defalias 'pim 'pim-fix-imports)

(provide 'python-import-manager)

;;; python-import-manager.el ends here


(message "%s was loaded." (file-name-nondirectory (or load-file-name buffer-file-name)))
