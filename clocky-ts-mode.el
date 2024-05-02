;;; clocky-ts-mode.el --- major mode for clocky-lang, using tree-sitter

;;; Commentary:

;;; Code:

(defvar clocky-ts-font-lock-rules
  '(:language clocky
    :feature clock
    ((clock_coeff) @font-lock-number-face)

    :language clocky
    :feature clock
    ((clock (identifier) @font-lock-warning-face))

    :language clocky
    :feature keyword
    ((let_expression ["let" "in"] @font-lock-keyword-face))

    :language clocky
    :feature keyword
    ((unpair_expression ["let" "in"] @font-lock-keyword-face))

    :language clocky
    :feature keyword
    ((top_level_def "def" @font-lock-keyword-face))

    :language clocky
    :feature keyword
    ((forall_type "for" @font-lock-keyword-face))

    :language clocky
    :feature keyword
    ((case_expression ["case" @font-lock-keyword-face
                       ["inl" "inr"] @font-lock-builtin-face] ))

    :language clocky
    :feature keyword
    ((inl_expression "inl" @font-lock-builtin-face))

    :language clocky
    :feature keyword
    ((inr_expression "inr" @font-lock-builtin-face))
    ))

(defvar clocky-ts-indent-rules
  '((clocky
     ((match nil "let_expression" "bound") parent 2)
     ((match nil "unpair_expression" "bound") parent 2)
     ((match nil "let_expression" "body") parent 0)
     ((match nil "unpair_expression" "body") parent 0)
     ((match nil "pair_expression" nil 3 3) (nth-sibling 1) 0)
     ((node-is "expression") parent-bol 2)
     (no-node parent-bol 2))))

;;;###autoload
(define-derived-mode clocky-ts-mode prog-mode "Clocky"
  "Major mode for Clocky files."

  (setq-local font-lock-defaults nil)
  (when (treesit-ready-p 'clocky)
    (treesit-parser-create 'clocky)
    (clocky-ts-setup)))

(defun clocky-ts-setup ()
  "Set up treesit for clocky-ts-mode."

  (setq-local treesit-font-lock-settings
              (apply #'treesit-font-lock-rules clocky-ts-font-lock-rules))

  (setq-local treesit-simple-indent-rules clocky-ts-indent-rules)

  (setq-local treesit-font-lock-feature-list
              '((clock)
                (keyword)))

  (treesit-major-mode-setup))

(provide 'clocky-ts-mode)
;;; clocky-ts-mode.el ends here
