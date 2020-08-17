; The following SMT-LIB verifies that a 2-bit multiplier is equivalent
; to CVC4's built in bit-vector multiplication.

(set-logic ALL)

; Declare bit-vectors and proxies for indices
(declare-fun x () (_ BitVec 2))
(declare-fun y () (_ BitVec 2))
(define-fun x0 () (_ BitVec 1) ((_ extract 0 0) x))
(define-fun y0 () (_ BitVec 1) ((_ extract 0 0) y))
(define-fun x1 () (_ BitVec 1) ((_ extract 1 1) x))
(define-fun y1 () (_ BitVec 1) ((_ extract 1 1) y))

; Multiply x by y0, shifting x bits accordingly
(define-fun m0_0 () (_ BitVec 1) (bvand x0 y0))
(define-fun m0_1 () (_ BitVec 1) (bvand x1 y0))
(define-fun m0 () (_ BitVec 2) (concat m0_1 m0_0))

; Multiply x by y1, shifting x bits accordingly
(define-fun m1_1 () (_ BitVec 1) (bvand x0 y1))
(define-fun m1 () (_ BitVec 2) (concat m1_1 #b0))

; Add all m bit-vectors to create mul
(define-fun mul () (_ BitVec 2) (bvadd m1 m0))

; Assert and solve
(assert (not (= mul (bvmul x y))))
(check-sat)
