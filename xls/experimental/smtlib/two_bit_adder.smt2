; Instructions for using CVC4 can be found in cvc4.md.

; The following SMT-LIB verifies that a two-bit adder is equivalent
; to CVC4's built in bit-vector addition.

(set-logic ALL)

; Declare bit-vector variables and proxies for indices
(declare-fun x () (_ BitVec 2))
(declare-fun y () (_ BitVec 2))
(define-fun x0 () (_ BitVec 1) ((_ extract 0 0) x))
(define-fun x1 () (_ BitVec 1) ((_ extract 1 1) x))
(define-fun y0 () (_ BitVec 1) ((_ extract 0 0) y))
(define-fun y1 () (_ BitVec 1) ((_ extract 1 1) y))

; Half adder
(define-fun s0 () (_ BitVec 1) (bvxor x0 y0))
(define-fun c0 () (_ BitVec 1) (bvand x0 y0))

; Full adder
(define-fun s1 () (_ BitVec 1) (bvxor c0 (bvxor x1 y1)))
(define-fun c1 () (_ BitVec 1) (bvor (bvand (bvxor x1 y1) c0) (bvand x1 y1)))

; Create result
(define-fun out () (_ BitVec 2) (concat s1 s0))

; Solve
(assert (not (= out (bvadd x y))))
(check-sat)

