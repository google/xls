; The following SMT-LIB verifies that a 4-bit adder is equivalent
; to CVC4's built in bit-vector addition.

(set-logic ALL)

; Declare bit-vectors and proxies for indices
(declare-fun x () (_ BitVec 4))
(declare-fun y () (_ BitVec 4))
(define-fun x0 () (_ BitVec 1) ((_ extract 0 0) x))
(define-fun y0 () (_ BitVec 1) ((_ extract 0 0) y))
(define-fun x1 () (_ BitVec 1) ((_ extract 1 1) x))
(define-fun y1 () (_ BitVec 1) ((_ extract 1 1) y))
(define-fun x2 () (_ BitVec 1) ((_ extract 2 2) x))
(define-fun y2 () (_ BitVec 1) ((_ extract 2 2) y))
(define-fun x3 () (_ BitVec 1) ((_ extract 3 3) x))
(define-fun y3 () (_ BitVec 1) ((_ extract 3 3) y))

; Half adder for bit 0
(define-fun s0 () (_ BitVec 1) (bvxor x0 y0))
(define-fun c0 () (_ BitVec 1) (bvand x0 y0))

; Full adder for bit 1
(define-fun s1 () (_ BitVec 1) (bvxor c0 (bvxor x1 y1)))
(define-fun c1 () (_ BitVec 1) (bvor (bvand (bvxor x1 y1) c0) (bvand x1 y1)))

; Full adder for bit 2
(define-fun s2 () (_ BitVec 1) (bvxor c1 (bvxor x2 y2)))
(define-fun c2 () (_ BitVec 1) (bvor (bvand (bvxor x2 y2) c1) (bvand x2 y2)))

; Full adder for bit 3
(define-fun s3 () (_ BitVec 1) (bvxor c2 (bvxor x3 y3)))
(define-fun c3 () (_ BitVec 1) (bvor (bvand (bvxor x3 y3) c2) (bvand x3 y3)))

; Concatenate s bits to create sum
(define-fun sum () (_ BitVec 4) (concat s3 s2 s1 s0))

; Compare 4-bit adder result and internal addition and solve
(assert (not (= sum (bvadd x y))))
(check-sat)