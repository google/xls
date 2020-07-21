; The following SMT-LIB verifies that a 8-bit adder is equivalent
; to CVC4's built in bit-vector addition.

(set-logic ALL)

; Declare bit-vectors and proxies for indices
(declare-fun x () (_ BitVec 8))
(declare-fun y () (_ BitVec 8))
(define-fun x0 () (_ BitVec 1) ((_ extract 0 0) x))
(define-fun y0 () (_ BitVec 1) ((_ extract 0 0) y))
(define-fun x1 () (_ BitVec 1) ((_ extract 1 1) x))
(define-fun y1 () (_ BitVec 1) ((_ extract 1 1) y))
(define-fun x2 () (_ BitVec 1) ((_ extract 2 2) x))
(define-fun y2 () (_ BitVec 1) ((_ extract 2 2) y))
(define-fun x3 () (_ BitVec 1) ((_ extract 3 3) x))
(define-fun y3 () (_ BitVec 1) ((_ extract 3 3) y))
(define-fun x4 () (_ BitVec 1) ((_ extract 4 4) x))
(define-fun y4 () (_ BitVec 1) ((_ extract 4 4) y))
(define-fun x5 () (_ BitVec 1) ((_ extract 5 5) x))
(define-fun y5 () (_ BitVec 1) ((_ extract 5 5) y))
(define-fun x6 () (_ BitVec 1) ((_ extract 6 6) x))
(define-fun y6 () (_ BitVec 1) ((_ extract 6 6) y))
(define-fun x7 () (_ BitVec 1) ((_ extract 7 7) x))
(define-fun y7 () (_ BitVec 1) ((_ extract 7 7) y))

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

; Full adder for bit 4
(define-fun s4 () (_ BitVec 1) (bvxor c3 (bvxor x4 y4)))
(define-fun c4 () (_ BitVec 1) (bvor (bvand (bvxor x4 y4) c3) (bvand x4 y4)))

; Full adder for bit 5
(define-fun s5 () (_ BitVec 1) (bvxor c4 (bvxor x5 y5)))
(define-fun c5 () (_ BitVec 1) (bvor (bvand (bvxor x5 y5) c4) (bvand x5 y5)))

; Full adder for bit 6
(define-fun s6 () (_ BitVec 1) (bvxor c5 (bvxor x6 y6)))
(define-fun c6 () (_ BitVec 1) (bvor (bvand (bvxor x6 y6) c5) (bvand x6 y6)))

; Full adder for bit 7
(define-fun s7 () (_ BitVec 1) (bvxor c6 (bvxor x7 y7)))
(define-fun c7 () (_ BitVec 1) (bvor (bvand (bvxor x7 y7) c6) (bvand x7 y7)))

; Concatenate s bits to create sum
(define-fun sum () (_ BitVec 8) (concat s7 s6 s5 s4 s3 s2 s1 s0))

; Compare 8-bit adder result and internal addition and solve
(assert (not (= sum (bvadd x y))))
(check-sat)