; The following SMT-LIB verifies that a 5-bit multiplier is equivalent
; to CVC4's built in bit-vector multiplication.

(set-logic ALL)

; Declare bit-vectors and proxies for indices
(declare-fun x () (_ BitVec 5))
(declare-fun y () (_ BitVec 5))
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

; Multiply x by y0, shifting x bits accordingly
(define-fun m0_0 () (_ BitVec 1) (bvand x0 y0))
(define-fun m0_1 () (_ BitVec 1) (bvand x1 y0))
(define-fun m0_2 () (_ BitVec 1) (bvand x2 y0))
(define-fun m0_3 () (_ BitVec 1) (bvand x3 y0))
(define-fun m0_4 () (_ BitVec 1) (bvand x4 y0))
(define-fun m0 () (_ BitVec 5) (concat m0_4 m0_3 m0_2 m0_1 m0_0))

; Multiply x by y1, shifting x bits accordingly
(define-fun m1_1 () (_ BitVec 1) (bvand x0 y1))
(define-fun m1_2 () (_ BitVec 1) (bvand x1 y1))
(define-fun m1_3 () (_ BitVec 1) (bvand x2 y1))
(define-fun m1_4 () (_ BitVec 1) (bvand x3 y1))
(define-fun m1 () (_ BitVec 5) (concat m1_4 m1_3 m1_2 m1_1 #b0))

; Multiply x by y2, shifting x bits accordingly
(define-fun m2_2 () (_ BitVec 1) (bvand x0 y2))
(define-fun m2_3 () (_ BitVec 1) (bvand x1 y2))
(define-fun m2_4 () (_ BitVec 1) (bvand x2 y2))
(define-fun m2 () (_ BitVec 5) (concat m2_4 m2_3 m2_2 #b00))

; Multiply x by y3, shifting x bits accordingly
(define-fun m3_3 () (_ BitVec 1) (bvand x0 y3))
(define-fun m3_4 () (_ BitVec 1) (bvand x1 y3))
(define-fun m3 () (_ BitVec 5) (concat m3_4 m3_3 #b000))

; Multiply x by y4, shifting x bits accordingly
(define-fun m4_4 () (_ BitVec 1) (bvand x0 y4))
(define-fun m4 () (_ BitVec 5) (concat m4_4 #b0000))

; Add all m bit-vectors to create mul
(define-fun mul () (_ BitVec 5) (bvadd m4 m3 m2 m1 m0))

; Assert and solve
(assert (not (= mul (bvmul x y))))
(check-sat)
