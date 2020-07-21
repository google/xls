; The following SMT-LIB verifies that a 32-bit adder is equivalent
; to CVC4's built in bit-vector addition.

(set-logic ALL)

; Declare bit-vectors and proxies for indices
(declare-fun x () (_ BitVec 32))
(declare-fun y () (_ BitVec 32))
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
(define-fun x8 () (_ BitVec 1) ((_ extract 8 8) x))
(define-fun y8 () (_ BitVec 1) ((_ extract 8 8) y))
(define-fun x9 () (_ BitVec 1) ((_ extract 9 9) x))
(define-fun y9 () (_ BitVec 1) ((_ extract 9 9) y))
(define-fun x10 () (_ BitVec 1) ((_ extract 10 10) x))
(define-fun y10 () (_ BitVec 1) ((_ extract 10 10) y))
(define-fun x11 () (_ BitVec 1) ((_ extract 11 11) x))
(define-fun y11 () (_ BitVec 1) ((_ extract 11 11) y))
(define-fun x12 () (_ BitVec 1) ((_ extract 12 12) x))
(define-fun y12 () (_ BitVec 1) ((_ extract 12 12) y))
(define-fun x13 () (_ BitVec 1) ((_ extract 13 13) x))
(define-fun y13 () (_ BitVec 1) ((_ extract 13 13) y))
(define-fun x14 () (_ BitVec 1) ((_ extract 14 14) x))
(define-fun y14 () (_ BitVec 1) ((_ extract 14 14) y))
(define-fun x15 () (_ BitVec 1) ((_ extract 15 15) x))
(define-fun y15 () (_ BitVec 1) ((_ extract 15 15) y))
(define-fun x16 () (_ BitVec 1) ((_ extract 16 16) x))
(define-fun y16 () (_ BitVec 1) ((_ extract 16 16) y))
(define-fun x17 () (_ BitVec 1) ((_ extract 17 17) x))
(define-fun y17 () (_ BitVec 1) ((_ extract 17 17) y))
(define-fun x18 () (_ BitVec 1) ((_ extract 18 18) x))
(define-fun y18 () (_ BitVec 1) ((_ extract 18 18) y))
(define-fun x19 () (_ BitVec 1) ((_ extract 19 19) x))
(define-fun y19 () (_ BitVec 1) ((_ extract 19 19) y))
(define-fun x20 () (_ BitVec 1) ((_ extract 20 20) x))
(define-fun y20 () (_ BitVec 1) ((_ extract 20 20) y))
(define-fun x21 () (_ BitVec 1) ((_ extract 21 21) x))
(define-fun y21 () (_ BitVec 1) ((_ extract 21 21) y))
(define-fun x22 () (_ BitVec 1) ((_ extract 22 22) x))
(define-fun y22 () (_ BitVec 1) ((_ extract 22 22) y))
(define-fun x23 () (_ BitVec 1) ((_ extract 23 23) x))
(define-fun y23 () (_ BitVec 1) ((_ extract 23 23) y))
(define-fun x24 () (_ BitVec 1) ((_ extract 24 24) x))
(define-fun y24 () (_ BitVec 1) ((_ extract 24 24) y))
(define-fun x25 () (_ BitVec 1) ((_ extract 25 25) x))
(define-fun y25 () (_ BitVec 1) ((_ extract 25 25) y))
(define-fun x26 () (_ BitVec 1) ((_ extract 26 26) x))
(define-fun y26 () (_ BitVec 1) ((_ extract 26 26) y))
(define-fun x27 () (_ BitVec 1) ((_ extract 27 27) x))
(define-fun y27 () (_ BitVec 1) ((_ extract 27 27) y))
(define-fun x28 () (_ BitVec 1) ((_ extract 28 28) x))
(define-fun y28 () (_ BitVec 1) ((_ extract 28 28) y))
(define-fun x29 () (_ BitVec 1) ((_ extract 29 29) x))
(define-fun y29 () (_ BitVec 1) ((_ extract 29 29) y))
(define-fun x30 () (_ BitVec 1) ((_ extract 30 30) x))
(define-fun y30 () (_ BitVec 1) ((_ extract 30 30) y))
(define-fun x31 () (_ BitVec 1) ((_ extract 31 31) x))
(define-fun y31 () (_ BitVec 1) ((_ extract 31 31) y))

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

; Full adder for bit 8
(define-fun s8 () (_ BitVec 1) (bvxor c7 (bvxor x8 y8)))
(define-fun c8 () (_ BitVec 1) (bvor (bvand (bvxor x8 y8) c7) (bvand x8 y8)))

; Full adder for bit 9
(define-fun s9 () (_ BitVec 1) (bvxor c8 (bvxor x9 y9)))
(define-fun c9 () (_ BitVec 1) (bvor (bvand (bvxor x9 y9) c8) (bvand x9 y9)))

; Full adder for bit 10
(define-fun s10 () (_ BitVec 1) (bvxor c9 (bvxor x10 y10)))
(define-fun c10 () (_ BitVec 1) (bvor (bvand (bvxor x10 y10) c9) (bvand x10 y10)))

; Full adder for bit 11
(define-fun s11 () (_ BitVec 1) (bvxor c10 (bvxor x11 y11)))
(define-fun c11 () (_ BitVec 1) (bvor (bvand (bvxor x11 y11) c10) (bvand x11 y11)))

; Full adder for bit 12
(define-fun s12 () (_ BitVec 1) (bvxor c11 (bvxor x12 y12)))
(define-fun c12 () (_ BitVec 1) (bvor (bvand (bvxor x12 y12) c11) (bvand x12 y12)))

; Full adder for bit 13
(define-fun s13 () (_ BitVec 1) (bvxor c12 (bvxor x13 y13)))
(define-fun c13 () (_ BitVec 1) (bvor (bvand (bvxor x13 y13) c12) (bvand x13 y13)))

; Full adder for bit 14
(define-fun s14 () (_ BitVec 1) (bvxor c13 (bvxor x14 y14)))
(define-fun c14 () (_ BitVec 1) (bvor (bvand (bvxor x14 y14) c13) (bvand x14 y14)))

; Full adder for bit 15
(define-fun s15 () (_ BitVec 1) (bvxor c14 (bvxor x15 y15)))
(define-fun c15 () (_ BitVec 1) (bvor (bvand (bvxor x15 y15) c14) (bvand x15 y15)))

; Full adder for bit 16
(define-fun s16 () (_ BitVec 1) (bvxor c15 (bvxor x16 y16)))
(define-fun c16 () (_ BitVec 1) (bvor (bvand (bvxor x16 y16) c15) (bvand x16 y16)))

; Full adder for bit 17
(define-fun s17 () (_ BitVec 1) (bvxor c16 (bvxor x17 y17)))
(define-fun c17 () (_ BitVec 1) (bvor (bvand (bvxor x17 y17) c16) (bvand x17 y17)))

; Full adder for bit 18
(define-fun s18 () (_ BitVec 1) (bvxor c17 (bvxor x18 y18)))
(define-fun c18 () (_ BitVec 1) (bvor (bvand (bvxor x18 y18) c17) (bvand x18 y18)))

; Full adder for bit 19
(define-fun s19 () (_ BitVec 1) (bvxor c18 (bvxor x19 y19)))
(define-fun c19 () (_ BitVec 1) (bvor (bvand (bvxor x19 y19) c18) (bvand x19 y19)))

; Full adder for bit 20
(define-fun s20 () (_ BitVec 1) (bvxor c19 (bvxor x20 y20)))
(define-fun c20 () (_ BitVec 1) (bvor (bvand (bvxor x20 y20) c19) (bvand x20 y20)))

; Full adder for bit 21
(define-fun s21 () (_ BitVec 1) (bvxor c20 (bvxor x21 y21)))
(define-fun c21 () (_ BitVec 1) (bvor (bvand (bvxor x21 y21) c20) (bvand x21 y21)))

; Full adder for bit 22
(define-fun s22 () (_ BitVec 1) (bvxor c21 (bvxor x22 y22)))
(define-fun c22 () (_ BitVec 1) (bvor (bvand (bvxor x22 y22) c21) (bvand x22 y22)))

; Full adder for bit 23
(define-fun s23 () (_ BitVec 1) (bvxor c22 (bvxor x23 y23)))
(define-fun c23 () (_ BitVec 1) (bvor (bvand (bvxor x23 y23) c22) (bvand x23 y23)))

; Full adder for bit 24
(define-fun s24 () (_ BitVec 1) (bvxor c23 (bvxor x24 y24)))
(define-fun c24 () (_ BitVec 1) (bvor (bvand (bvxor x24 y24) c23) (bvand x24 y24)))

; Full adder for bit 25
(define-fun s25 () (_ BitVec 1) (bvxor c24 (bvxor x25 y25)))
(define-fun c25 () (_ BitVec 1) (bvor (bvand (bvxor x25 y25) c24) (bvand x25 y25)))

; Full adder for bit 26
(define-fun s26 () (_ BitVec 1) (bvxor c25 (bvxor x26 y26)))
(define-fun c26 () (_ BitVec 1) (bvor (bvand (bvxor x26 y26) c25) (bvand x26 y26)))

; Full adder for bit 27
(define-fun s27 () (_ BitVec 1) (bvxor c26 (bvxor x27 y27)))
(define-fun c27 () (_ BitVec 1) (bvor (bvand (bvxor x27 y27) c26) (bvand x27 y27)))

; Full adder for bit 28
(define-fun s28 () (_ BitVec 1) (bvxor c27 (bvxor x28 y28)))
(define-fun c28 () (_ BitVec 1) (bvor (bvand (bvxor x28 y28) c27) (bvand x28 y28)))

; Full adder for bit 29
(define-fun s29 () (_ BitVec 1) (bvxor c28 (bvxor x29 y29)))
(define-fun c29 () (_ BitVec 1) (bvor (bvand (bvxor x29 y29) c28) (bvand x29 y29)))

; Full adder for bit 30
(define-fun s30 () (_ BitVec 1) (bvxor c29 (bvxor x30 y30)))
(define-fun c30 () (_ BitVec 1) (bvor (bvand (bvxor x30 y30) c29) (bvand x30 y30)))

; Full adder for bit 31
(define-fun s31 () (_ BitVec 1) (bvxor c30 (bvxor x31 y31)))
(define-fun c31 () (_ BitVec 1) (bvor (bvand (bvxor x31 y31) c30) (bvand x31 y31)))

; Concatenate s bits to create sum
(define-fun sum () (_ BitVec 32) (concat s31 s30 s29 s28 s27 s26 s25 s24 s23 s22 s21 s20 s19 s18 s17 s16 s15 s14 s13 s12 s11 s10 s9 s8 s7 s6 s5 s4 s3 s2 s1 s0))

; Compare 32-bit adder result and internal addition and solve
(assert (not (= sum (bvadd x y))))
(check-sat)