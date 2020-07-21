; The following SMT-LIB verifies that a 64-bit adder is equivalent
; to CVC4's built in bit-vector addition.

(set-logic ALL)

; Declare bit-vectors and proxies for indices
(declare-fun x () (_ BitVec 64))
(declare-fun y () (_ BitVec 64))
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
(define-fun x32 () (_ BitVec 1) ((_ extract 32 32) x))
(define-fun y32 () (_ BitVec 1) ((_ extract 32 32) y))
(define-fun x33 () (_ BitVec 1) ((_ extract 33 33) x))
(define-fun y33 () (_ BitVec 1) ((_ extract 33 33) y))
(define-fun x34 () (_ BitVec 1) ((_ extract 34 34) x))
(define-fun y34 () (_ BitVec 1) ((_ extract 34 34) y))
(define-fun x35 () (_ BitVec 1) ((_ extract 35 35) x))
(define-fun y35 () (_ BitVec 1) ((_ extract 35 35) y))
(define-fun x36 () (_ BitVec 1) ((_ extract 36 36) x))
(define-fun y36 () (_ BitVec 1) ((_ extract 36 36) y))
(define-fun x37 () (_ BitVec 1) ((_ extract 37 37) x))
(define-fun y37 () (_ BitVec 1) ((_ extract 37 37) y))
(define-fun x38 () (_ BitVec 1) ((_ extract 38 38) x))
(define-fun y38 () (_ BitVec 1) ((_ extract 38 38) y))
(define-fun x39 () (_ BitVec 1) ((_ extract 39 39) x))
(define-fun y39 () (_ BitVec 1) ((_ extract 39 39) y))
(define-fun x40 () (_ BitVec 1) ((_ extract 40 40) x))
(define-fun y40 () (_ BitVec 1) ((_ extract 40 40) y))
(define-fun x41 () (_ BitVec 1) ((_ extract 41 41) x))
(define-fun y41 () (_ BitVec 1) ((_ extract 41 41) y))
(define-fun x42 () (_ BitVec 1) ((_ extract 42 42) x))
(define-fun y42 () (_ BitVec 1) ((_ extract 42 42) y))
(define-fun x43 () (_ BitVec 1) ((_ extract 43 43) x))
(define-fun y43 () (_ BitVec 1) ((_ extract 43 43) y))
(define-fun x44 () (_ BitVec 1) ((_ extract 44 44) x))
(define-fun y44 () (_ BitVec 1) ((_ extract 44 44) y))
(define-fun x45 () (_ BitVec 1) ((_ extract 45 45) x))
(define-fun y45 () (_ BitVec 1) ((_ extract 45 45) y))
(define-fun x46 () (_ BitVec 1) ((_ extract 46 46) x))
(define-fun y46 () (_ BitVec 1) ((_ extract 46 46) y))
(define-fun x47 () (_ BitVec 1) ((_ extract 47 47) x))
(define-fun y47 () (_ BitVec 1) ((_ extract 47 47) y))
(define-fun x48 () (_ BitVec 1) ((_ extract 48 48) x))
(define-fun y48 () (_ BitVec 1) ((_ extract 48 48) y))
(define-fun x49 () (_ BitVec 1) ((_ extract 49 49) x))
(define-fun y49 () (_ BitVec 1) ((_ extract 49 49) y))
(define-fun x50 () (_ BitVec 1) ((_ extract 50 50) x))
(define-fun y50 () (_ BitVec 1) ((_ extract 50 50) y))
(define-fun x51 () (_ BitVec 1) ((_ extract 51 51) x))
(define-fun y51 () (_ BitVec 1) ((_ extract 51 51) y))
(define-fun x52 () (_ BitVec 1) ((_ extract 52 52) x))
(define-fun y52 () (_ BitVec 1) ((_ extract 52 52) y))
(define-fun x53 () (_ BitVec 1) ((_ extract 53 53) x))
(define-fun y53 () (_ BitVec 1) ((_ extract 53 53) y))
(define-fun x54 () (_ BitVec 1) ((_ extract 54 54) x))
(define-fun y54 () (_ BitVec 1) ((_ extract 54 54) y))
(define-fun x55 () (_ BitVec 1) ((_ extract 55 55) x))
(define-fun y55 () (_ BitVec 1) ((_ extract 55 55) y))
(define-fun x56 () (_ BitVec 1) ((_ extract 56 56) x))
(define-fun y56 () (_ BitVec 1) ((_ extract 56 56) y))
(define-fun x57 () (_ BitVec 1) ((_ extract 57 57) x))
(define-fun y57 () (_ BitVec 1) ((_ extract 57 57) y))
(define-fun x58 () (_ BitVec 1) ((_ extract 58 58) x))
(define-fun y58 () (_ BitVec 1) ((_ extract 58 58) y))
(define-fun x59 () (_ BitVec 1) ((_ extract 59 59) x))
(define-fun y59 () (_ BitVec 1) ((_ extract 59 59) y))
(define-fun x60 () (_ BitVec 1) ((_ extract 60 60) x))
(define-fun y60 () (_ BitVec 1) ((_ extract 60 60) y))
(define-fun x61 () (_ BitVec 1) ((_ extract 61 61) x))
(define-fun y61 () (_ BitVec 1) ((_ extract 61 61) y))
(define-fun x62 () (_ BitVec 1) ((_ extract 62 62) x))
(define-fun y62 () (_ BitVec 1) ((_ extract 62 62) y))
(define-fun x63 () (_ BitVec 1) ((_ extract 63 63) x))
(define-fun y63 () (_ BitVec 1) ((_ extract 63 63) y))

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

; Full adder for bit 32
(define-fun s32 () (_ BitVec 1) (bvxor c31 (bvxor x32 y32)))
(define-fun c32 () (_ BitVec 1) (bvor (bvand (bvxor x32 y32) c31) (bvand x32 y32)))

; Full adder for bit 33
(define-fun s33 () (_ BitVec 1) (bvxor c32 (bvxor x33 y33)))
(define-fun c33 () (_ BitVec 1) (bvor (bvand (bvxor x33 y33) c32) (bvand x33 y33)))

; Full adder for bit 34
(define-fun s34 () (_ BitVec 1) (bvxor c33 (bvxor x34 y34)))
(define-fun c34 () (_ BitVec 1) (bvor (bvand (bvxor x34 y34) c33) (bvand x34 y34)))

; Full adder for bit 35
(define-fun s35 () (_ BitVec 1) (bvxor c34 (bvxor x35 y35)))
(define-fun c35 () (_ BitVec 1) (bvor (bvand (bvxor x35 y35) c34) (bvand x35 y35)))

; Full adder for bit 36
(define-fun s36 () (_ BitVec 1) (bvxor c35 (bvxor x36 y36)))
(define-fun c36 () (_ BitVec 1) (bvor (bvand (bvxor x36 y36) c35) (bvand x36 y36)))

; Full adder for bit 37
(define-fun s37 () (_ BitVec 1) (bvxor c36 (bvxor x37 y37)))
(define-fun c37 () (_ BitVec 1) (bvor (bvand (bvxor x37 y37) c36) (bvand x37 y37)))

; Full adder for bit 38
(define-fun s38 () (_ BitVec 1) (bvxor c37 (bvxor x38 y38)))
(define-fun c38 () (_ BitVec 1) (bvor (bvand (bvxor x38 y38) c37) (bvand x38 y38)))

; Full adder for bit 39
(define-fun s39 () (_ BitVec 1) (bvxor c38 (bvxor x39 y39)))
(define-fun c39 () (_ BitVec 1) (bvor (bvand (bvxor x39 y39) c38) (bvand x39 y39)))

; Full adder for bit 40
(define-fun s40 () (_ BitVec 1) (bvxor c39 (bvxor x40 y40)))
(define-fun c40 () (_ BitVec 1) (bvor (bvand (bvxor x40 y40) c39) (bvand x40 y40)))

; Full adder for bit 41
(define-fun s41 () (_ BitVec 1) (bvxor c40 (bvxor x41 y41)))
(define-fun c41 () (_ BitVec 1) (bvor (bvand (bvxor x41 y41) c40) (bvand x41 y41)))

; Full adder for bit 42
(define-fun s42 () (_ BitVec 1) (bvxor c41 (bvxor x42 y42)))
(define-fun c42 () (_ BitVec 1) (bvor (bvand (bvxor x42 y42) c41) (bvand x42 y42)))

; Full adder for bit 43
(define-fun s43 () (_ BitVec 1) (bvxor c42 (bvxor x43 y43)))
(define-fun c43 () (_ BitVec 1) (bvor (bvand (bvxor x43 y43) c42) (bvand x43 y43)))

; Full adder for bit 44
(define-fun s44 () (_ BitVec 1) (bvxor c43 (bvxor x44 y44)))
(define-fun c44 () (_ BitVec 1) (bvor (bvand (bvxor x44 y44) c43) (bvand x44 y44)))

; Full adder for bit 45
(define-fun s45 () (_ BitVec 1) (bvxor c44 (bvxor x45 y45)))
(define-fun c45 () (_ BitVec 1) (bvor (bvand (bvxor x45 y45) c44) (bvand x45 y45)))

; Full adder for bit 46
(define-fun s46 () (_ BitVec 1) (bvxor c45 (bvxor x46 y46)))
(define-fun c46 () (_ BitVec 1) (bvor (bvand (bvxor x46 y46) c45) (bvand x46 y46)))

; Full adder for bit 47
(define-fun s47 () (_ BitVec 1) (bvxor c46 (bvxor x47 y47)))
(define-fun c47 () (_ BitVec 1) (bvor (bvand (bvxor x47 y47) c46) (bvand x47 y47)))

; Full adder for bit 48
(define-fun s48 () (_ BitVec 1) (bvxor c47 (bvxor x48 y48)))
(define-fun c48 () (_ BitVec 1) (bvor (bvand (bvxor x48 y48) c47) (bvand x48 y48)))

; Full adder for bit 49
(define-fun s49 () (_ BitVec 1) (bvxor c48 (bvxor x49 y49)))
(define-fun c49 () (_ BitVec 1) (bvor (bvand (bvxor x49 y49) c48) (bvand x49 y49)))

; Full adder for bit 50
(define-fun s50 () (_ BitVec 1) (bvxor c49 (bvxor x50 y50)))
(define-fun c50 () (_ BitVec 1) (bvor (bvand (bvxor x50 y50) c49) (bvand x50 y50)))

; Full adder for bit 51
(define-fun s51 () (_ BitVec 1) (bvxor c50 (bvxor x51 y51)))
(define-fun c51 () (_ BitVec 1) (bvor (bvand (bvxor x51 y51) c50) (bvand x51 y51)))

; Full adder for bit 52
(define-fun s52 () (_ BitVec 1) (bvxor c51 (bvxor x52 y52)))
(define-fun c52 () (_ BitVec 1) (bvor (bvand (bvxor x52 y52) c51) (bvand x52 y52)))

; Full adder for bit 53
(define-fun s53 () (_ BitVec 1) (bvxor c52 (bvxor x53 y53)))
(define-fun c53 () (_ BitVec 1) (bvor (bvand (bvxor x53 y53) c52) (bvand x53 y53)))

; Full adder for bit 54
(define-fun s54 () (_ BitVec 1) (bvxor c53 (bvxor x54 y54)))
(define-fun c54 () (_ BitVec 1) (bvor (bvand (bvxor x54 y54) c53) (bvand x54 y54)))

; Full adder for bit 55
(define-fun s55 () (_ BitVec 1) (bvxor c54 (bvxor x55 y55)))
(define-fun c55 () (_ BitVec 1) (bvor (bvand (bvxor x55 y55) c54) (bvand x55 y55)))

; Full adder for bit 56
(define-fun s56 () (_ BitVec 1) (bvxor c55 (bvxor x56 y56)))
(define-fun c56 () (_ BitVec 1) (bvor (bvand (bvxor x56 y56) c55) (bvand x56 y56)))

; Full adder for bit 57
(define-fun s57 () (_ BitVec 1) (bvxor c56 (bvxor x57 y57)))
(define-fun c57 () (_ BitVec 1) (bvor (bvand (bvxor x57 y57) c56) (bvand x57 y57)))

; Full adder for bit 58
(define-fun s58 () (_ BitVec 1) (bvxor c57 (bvxor x58 y58)))
(define-fun c58 () (_ BitVec 1) (bvor (bvand (bvxor x58 y58) c57) (bvand x58 y58)))

; Full adder for bit 59
(define-fun s59 () (_ BitVec 1) (bvxor c58 (bvxor x59 y59)))
(define-fun c59 () (_ BitVec 1) (bvor (bvand (bvxor x59 y59) c58) (bvand x59 y59)))

; Full adder for bit 60
(define-fun s60 () (_ BitVec 1) (bvxor c59 (bvxor x60 y60)))
(define-fun c60 () (_ BitVec 1) (bvor (bvand (bvxor x60 y60) c59) (bvand x60 y60)))

; Full adder for bit 61
(define-fun s61 () (_ BitVec 1) (bvxor c60 (bvxor x61 y61)))
(define-fun c61 () (_ BitVec 1) (bvor (bvand (bvxor x61 y61) c60) (bvand x61 y61)))

; Full adder for bit 62
(define-fun s62 () (_ BitVec 1) (bvxor c61 (bvxor x62 y62)))
(define-fun c62 () (_ BitVec 1) (bvor (bvand (bvxor x62 y62) c61) (bvand x62 y62)))

; Full adder for bit 63
(define-fun s63 () (_ BitVec 1) (bvxor c62 (bvxor x63 y63)))
(define-fun c63 () (_ BitVec 1) (bvor (bvand (bvxor x63 y63) c62) (bvand x63 y63)))

; Concatenate s bits to create sum
(define-fun sum () (_ BitVec 64) (concat s63 s62 s61 s60 s59 s58 s57 s56 s55 s54 s53 s52 s51 s50 s49 s48 s47 s46 s45 s44 s43 s42 s41 s40 s39 s38 s37 s36 s35 s34 s33 s32 s31 s30 s29 s28 s27 s26 s25 s24 s23 s22 s21 s20 s19 s18 s17 s16 s15 s14 s13 s12 s11 s10 s9 s8 s7 s6 s5 s4 s3 s2 s1 s0))

; Compare 64-bit adder result and internal addition and solve
(assert (not (= sum (bvadd x y))))
(check-sat)