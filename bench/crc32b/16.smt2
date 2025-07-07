(set-info :controlled input1_8)
(declare-fun bs_unknown1_for_esp_32__4 ()
(_ BitVec 32))
(assert; assume
(bvuge bs_unknown1_for_esp_32__4 (_ bv4294901760 32)))
(assert; assume
(bvule bs_unknown1_for_esp_32__4 (_ bv4294967040 32)))
(declare-fun input1_8 ()
(_ BitVec 16))
(declare-fun input2_11 ()
(_ BitVec 16))
(define-fun
res32_59 () (_ BitVec 32)
(bvnot ((_ zero_extend 24) ((_ extract 7 0) input1_8))))
(define-fun
res32_100 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_59 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_59 (_ bv1 32))))
(define-fun
res32_149 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_100 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_100 (_ bv1 32))))
(define-fun
res32_198 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_149 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_149 (_ bv1 32))))
(define-fun
res32_247 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_198 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_198 (_ bv1 32))))
(define-fun
res32_296 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_247 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_247 (_ bv1 32))))
(define-fun
res32_345 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_296 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_296 (_ bv1 32))))
(define-fun
res32_394 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_345 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_345 (_ bv1 32))))
(define-fun
res32_528 () (_ BitVec 32)
(bvnot ((_ zero_extend 24) ((_ extract 7 0) input2_11))))
(define-fun
res32_569 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_528 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_528 (_ bv1 32))))
(define-fun
res32_618 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_569 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_569 (_ bv1 32))))
(define-fun
res32_667 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_618 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_618 (_ bv1 32))))
(define-fun
res32_716 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_667 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_667 (_ bv1 32))))
(define-fun
res32_765 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_716 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_716 (_ bv1 32))))
(define-fun
res32_814 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_765 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_765 (_ bv1 32))))
(define-fun
res32_863 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_814 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_814 (_ bv1 32))))
(assert
(=
((_ zero_extend 24)
(ite
(=
(bvxor (bvand (bvneg (bvand res32_394 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_394 (_ bv1 32)))
(bvxor (bvand (bvneg (bvand res32_863 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_863 (_ bv1 32)))) (_ bv1 8) (_ bv0 8))) (_ bv1 32)))