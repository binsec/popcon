(declare-fun bs_unknown1_for_esp_32__4 ()
(_ BitVec 32))
(assert; assume
(bvuge bs_unknown1_for_esp_32__4 (_ bv4294901760 32)))
(assert; assume
(bvule bs_unknown1_for_esp_32__4 (_ bv4294967040 32)))
(declare-fun input1_9 ()
(_ BitVec 8))
(declare-fun input2_12 ()
(_ BitVec 8))
(define-fun
res32_60 () (_ BitVec 32) (bvnot ((_ zero_extend 24) input1_9)))
(define-fun
res32_101 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_60 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_60 (_ bv1 32))))
(define-fun
res32_150 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_101 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_101 (_ bv1 32))))
(define-fun
res32_199 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_150 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_150 (_ bv1 32))))
(define-fun
res32_248 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_199 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_199 (_ bv1 32))))
(define-fun
res32_297 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_248 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_248 (_ bv1 32))))
(define-fun
res32_346 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_297 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_297 (_ bv1 32))))
(define-fun
res32_395 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_346 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_346 (_ bv1 32))))
(define-fun
res32_529 () (_ BitVec 32) (bvnot ((_ zero_extend 24) input2_12)))
(define-fun
res32_570 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_529 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_529 (_ bv1 32))))
(define-fun
res32_619 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_570 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_570 (_ bv1 32))))
(define-fun
res32_668 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_619 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_619 (_ bv1 32))))
(define-fun
res32_717 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_668 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_668 (_ bv1 32))))
(define-fun
res32_766 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_717 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_717 (_ bv1 32))))
(define-fun
res32_815 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_766 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_766 (_ bv1 32))))
(define-fun
res32_864 () (_ BitVec 32)
(bvxor (bvand (bvneg (bvand res32_815 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_815 (_ bv1 32))))
(assert
(=
((_ zero_extend 24)
(ite
(=
(bvxor (bvand (bvneg (bvand res32_395 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_395 (_ bv1 32)))
(bvxor (bvand (bvneg (bvand res32_864 (_ bv1 32))) (_ bv3988292384 32))
(bvlshr res32_864 (_ bv1 32)))) (_ bv1 8) (_ bv0 8))) (_ bv1 32)))