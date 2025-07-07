(set-info :controlled userpin_low_6)
(declare-fun __array_symbolic___memory_3_1 ()
(_ BitVec 8))
(assert
(distinct ((_ zero_extend 24) __array_symbolic___memory_3_1) (_ bv1 32)))
(declare-fun __array_symbolic___memory_3_2 ()
(_ BitVec 8))
(declare-fun __array_symbolic___memory_3_3 ()
(_ BitVec 8))
(declare-fun __array_symbolic___memory_3_4 ()
(_ BitVec 8))
(declare-fun __array_symbolic___memory_3_5 ()
(_ BitVec 8))
(declare-fun userpin_low_6 ()
(_ BitVec 16))
(declare-fun cardpin_low_7 ()
(_ BitVec 16))
(define-fun
cardpin_8 () (_ BitVec 32) ((_ zero_extend 16) cardpin_low_7))
(define-fun
userpin_9 () (_ BitVec 32) ((_ zero_extend 16) userpin_low_6))
(assert (distinct cardpin_8 userpin_9))
(declare-fun bs_unknown1_for___0x00002865_1__13 ()
(_ BitVec 8))
(assert; assume
(bvsle bs_unknown1_for___0x00002865_1__13 (_ bv3 8)))
(assert; assume
(bvsge bs_unknown1_for___0x00002865_1__13 (_ bv1 8)))
(define-fun
nxt_r2_35 () (_ BitVec 32)
((_ sign_extend 24) bs_unknown1_for___0x00002865_1__13))
(define-fun
tmp1_0_39 () (_ BitVec 1)
(ite (bvslt (bvsub nxt_r2_35 (_ bv1 32)) (_ bv0 32)) (_ bv1 1) (_ bv0 1)))
(assert
(distinct
(bvxor tmp1_0_39
(bvxor tmp1_0_39 (ite (bvslt nxt_r2_35 (_ bv1 32)) (_ bv1 1) (_ bv0 1))))
(_ bv1 1)))
(declare-fun r3_50 ()
(_ BitVec 32))
(define-fun
nxt_r3_51 () (_ BitVec 32) (bvand r3_50 (_ bv65535 32)))
(define-fun
assert_bv_unop_1 () (_ BitVec 8) ((_ extract 7 0) userpin_9))
(define-fun
assert_bv_unop_13 () (_ BitVec 8) ((_ extract 15 8) userpin_9))
(define-fun
assert_bv_unop_16 () (_ BitVec 8) ((_ extract 15 8) cardpin_8))
(define-fun
assert_bv_unop_17 () (_ BitVec 8) ((_ extract 7 0) cardpin_8))
(assert
(= ((_ zero_extend 24) assert_bv_unop_1)
((_ zero_extend 24)
(ite (= nxt_r3_51 (_ bv10342 32)) (_ bv85 8)
(ite (= nxt_r3_51 (_ bv10341 32)) bs_unknown1_for___0x00002865_1__13
(ite (= nxt_r3_51 (_ bv10346 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10345 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10344 32)) assert_bv_unop_13
(ite (= nxt_r3_51 (_ bv10343 32)) assert_bv_unop_1
(ite (= nxt_r3_51 (_ bv10350 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10349 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10348 32)) assert_bv_unop_16
(ite (= nxt_r3_51 (_ bv10347 32)) assert_bv_unop_17
(ite (= nxt_r3_51 (_ bv10342 32)) (_ bv85 8)
(ite (= nxt_r3_51 (_ bv10340 32)) __array_symbolic___memory_3_1
(ite (= nxt_r3_51 (bvadd nxt_r3_51 (_ bv2 32))) __array_symbolic___memory_3_2
__array_symbolic___memory_3_3))))))))))))))))
(assert
(=
((_ zero_extend 24)
(ite (= nxt_r3_51 (_ bv10339 32)) (_ bv85 8)
(ite (= nxt_r3_51 (_ bv10338 32)) bs_unknown1_for___0x00002865_1__13
(ite (= nxt_r3_51 (_ bv10343 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10342 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10341 32)) assert_bv_unop_13
(ite (= nxt_r3_51 (_ bv10340 32)) assert_bv_unop_1
(ite (= nxt_r3_51 (_ bv10347 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10346 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10345 32)) assert_bv_unop_16
(ite (= nxt_r3_51 (_ bv10344 32)) assert_bv_unop_17
(ite (= nxt_r3_51 (_ bv10339 32)) (_ bv85 8)
(ite (= nxt_r3_51 (_ bv10337 32)) __array_symbolic___memory_3_1
(ite (= (bvadd nxt_r3_51 (_ bv3 32)) (bvadd nxt_r3_51 (_ bv2 32)))
__array_symbolic___memory_3_2
(ite (= (bvadd nxt_r3_51 (_ bv3 32)) nxt_r3_51) __array_symbolic___memory_3_3
(ite (= (bvadd nxt_r3_51 (_ bv3 32)) (bvadd nxt_r3_51 (_ bv1 32)))
__array_symbolic___memory_3_4 __array_symbolic___memory_3_5))))))))))))))))
(_ bv0 32)))
(assert
(= ((_ zero_extend 24) assert_bv_unop_13)
((_ zero_extend 24)
(ite (= nxt_r3_51 (_ bv10341 32)) (_ bv85 8)
(ite (= nxt_r3_51 (_ bv10340 32)) bs_unknown1_for___0x00002865_1__13
(ite (= nxt_r3_51 (_ bv10345 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10344 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10343 32)) assert_bv_unop_13
(ite (= nxt_r3_51 (_ bv10342 32)) assert_bv_unop_1
(ite (= nxt_r3_51 (_ bv10349 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10348 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10347 32)) assert_bv_unop_16
(ite (= nxt_r3_51 (_ bv10346 32)) assert_bv_unop_17
(ite (= nxt_r3_51 (_ bv10341 32)) (_ bv85 8)
(ite (= nxt_r3_51 (_ bv10339 32)) __array_symbolic___memory_3_1
(ite (= (bvadd nxt_r3_51 (_ bv1 32)) (bvadd nxt_r3_51 (_ bv2 32)))
__array_symbolic___memory_3_2
(ite (= (bvadd nxt_r3_51 (_ bv1 32)) nxt_r3_51) __array_symbolic___memory_3_3
__array_symbolic___memory_3_4)))))))))))))))))
(assert
(=
((_ zero_extend 24)
(ite (= nxt_r3_51 (_ bv10340 32)) (_ bv85 8)
(ite (= nxt_r3_51 (_ bv10339 32)) bs_unknown1_for___0x00002865_1__13
(ite (= nxt_r3_51 (_ bv10344 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10343 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10342 32)) assert_bv_unop_13
(ite (= nxt_r3_51 (_ bv10341 32)) assert_bv_unop_1
(ite (= nxt_r3_51 (_ bv10348 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10347 32)) (_ bv0 8)
(ite (= nxt_r3_51 (_ bv10346 32)) assert_bv_unop_16
(ite (= nxt_r3_51 (_ bv10345 32)) assert_bv_unop_17
(ite (= nxt_r3_51 (_ bv10340 32)) (_ bv85 8)
(ite (= nxt_r3_51 (_ bv10338 32)) __array_symbolic___memory_3_1
__array_symbolic___memory_3_2))))))))))))) (_ bv0 32)))