(set-info :controlled cardpin_6)
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
(declare-fun cardpin_6 ()
(_ BitVec 32))
(declare-fun userpin_9 ()
(_ BitVec 32))
(assert (distinct cardpin_6 userpin_9))
(declare-fun bs_unknown1_for___0x00002865_1__11 ()
(_ BitVec 8))
(assert; assume
(bvsle bs_unknown1_for___0x00002865_1__11 (_ bv3 8)))
(assert; assume
(bvsge bs_unknown1_for___0x00002865_1__11 (_ bv1 8)))
(define-fun
nxt_r2_33 () (_ BitVec 32)
((_ sign_extend 24) bs_unknown1_for___0x00002865_1__11))
(define-fun
tmp1_0_37 () (_ BitVec 1)
(ite (bvslt (bvsub nxt_r2_33 (_ bv1 32)) (_ bv0 32)) (_ bv1 1) (_ bv0 1)))
(assert
(distinct
(bvxor tmp1_0_37
(bvxor tmp1_0_37 (ite (bvslt nxt_r2_33 (_ bv1 32)) (_ bv1 1) (_ bv0 1))))
(_ bv1 1)))
(declare-fun r3_48 ()
(_ BitVec 32))
(define-fun
nxt_r3_49 () (_ BitVec 32) (bvand r3_48 (_ bv65535 32)))
(define-fun
assert_bv_unop_1 () (_ BitVec 8) ((_ extract 23 16) userpin_9))
(define-fun
assert_bv_unop_11 () (_ BitVec 8) ((_ extract 31 24) userpin_9))
(define-fun
assert_bv_unop_12 () (_ BitVec 8) ((_ extract 15 8) userpin_9))
(define-fun
assert_bv_unop_13 () (_ BitVec 8) ((_ extract 7 0) userpin_9))
(define-fun
assert_bv_unop_14 () (_ BitVec 8) ((_ extract 31 24) cardpin_6))
(define-fun
assert_bv_unop_15 () (_ BitVec 8) ((_ extract 23 16) cardpin_6))
(define-fun
assert_bv_unop_16 () (_ BitVec 8) ((_ extract 15 8) cardpin_6))
(define-fun
assert_bv_unop_17 () (_ BitVec 8) ((_ extract 7 0) cardpin_6))
(assert
(= ((_ zero_extend 24) assert_bv_unop_1)
((_ zero_extend 24)
(ite (= nxt_r3_49 (_ bv10340 32)) (_ bv85 8)
(ite (= nxt_r3_49 (_ bv10339 32)) bs_unknown1_for___0x00002865_1__11
(ite (= nxt_r3_49 (_ bv10344 32)) assert_bv_unop_11
(ite (= nxt_r3_49 (_ bv10343 32)) assert_bv_unop_1
(ite (= nxt_r3_49 (_ bv10342 32)) assert_bv_unop_12
(ite (= nxt_r3_49 (_ bv10341 32)) assert_bv_unop_13
(ite (= nxt_r3_49 (_ bv10348 32)) assert_bv_unop_14
(ite (= nxt_r3_49 (_ bv10347 32)) assert_bv_unop_15
(ite (= nxt_r3_49 (_ bv10346 32)) assert_bv_unop_16
(ite (= nxt_r3_49 (_ bv10345 32)) assert_bv_unop_17
(ite (= nxt_r3_49 (_ bv10340 32)) (_ bv85 8)
(ite (= nxt_r3_49 (_ bv10338 32)) __array_symbolic___memory_3_1
__array_symbolic___memory_3_2)))))))))))))))
(assert
(= ((_ zero_extend 24) assert_bv_unop_12)
((_ zero_extend 24)
(ite (= nxt_r3_49 (_ bv10341 32)) (_ bv85 8)
(ite (= nxt_r3_49 (_ bv10340 32)) bs_unknown1_for___0x00002865_1__11
(ite (= nxt_r3_49 (_ bv10345 32)) assert_bv_unop_11
(ite (= nxt_r3_49 (_ bv10344 32)) assert_bv_unop_1
(ite (= nxt_r3_49 (_ bv10343 32)) assert_bv_unop_12
(ite (= nxt_r3_49 (_ bv10342 32)) assert_bv_unop_13
(ite (= nxt_r3_49 (_ bv10349 32)) assert_bv_unop_14
(ite (= nxt_r3_49 (_ bv10348 32)) assert_bv_unop_15
(ite (= nxt_r3_49 (_ bv10347 32)) assert_bv_unop_16
(ite (= nxt_r3_49 (_ bv10346 32)) assert_bv_unop_17
(ite (= nxt_r3_49 (_ bv10341 32)) (_ bv85 8)
(ite (= nxt_r3_49 (_ bv10339 32)) __array_symbolic___memory_3_1
(ite (= (bvadd nxt_r3_49 (_ bv1 32)) (bvadd nxt_r3_49 (_ bv2 32)))
__array_symbolic___memory_3_2
(ite (= (bvadd nxt_r3_49 (_ bv1 32)) nxt_r3_49) __array_symbolic___memory_3_3
__array_symbolic___memory_3_4)))))))))))))))))
(assert
(= ((_ zero_extend 24) assert_bv_unop_11)
((_ zero_extend 24)
(ite (= nxt_r3_49 (_ bv10339 32)) (_ bv85 8)
(ite (= nxt_r3_49 (_ bv10338 32)) bs_unknown1_for___0x00002865_1__11
(ite (= nxt_r3_49 (_ bv10343 32)) assert_bv_unop_11
(ite (= nxt_r3_49 (_ bv10342 32)) assert_bv_unop_1
(ite (= nxt_r3_49 (_ bv10341 32)) assert_bv_unop_12
(ite (= nxt_r3_49 (_ bv10340 32)) assert_bv_unop_13
(ite (= nxt_r3_49 (_ bv10347 32)) assert_bv_unop_14
(ite (= nxt_r3_49 (_ bv10346 32)) assert_bv_unop_15
(ite (= nxt_r3_49 (_ bv10345 32)) assert_bv_unop_16
(ite (= nxt_r3_49 (_ bv10344 32)) assert_bv_unop_17
(ite (= nxt_r3_49 (_ bv10339 32)) (_ bv85 8)
(ite (= nxt_r3_49 (_ bv10337 32)) __array_symbolic___memory_3_1
(ite (= (bvadd nxt_r3_49 (_ bv3 32)) (bvadd nxt_r3_49 (_ bv2 32)))
__array_symbolic___memory_3_2
(ite (= (bvadd nxt_r3_49 (_ bv3 32)) nxt_r3_49) __array_symbolic___memory_3_3
(ite (= (bvadd nxt_r3_49 (_ bv3 32)) (bvadd nxt_r3_49 (_ bv1 32)))
__array_symbolic___memory_3_4 __array_symbolic___memory_3_5))))))))))))))))))
(assert
(= ((_ zero_extend 24) assert_bv_unop_13)
((_ zero_extend 24)
(ite (= nxt_r3_49 (_ bv10342 32)) (_ bv85 8)
(ite (= nxt_r3_49 (_ bv10341 32)) bs_unknown1_for___0x00002865_1__11
(ite (= nxt_r3_49 (_ bv10346 32)) assert_bv_unop_11
(ite (= nxt_r3_49 (_ bv10345 32)) assert_bv_unop_1
(ite (= nxt_r3_49 (_ bv10344 32)) assert_bv_unop_12
(ite (= nxt_r3_49 (_ bv10343 32)) assert_bv_unop_13
(ite (= nxt_r3_49 (_ bv10350 32)) assert_bv_unop_14
(ite (= nxt_r3_49 (_ bv10349 32)) assert_bv_unop_15
(ite (= nxt_r3_49 (_ bv10348 32)) assert_bv_unop_16
(ite (= nxt_r3_49 (_ bv10347 32)) assert_bv_unop_17
(ite (= nxt_r3_49 (_ bv10342 32)) (_ bv85 8)
(ite (= nxt_r3_49 (_ bv10340 32)) __array_symbolic___memory_3_1
(ite (= nxt_r3_49 (bvadd nxt_r3_49 (_ bv2 32))) __array_symbolic___memory_3_2
__array_symbolic___memory_3_3))))))))))))))))