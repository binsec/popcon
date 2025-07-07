; this has a 7GB output with d4 in about 20min
(set-info :controlled userpin_low_6)
(declare-fun __array_symbolic___memory_3_1 ()
(_ BitVec 8))
(declare-fun __array_symbolic___memory_3_2 ()
(_ BitVec 8))
(declare-fun userpin_low_6 ()
(_ BitVec 24))
(declare-fun cardpin_low_7 ()
(_ BitVec 24))
(define-fun
cardpin_8 () (_ BitVec 32) ((_ zero_extend 8) cardpin_low_7))
(define-fun
userpin_9 () (_ BitVec 32) ((_ zero_extend 8) userpin_low_6))
(assert
(distinct ((_ zero_extend 24) ((_ extract 7 0) userpin_9))
((_ zero_extend 24) ((_ extract 7 0) cardpin_8))))
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
(assert
(distinct ((_ zero_extend 24) __array_symbolic___memory_3_1) (_ bv1 32)))
(assert (= ((_ zero_extend 24) __array_symbolic___memory_3_2) (_ bv170 32)))
