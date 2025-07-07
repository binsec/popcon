(set-info :controlled packet_8)
(declare-fun __array_symbolic___memory_3_1 ()
(_ BitVec 8))
(declare-fun __array_symbolic___memory_3_2 ()
(_ BitVec 8))
(declare-fun __array_symbolic___memory_3_3 ()
(_ BitVec 8))
(declare-fun __array_symbolic___memory_3_4 ()
(_ BitVec 8))
(declare-fun bs_unknown1_for_sp_32__4 ()
(_ BitVec 32))
(assert; assume
(bvuge bs_unknown1_for_sp_32__4 (_ bv4278190080 32)))
(assert; assume
(bvule bs_unknown1_for_sp_32__4 (_ bv4294901760 32)))
(declare-fun packet_8 ()
(_ BitVec 64))
(declare-fun bs_unknown2_for_packet_p_32__9 ()
(_ BitVec 32))
(assert; assume
(bvuge bs_unknown2_for_packet_p_32__9 (_ bv2863267840 32)))
(assert; assume
(bvule bs_unknown2_for_packet_p_32__9 (_ bv2863315899 32)))
(define-fun
nxt_r0_88 () (_ BitVec 32)
(concat __array_symbolic___memory_3_1
(concat __array_symbolic___memory_3_2
(concat __array_symbolic___memory_3_3 __array_symbolic___memory_3_4))))
(define-fun
nxt_r3_90_bv_bnop_0_bv_unop_1 () (_ BitVec 32)
((_ zero_extend 16) ((_ extract 15 0) packet_8)))
(define-fun
nxt_r3_90_bv_bnop_0_bv_unop_3 () (_ BitVec 32)
((_ zero_extend 16) ((_ extract 31 16) packet_8)))
(define-fun
nxt_r3_90 () (_ BitVec 32)
(bvand (bvadd nxt_r3_90_bv_bnop_0_bv_unop_1 nxt_r3_90_bv_bnop_0_bv_unop_3)
nxt_r0_88))
(declare-fun target_checksum_100 ()
(_ BitVec 32))
(define-fun
assert_bv_bnop_0 () (_ BitVec 32)
(bvashr (bvadd nxt_r3_90_bv_bnop_0_bv_unop_1 nxt_r3_90_bv_bnop_0_bv_unop_3)
(_ bv16 32)))
(assert
(=
(bvand nxt_r0_88
(bvnot
(bvadd
(bvadd (bvashr (bvadd assert_bv_bnop_0 nxt_r3_90) (_ bv16 32))
assert_bv_bnop_0) nxt_r3_90))) target_checksum_100))