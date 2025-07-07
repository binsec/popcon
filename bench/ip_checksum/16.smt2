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
nxt_r0_71 () (_ BitVec 32)
(concat __array_symbolic___memory_3_1
(concat __array_symbolic___memory_3_2
(concat __array_symbolic___memory_3_3 __array_symbolic___memory_3_4))))
(declare-fun target_checksum_83 ()
(_ BitVec 32))
(assert
(=
(bvand nxt_r0_71
(bvnot (bvand ((_ zero_extend 16) ((_ extract 15 0) packet_8)) nxt_r0_71)))
target_checksum_83))