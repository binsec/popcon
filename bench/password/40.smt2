(set-info :controlled input_7)
(declare-fun bs_unknown1_for_esp_32__4 ()
(_ BitVec 32))
(assert; assume
(bvuge bs_unknown1_for_esp_32__4 (_ bv4294901760 32)))
(assert; assume
(bvule bs_unknown1_for_esp_32__4 (_ bv4294967040 32)))
(declare-fun input_7 ()
(_ BitVec 40))
(declare-fun password_9 ()
(_ BitVec 40))
(assert
(=
(bvand
(bvand
(bvand
(bvand
(bvand
((_ zero_extend 24)
(ite (= ((_ extract 7 0) password_9) ((_ extract 7 0) input_7)) (_ bv1 8)
(_ bv0 8))) (_ bv1 32))
((_ zero_extend 24)
(ite (= ((_ extract 15 8) password_9) ((_ extract 15 8) input_7)) (_ bv1 8)
(_ bv0 8))))
((_ zero_extend 24)
(ite (= ((_ extract 23 16) password_9) ((_ extract 23 16) input_7)) (_ bv1 8)
(_ bv0 8))))
((_ zero_extend 24)
(ite (= ((_ extract 31 24) password_9) ((_ extract 31 24) input_7)) (_ bv1 8)
(_ bv0 8))))
((_ zero_extend 24)
(ite (= ((_ extract 39 32) password_9) ((_ extract 39 32) input_7)) (_ bv1 8)
(_ bv0 8)))) (_ bv0 32)))