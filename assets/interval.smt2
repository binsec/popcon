(set-logic QF_BV)
(declare-fun thevar () (_ BitVec 12))
(assert (bvult #x123 thevar))
(assert (bvult thevar #x234))
(check-sat)
(exit)
