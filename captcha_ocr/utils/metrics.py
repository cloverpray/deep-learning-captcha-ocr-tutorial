# -*- coding: utf-8 -*-
def exact_match(preds, gts):
    if not gts: return 0.0
    return sum(1 for p,g in zip(preds,gts) if p==g)/len(gts)
