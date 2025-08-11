# -*- coding: utf-8 -*-
def ctc_greedy_decode(logits, charset):
    blank_idx=0; T,N,C=logits.shape
    preds=logits.argmax(dim=2).detach().cpu().numpy(); res=[]
    for n in range(N):
        last=-1; s=[]
        for t in range(T):
            p=int(preds[t,n])
            if p!=blank_idx and p!=last: s.append(charset[p-1])
            last=p
        res.append("".join(s))
    return res
