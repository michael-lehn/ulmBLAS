import csv,re
for n in (1024,2048,4000):
    rows=list(csv.reader(open('raw-%d.csv'%n)))
    h=rows[0]; ks=[rows[4],rows[7]]
    print("== n",n,"  (pc sampling: share of samples)")
    idx=[i for i,x in enumerate(h) if re.match(r'smsp__pcsamp_warps_issue_stalled_[a-z_]+$',x) and 'not_issued' not in x]
    tot=[sum(float(k[i].replace(',','') or 0) for i in idx) for k in ks]
    out=[]
    for i in idx:
        v=[float(k[i].replace(',','') or 0)/t*100 for k,t in zip(ks,tot)]
        if max(v)>1: out.append((v[1],h[i].replace('smsp__pcsamp_warps_issue_stalled_',''),v))
    for _,name,v in sorted(out,reverse=True): print("   %-24s cuBLAS %5.1f  step13 %5.1f"%(name,v[0],v[1]))
