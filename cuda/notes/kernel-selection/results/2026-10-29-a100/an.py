import glob,statistics as st,math
# x[(kernel,(m,n,k))] = list of GFLOP/s over the 3 jobs
x={}
for f in glob.glob('d[123]/*.bench.log'):
    ker=f.split('/')[-1].replace('.bench.log','')
    for line in open(f):
        w=line.split()
        if len(w)>=9 and w[0]=='step14': x.setdefault((ker,(int(w[1]),int(w[2]),int(w[3]))),[]).append(float(w[7]))
        if len(w)>=9 and w[0]=='cuBLAS': x.setdefault(('cuBLAS',(int(w[1]),int(w[2]),int(w[3]))),[]).append(float(w[7]))
shapes=sorted({s for (_,s) in x},key=lambda s:(s[0]*s[1],s[0],s[2]))
BR={'small':(32,16),'medium':(64,32),'large':(64,64)}
slots={'small':7*108,'medium':4*108,'large':4*108}
print("%-17s %8s %8s %8s %8s %8s | best  rule  med/best  waves s/m/l"%("m x n x k","small","medium","large","dispatch","cuBLAS"))
for s in shapes:
    m,n,k=s
    v={K:st.median(x[(K,s)]) for K in ('small','medium','large','dispatch','cuBLAS')}
    best=max(('small','medium','large'),key=lambda K:v[K])
    rule='small' if m*n<1000*1000 else 'medium' if m*n<3000*3000 else 'large'
    waves=["%.1f"%(math.ceil(m/BR[K][0])*math.ceil(n/BR[K][1])/slots[K]) for K in ('small','medium','large')]
    spread=max(x[('medium',s)])/min(x[('medium',s)])
    print("%-17s %8.0f %8.0f %8.0f %8.0f %8.0f | %-6s %-6s %.3f  %s  (spread %.2f)"%("%dx%dx%d"%s,v['small'],v['medium'],v['large'],v['dispatch'],v['cuBLAS'],best,rule,v['dispatch']/v[best],"/".join(waves),spread))
