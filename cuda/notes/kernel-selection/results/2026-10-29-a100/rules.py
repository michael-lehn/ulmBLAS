import glob,statistics as st,math
x={}
for f in glob.glob('d[123]/*.bench.log'):
    ker=f.split('/')[-1].replace('.bench.log','')
    for line in open(f):
        w=line.split()
        if len(w)>=9 and w[0]=='step14': x.setdefault((ker,(int(w[1]),int(w[2]),int(w[3]))),[]).append(float(w[7]))
shapes=sorted({s for (_,s) in x})
v=lambda K,s:st.median(x[(K,s)])
def cur(m,n,k): return 'small' if m*n<1000*1000 else 'medium' if m*n<3000*3000 else 'large'
rules={
 'current (m*n)':cur,
 'large from m*n >= 2048^2':lambda m,n,k:'small' if m*n<1e6 else 'medium' if m*n<2048*2048 else 'large',
 'large from 2048^2 if k <= 1024':lambda m,n,k:'small' if m*n<1e6 else 'large' if (m*n>=3000*3000 or (m*n>=2048*2048 and k<=1024)) else 'medium',
 'oracle (best of three)':None,
}
for name,r in rules.items():
    rat=[]
    for s in shapes:
        best=max(v(K,s) for K in ('small','medium','large'))
        ch=best if r is None else v(r(*s),s)
        rat.append(ch/v(cur(*s),s))
    g=math.exp(sum(map(math.log,rat))/len(rat))
    print("%-34s geo-mean vs current %.4f  min %.3f  max %.3f  shapes changed %d"%(name,g,min(rat),max(rat),sum(1 for q in rat if abs(q-1)>1e-9)))
