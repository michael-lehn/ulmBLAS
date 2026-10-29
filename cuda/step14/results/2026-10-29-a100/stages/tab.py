import glob,statistics as st,sys
K,R=sys.argv[1],sys.argv[2]
def load(f):
    x={}
    for line in open(f):
        w=line.split()
        if len(w)>=9 and w[1].isdigit(): x[(w[0],int(w[1]))]=float(w[7])
    return x
rng=[("<=500",lambda n:n<=500),("550-950",lambda n:550<=n<=950),("1000-1950",lambda n:1000<=n<1999),("2000-2950",lambda n:2000<=n<3000),(">=3000",lambda n:n>=3000)]
print("%-16s"%"spec"+"".join("%18s"%r for r,_ in rng))
for f in sorted(glob.glob(sys.argv[3])):
    x=load(f); ns=sorted(n for (v,n) in x if v==K)
    out=[]
    for r,c in rng:
        s=[n for n in ns if c(n)]
        if not s: out.append("%18s"%"-"); continue
        a=st.median(x[(K,n)]/x[(R,n)] for n in s); b=st.median(x[(K,n)]/x[('cuBLAS',n)] for n in s)
        out.append("%9.3f /%6.3f "%(a,b))
    print("%-16s"%f.split('a100-s14-')[1].replace('.bench.log','')+"".join(out))
