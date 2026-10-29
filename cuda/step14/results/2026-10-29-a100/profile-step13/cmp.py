import csv,sys,re
want=sys.argv[1:]
for n in (512,1024,2048,4000):
    rows=list(csv.reader(open('raw-%d.csv'%n)))
    h=rows[0]; u=rows[1]
    ks=[rows[4],rows[7]]  # 3rd launch of each
    print("== n",n)
    for w in want:
        for i,name in enumerate(h):
            if re.search(w,name):
                print("  %-75s %8s | %14s %14s"%(name[:75],u[i][:8],ks[0][i],ks[1][i]))
