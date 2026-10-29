import csv
R=['stall_barrier','stall_long_sb','stall_math','stall_wait','stall_mio','stall_short_sb','stall_selected','stall_lg']
for n in (1024,2048,4000):
    rows=list(csv.reader(open('src13-%d.csv'%n)))
    h=rows[1]
    seen=set(); data=[]
    for r in rows[2:]:
        if len(r)>3 and r[2].isdigit() and r[0] not in seen: seen.add(r[0]); data.append(r)
    ix={k:h.index(k) for k in R}
    tot=sum(int(r[2]) for r in data)
    print("== n",n,"samples",tot)
    for k in R:
        s=sorted(data,key=lambda r:-int(r[ix[k]] or 0))
        kt=sum(int(r[ix[k]] or 0) for r in data)
        print("  %-15s %5.1f%%  top:"%(k[6:],100*kt/tot)," | ".join("%s %.1f%%"%(r[1].strip()[:38],100*int(r[ix[k]])/tot) for r in s[:3]))
