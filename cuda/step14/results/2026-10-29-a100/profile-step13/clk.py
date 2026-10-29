import csv,datetime as D,statistics as st
f=lambda s:D.datetime.strptime(s.strip(),'%Y/%m/%d %H:%M:%S.%f')
smi=[]
for r in csv.reader(open('clocks.csv')):
    try: smi.append((f(r[0]),float(r[1].split()[0]),float(r[3].split()[0]),r[5].strip()))
    except: pass
for line in open('clockrun.log'):
    w=line.split(); a=f(w[6]+' '+w[7]); b=f(w[9]+' '+w[10])
    s=[x for x in smi if a+D.timedelta(seconds=0.5)<=x[0]<=b]
    print("%-7s n %5s %6s GFLOP/s  SM clock %5.0f MHz (%4.0f-%4.0f)  power %5.1f W  throttle %s"%(w[0],w[2],w[15],st.median(x[1] for x in s),min(x[1] for x in s),max(x[1] for x in s),st.median(x[2] for x in s),sorted(set(x[3] for x in s))))
