import csv,sys
M={'t':'gpu__time_duration.sum','tc':'sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active','reg':'launch__registers_per_thread','smem':'launch__shared_mem_per_block','lreg':'launch__occupancy_limit_registers','lsm':'launch__occupancy_limit_shared_mem','warps':'sm__warps_active.avg.pct_of_peak_sustained_active','inst':'smsp__inst_executed.sum'}
R=['stall_barrier','stall_long_sb','stall_math','stall_wait','stall_mio','stall_short_sb']
d=sys.argv[1]; vs=sys.argv[2].split(','); pre=sys.argv[3] if len(sys.argv)>3 else ''
print("%-14s %5s %9s %6s %4s %7s %4s %6s | %s"%("variant","n","time","TC%","reg","smemKB","blk","warps%"," ".join("%7s"%r[6:] for r in R)))
for n in (1024,2048,4000):
  for v in vs:
    rows=list(csv.reader(open('%s/raw-%s%s-%d.csv'%(d,pre,v,n))))
    h=rows[0]; k=rows[-1]
    g=lambda m:k[h.index(m)]
    src=list(csv.reader(open('%s/src-%s%s-%d.csv'%(d,pre,v,n))))
    hh=src[1]; seen=set(); data=[]
    for r in src[2:]:
        if len(r)>3 and r[2].isdigit() and r[0] not in seen: seen.add(r[0]); data.append(r)
    tot=sum(int(r[2]) for r in data)
    st=[100*sum(int(r[hh.index(x)] or 0) for r in data)/tot for x in R]
    blk=min(float(g(M['lreg'])),float(g(M['lsm'])))
    print("%-14s %5d %9s %6.1f %4s %7s %4.0f %6.1f | %s"%(v,n,g(M['t']),float(g(M['tc'])),g(M['reg']),g(M['smem']),blk,float(g(M['warps'])), " ".join("%7.1f"%x for x in st)))
