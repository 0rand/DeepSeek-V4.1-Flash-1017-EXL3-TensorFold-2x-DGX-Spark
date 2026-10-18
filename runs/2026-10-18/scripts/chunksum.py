import re,sys
runs=[];cur=[]
for l in sys.stdin:
    m=re.search(r"T(\S{8}).*chunk at (\d+): ([\d.]+)s",l)
    if not m: continue
    t,pos,s=m.group(1),int(m.group(2)),float(m.group(3))
    if pos==0 and cur: runs.append(cur); cur=[]
    cur.append((t,pos,s))
runs.append(cur)
for r in runs:
    if len(r)<4: continue
    ss=[s for _,_,s in r]
    big=[(p//1024,s) for _,p,s in r if s>1.3]
    print(r[0][0],"chunks",len(r),"sum %.1f"%sum(ss),"median %.2f"%sorted(ss)[len(ss)//2],"min %.2f"%min(ss),"slow>1.3s:",len(big),"=%.1fs"%sum(s for _,s in big), big[:8])
