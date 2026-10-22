import os, glob, re, math, statistics
import xml.etree.ElementTree as ET
BASE="/media/gaoduan/0C1CFCAF1CFC94C4/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV/Annotations"
def load(v):
    frames={}
    for f in glob.glob(os.path.join(BASE,v,"*.xml")):
        m=re.search(r"_(\d+)\.xml$",f)
        if not m: continue
        try: r=ET.parse(f).getroot()
        except: continue
        objs=[]
        for o in r.findall("object"):
            bb=o.find("bndbox")
            if bb is None: continue
            objs.append(tuple(float(bb.find(k).text) for k in ("xmin","ymin","xmax","ymax")))
        frames[int(m.group(1))]=objs
    return frames
def D(b1,b2):
    x1i,y1i,x2i,y2i=b1; x1j,y1j,x2j,y2j=b2
    W=max(x2i,x2j)-min(x1i,x1j); H=max(y2i,y2j)-min(y1i,y1j)
    d=math.hypot(W,H)
    rho=math.hypot((x1j+x2j)/2-(x1i+x2i)/2,(y1j+y2j)/2-(y1i+y2i)/2)
    return rho/d if d>0 else 0.0
tests="05 08 09 10 19 30 41 43 46 47 58 63 65 70 86".split()
vals=[]
for t in tests:
    fr=load(f"phantom{t}"); idxs=sorted(fr)
    for k in range(len(idxs)-3):
        a,b=idxs[k],idxs[k+3]
        if len(fr[a])==1 and len(fr[b])==1: vals.append(D(fr[a][0],fr[b][0]))
vals.sort(); n=len(vals)
def pct(p): return vals[int(p/100*(n-1))]
print(f"N={n}")
for p in [1,5,10,25,50,75,90,95,99,99.9]:
    print(f"  P{p:<5} = {pct(p):.4f}")
print(f"  max    = {vals[-1]:.4f}")
print()
# 若把阈值定在 P90 或 P95，多少 GT 通过
for thr in [0.5,0.6,0.7,0.8,0.9]:
    c=sum(1 for v in vals if v>=thr)
    print(f"  thr={thr}: 通过 {c}/{n} = {100*c/n:.2f}%")
