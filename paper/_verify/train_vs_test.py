# 检验：在训练集标定的阈值，能否泛化到测试集？
import os, glob, re, math
import xml.etree.ElementTree as ET
BASE="/media/gaoduan/0C1CFCAF1CFC94C4/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV/Annotations"
def load(v):
    fr={}
    for f in glob.glob(os.path.join(BASE,v,"*.xml")):
        m=re.search(r"_(\d+)\.xml$",f)
        if not m: continue
        try: r=ET.parse(f).getroot()
        except: continue
        o=[]
        for x in r.findall("object"):
            bb=x.find("bndbox")
            if bb is None: continue
            o.append(tuple(float(bb.find(k).text) for k in ("xmin","ymin","xmax","ymax")))
        fr[int(m.group(1))]=o
    return fr
def D(b1,b2):
    x1i,y1i,x2i,y2i=b1; x1j,y1j,x2j,y2j=b2
    W=max(x2i,x2j)-min(x1i,x1j); H=max(y2i,y2j)-min(y1i,y1j)
    d=math.hypot(W,H)
    r=math.hypot((x1j+x2j)/2-(x1i+x2i)/2,(y1j+y2j)/2-(y1i+y2i)/2)
    return r/d if d>0 else 0.0
test={"phantom"+t for t in "05 08 09 10 19 30 41 43 46 47 58 63 65 70 86".split()}
allv=[d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE,d))]
train=[v for v in allv if v not in test]
def collect(vids):
    out=[]
    for v in vids:
        fr=load(v); ix=sorted(fr)
        for k in range(len(ix)-3):
            a,b=ix[k],ix[k+3]
            if len(fr[a])==1 and len(fr[b])==1: out.append(D(fr[a][0],fr[b][0]))
    return out
tr=collect(train); te=collect(sorted(test))
tr.sort(); te.sort()
def pct(a,p): return a[int(p/100*(len(a)-1))]
print(f"TRAIN (45 videos): N={len(tr)}  P50={pct(tr,50):.4f} P90={pct(tr,90):.4f} P95={pct(tr,95):.4f} max={tr[-1]:.4f}")
print(f"TEST  (15 videos): N={len(te)}  P50={pct(te,50):.4f} P90={pct(te,90):.4f} P95={pct(te,95):.4f} max={te[-1]:.4f}")
print()
print("若在 TRAIN 上用 P90 标定阈值，在 TEST 上的通过率:")
for p in [50,75,90,95]:
    thr=pct(tr,p)
    rate=100*sum(1 for x in te if x>=thr)/len(te)
    print(f"  train P{p}={thr:.4f} -> test 通过 {rate:.2f}%")
print()
print("D>=1.0 (论文原文):")
print(f"  train 通过 {100*sum(1 for x in tr if x>=1)/len(tr):.3f}%")
print(f"  test  通过 {100*sum(1 for x in te if x>=1)/len(te):.3f}%")
