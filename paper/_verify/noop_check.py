# tracking/ 的 d_min 下调到 0.003 后，Algorithm 1 是否还有筛选作用？
import os,glob,re,math
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
vals=[]
for v in sorted(os.listdir(BASE)):
    p=os.path.join(BASE,v)
    if not os.path.isdir(p): continue
    fr=load(v); ix=sorted(fr)
    for k in range(len(ix)-3):
        a,b=ix[k],ix[k+3]
        if len(fr[a])==1 and len(fr[b])==1: vals.append(D(fr[a][0],fr[b][0]))
n=len(vals)
print(f"N={n}")
for thr,label in [(1.0,"论文原文"),(0.1,"D3"),(0.01,"D2"),(0.003,"D1"),(0.002,"D4")]:
    keep=sum(1 for x in vals if x>=thr)
    print(f"  d_min={thr:<6} ({label:<5}) -> 保留 {keep}/{n} = {100*keep/n:.2f}%  (即剔除 {100-100*keep/n:.2f}%)")
