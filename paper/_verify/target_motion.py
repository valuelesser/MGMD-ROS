# 检验论文 D>=1 的隐含前提：目标 3 帧内位移是否 >= 自身尺寸
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
tests="05 08 09 10 19 30 41 43 46 47 58 63 65 70 86".split()
ratios=[]
for t in tests:
    fr=load(f"phantom{t}"); idxs=sorted(fr)
    for k in range(len(idxs)-3):
        a,b=idxs[k],idxs[k+3]
        if len(fr[a])==1 and len(fr[b])==1:
            x1i,y1i,x2i,y2i=fr[a][0]; x1j,y1j,x2j,y2j=fr[b][0]
            disp=math.hypot((x1j+x2j)/2-(x1i+x2i)/2,(y1j+y2j)/2-(y1i+y2i)/2)
            size=math.hypot(x2j-x1j,y2j-y1j)
            if size>0: ratios.append(disp/size)
ratios.sort(); n=len(ratios)
print(f"N={n} GT pairs, 3-frame displacement / target diagonal")
for p in [10,25,50,75,90,95,99]:
    print(f"  P{p:<3} = {ratios[int(p/100*(n-1))]:.3f}")
print(f"  max = {ratios[-1]:.3f}")
print()
print(f"位移 >= 1.0x 自身尺寸的比例: {100*sum(1 for r in ratios if r>=1.0)/n:.2f}%")
print(f"位移 >= 2.0x 自身尺寸的比例: {100*sum(1 for r in ratios if r>=2.0)/n:.2f}%")
