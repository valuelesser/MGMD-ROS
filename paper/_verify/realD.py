import os, glob, re, math
import xml.etree.ElementTree as ET

BASE="/media/gaoduan/0C1CFCAF1CFC94C4/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV/Annotations"

def load_video(v):
    frames={}
    for f in glob.glob(os.path.join(BASE,v,"*.xml")):
        m=re.search(r"_(\d+)\.xml$",f)
        if not m: continue
        idx=int(m.group(1))
        try: r=ET.parse(f).getroot()
        except: continue
        objs=[]
        for o in r.findall("object"):
            bb=o.find("bndbox")
            if bb is None: continue
            objs.append((float(bb.find("xmin").text), float(bb.find("ymin").text),
                         float(bb.find("xmax").text), float(bb.find("ymax").text)))
        frames[idx]=objs
    return frames

def D_enclosing(b1,b2):
    x1i,y1i,x2i,y2i=b1; x1j,y1j,x2j,y2j=b2
    W=max(x2i,x2j)-min(x1i,x1j); H=max(y2i,y2j)-min(y1i,y1j)
    d=math.hypot(W,H)
    rho=math.hypot((x1j+x2j)/2-(x1i+x2i)/2, (y1j+y2j)/2-(y1i+y2i)/2)
    return rho/d if d>0 else 0.0

def D_self(b1,b2):
    x1i,y1i,x2i,y2i=b1; x1j,y1j,x2j,y2j=b2
    d=math.hypot(x2j-x1j, y2j-y1j)
    rho=math.hypot((x1j+x2j)/2-(x1i+x2i)/2, (y1j+y2j)/2-(y1i+y2i)/2)
    return rho/d if d>0 else 0.0

tests="05 08 09 10 19 30 41 43 46 47 58 63 65 70 86".split()
allD_enc=[]; allD_self=[]; per=[]
for t in tests:
    v=f"phantom{t}"
    fr=load_video(v)
    if not fr: continue
    idxs=sorted(fr)
    ds_enc=[]; ds_self=[]
    for k in range(len(idxs)-3):
        a,b=idxs[k],idxs[k+3]
        if len(fr[a])==1 and len(fr[b])==1:
            ds_enc.append(D_enclosing(fr[a][0],fr[b][0]))
            ds_self.append(D_self(fr[a][0],fr[b][0]))
    if ds_enc:
        allD_enc+=ds_enc; allD_self+=ds_self
        per.append((v,len(ds_enc),max(ds_enc),sum(1 for x in ds_enc if x>=1),
                    max(ds_self),sum(1 for x in ds_self if x>=1)))

print(f"{'video':<11}{'pairs':>7}{'maxD_enc':>10}{'enc>=1':>8}{'maxD_self':>11}{'self>=1':>9}")
for r in per:
    print(f"{r[0]:<11}{r[1]:>7}{r[2]:>10.4f}{r[3]:>8}{r[4]:>11.4f}{r[5]:>9}")
n=len(allD_enc)
e1=sum(1 for x in allD_enc if x>=1); s1=sum(1 for x in allD_self if x>=1)
print()
print(f"TOTAL {n} GT pairs (gap=3 frames, single-object frames only)")
print(f"enclosing (paper literal): max={max(allD_enc):.4f}  D>=1: {e1} ({100*e1/n:.3f}%)")
print(f"self (alternative)       : max={max(allD_self):.4f}  D>=1: {s1} ({100*s1/n:.3f}%)")
