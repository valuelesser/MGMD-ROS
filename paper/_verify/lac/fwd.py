import zipfile, struct, math, os
CKPT="/home/gaoduan/tracking/data_raw/mgmd_weights/Net_best.pth"
z=zipfile.ZipFile(CKPT)
meta={"conv1.weight":(94149863180096,450),"conv1.bias":(94149863045648,6),
      "conv2.weight":(94149860432144,2400),"conv2.bias":(94149862622768,16),
      "fc1.weight":(94149862071984,48000),"fc1.bias":(94149864752288,120),
      "fc2.weight":(94149861902160,10080),"fc2.bias":(94149863057968,84),
      "fc3.weight":(94149860277680,168),"fc3.bias":(94149861903056,2)}
W={k:list(struct.unpack(f"<{n}f", z.read(f"archive/data/{s}")[:n*4])) for k,(s,n) in meta.items()}

def conv(inp,C,H,Wd,wt,bias,OC):
    oh,ow=H-4,Wd-4; out=[]
    for oc in range(OC):
        pl=[0.0]*(oh*ow); base_o=oc*25*C
        for y in range(oh):
            for x in range(ow):
                s=bias[oc]
                for c in range(C):
                    wb=base_o+c*25; ip=inp[c]
                    r=[y*Wd+x, (y+1)*Wd+x, (y+2)*Wd+x, (y+3)*Wd+x, (y+4)*Wd+x]
                    for ky in range(5):
                        b0=r[ky]
                        for kx in range(5):
                            s+=wt[wb+ky*5+kx]*ip[b0+kx]
                pl[y*ow+x]=s if s>0 else 0.0
        out.append(pl)
    return out,oh,ow

def pool(pl,H,Wd):
    oh,ow=H//2,Wd//2; out=[]
    for p in pl:
        q=[0.0]*(oh*ow)
        for y in range(oh):
            for x in range(ow):
                m=p[(2*y)*Wd+2*x]
                for dy in (0,1):
                    for dx in (0,1):
                        v=p[(2*y+dy)*Wd+2*x+dx]
                        if v>m: m=v
                q[y*ow+x]=m
        out.append(q)
    return out,oh,ow

def forward(planes):
    x,H,Wd=conv(planes,3,32,32,W["conv1.weight"],W["conv1.bias"],6)
    x,H,Wd=pool(x,H,Wd)
    x,H,Wd=conv(x,6,H,Wd,W["conv2.weight"],W["conv2.bias"],16)
    x,H,Wd=pool(x,H,Wd)
    flat=[]
    for p in x: flat+=p
    def fc(inp,nin,w,b,nout):
        o=[0.0]*nout
        for i in range(nout):
            s=b[i]; base=i*nin
            for j in range(nin): s+=w[base+j]*inp[j]
            o[i]=s
        return o
    f1=[max(0.0,v) for v in fc(flat,400,W["fc1.weight"],W["fc1.bias"],120)]
    f2=[max(0.0,v) for v in fc(f1,120,W["fc2.weight"],W["fc2.bias"],84)]
    lg=fc(f2,84,W["fc3.weight"],W["fc3.bias"],2)
    e=[math.exp(v) for v in lg]; s=sum(e)
    return lg, e[1]/s

def loadrgb(path):
    b=open(path,'rb').read()
    planes=[[0.0]*1024 for _ in range(3)]
    for i in range(1024):
        for c in range(3):
            planes[c][i]=b[i*3+c]/255.0
    return planes

for name,f in [("真实目标裁剪 32x32","crop.rgb")]:
    if os.path.exists(f):
        lg,p=forward(loadrgb(f))
        print(f"{name}: logits=({lg[0]:.4f}, {lg[1]:.4f})  P(MAV)={p:.6f}")

def loadrgb_bgr(path):
    b=open(path,'rb').read()
    planes=[[0.0]*1024 for _ in range(3)]
    for i in range(1024):
        r,g,bl=b[i*3],b[i*3+1],b[i*3+2]
        planes[0][i]=bl/255.0; planes[1][i]=g/255.0; planes[2][i]=r/255.0
    return planes

print()
print("=== 5 帧真实数据：目标裁剪 vs 背景裁剪 ===")
print(f"{'frame':>6} {'tgt_RGB':>9} {'tgt_BGR':>9} {'bg_RGB':>9} {'bg_BGR':>9}")
for idx in [100,300,600,900,1200]:
    tp=f"crops/tgt{idx}.rgb"; bp=f"crops/bg{idx}.rgb"
    if not (os.path.exists(tp) and os.path.exists(bp)): continue
    _,pt_rgb=forward(loadrgb(tp)); _,pt_bgr=forward(loadrgb_bgr(tp))
    _,pb_rgb=forward(loadrgb(bp)); _,pb_bgr=forward(loadrgb_bgr(bp))
    print(f"{idx:>6} {pt_rgb:>9.4f} {pt_bgr:>9.4f} {pb_rgb:>9.4f} {pb_bgr:>9.4f}")
