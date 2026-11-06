import numpy as np, cv2, glob
from PIL import Image
fs=sorted(glob.glob('logo/f*.png'))[-60:]
a=np.mean([np.asarray(Image.open(f).convert('RGB'),dtype=np.float32) for f in fs],axis=0)
S=8
blue=a[...,2]-0.5*(a[...,0]+a[...,1])
big=cv2.resize(blue,None,fx=S,fy=S,interpolation=cv2.INTER_CUBIC)
m=(cv2.GaussianBlur(big,(0,0),S*1.1)>40).astype(np.uint8)*255
cs,hier=cv2.findContours(m,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
R=4
out=np.zeros((m.shape[0]*R,m.shape[1]*R),np.uint8)
simp=[cv2.approxPolyDP(c,S*0.55,True) for c in cs]
# draw outer then holes
for i,c in enumerate(simp):
    col=255 if hier[0][i][3]<0 else 0
    if cv2.contourArea(cs[i])<300: continue
    if col==255: cv2.fillPoly(out,[c*R],255,lineType=cv2.LINE_AA)
for i,c in enumerate(simp):
    if hier[0][i][3]>=0 and cv2.contourArea(cs[i])>=300: cv2.fillPoly(out,[c*R],0,lineType=cv2.LINE_AA)
out=cv2.resize(out,None,fx=1/R*2,fy=1/R*2,interpolation=cv2.INTER_AREA)  # 2x of S-scale
ys,xs=np.where(out>10); out=np.pad(out[ys.min():ys.max()+1,xs.min():xs.max()+1],30)
cv2.imwrite('logo/logo_mask.png',out)
print(out.shape,[len(c) for c in simp])
