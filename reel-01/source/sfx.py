import numpy as np, scipy.signal as ss, scipy.io.wavfile as wf, sys
sys.argv=['x','still']; 
import importlib.util
spec=importlib.util.spec_from_file_location('r','render.py'); 
SR=48000; N=SR*40
out=np.zeros(N)
rng=np.random.default_rng(1)
def place(sig,t):
    i=int(t*SR); j=min(N,i+len(sig)); 
    if i<0: sig=sig[-i:]; i=0
    out[i:j]+=sig[:j-i]
def bp(x,lo,hi): b,a=ss.butter(2,[lo/(SR/2),hi/(SR/2)],'band'); return ss.lfilter(b,a,x)
def lp(x,f): b,a=ss.butter(2,f/(SR/2)); return ss.lfilter(b,a,x)
def whoosh(d=0.4,amp=0.35):
    n=int(d*SR); t=np.linspace(0,1,n)
    x=rng.normal(0,1,n); env=np.sin(np.pi*t)**2
    # sweeping filter via segments
    y=np.zeros(n); seg=n//20
    for k in range(20):
        f=600+3000*np.sin(np.pi*k/19)
        y[k*seg:(k+1)*seg]=bp(x,f*0.6,f*1.4)[k*seg:(k+1)*seg]
    return y*env*amp
def impact(amp=0.9):
    n=int(1.6*SR); t=np.arange(n)/SR
    f=60*np.exp(-t*2)+32
    s=np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*2.6)
    nz=lp(rng.normal(0,1,n),900)*np.exp(-t*18)*0.6
    return (s+nz)*amp
def riser(d=1.4,amp=0.35):
    n=int(d*SR); t=np.arange(n)/SR; p=t/d
    f=180+1400*p**2
    s=np.sin(2*np.pi*np.cumsum(f)/SR)*0.4
    x=rng.normal(0,1,n); y=np.zeros(n); seg=n//30
    for k in range(30):
        fc=500+6000*(k/29)**2
        y[k*seg:(k+1)*seg]=bp(x,fc*0.7,min(fc*1.3,23000))[k*seg:(k+1)*seg]
    return (s+y*0.8)*p**2*amp
B=[0.0,0.58,1.16,1.72,2.3,2.85,3.43,3.99,4.57]
place(riser(1.45),4.57-1.45)
place(impact(1.0),4.57)
whips=[2.287,6.87,9.13,11.43,12.58,16.02,18.30,20.56]  # whip-in cuts
for w in whips: place(whoosh(0.36,0.25),w-0.2)
for z in [1.16,7.99,10.28,13.72,17.13,21.71]: place(whoosh(0.25,0.15),z-0.13)
place(whoosh(0.6,0.35),25.15-0.35)
place(riser(0.9,0.22),34.85-0.9)
place(impact(1.0),34.85)
place(whoosh(0.5,0.3),31.41-0.3)
out=np.tanh(out*0.9)
st=np.stack([out,out],1)
wf.write('sfx.wav',SR,(st*0.5*32767).astype(np.int16))
