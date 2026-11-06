import sys; sys.argv=['x','none']
exec(open('render.py').read().split("if __name__ == '__main__':")[0])
fr = load_frames('IMG_9973.MOV', 7.45, 1.0, 1)[0].astype(np.float32)/255
img = grade(affine(fr, 1.08, 0, 60), 'day')
g = np.linspace(0,1,H,dtype=np.float32)[:,None,None]
img = img*(1-0.55*np.clip((g-0.35)/0.5,0,1)) * (1-0.35*np.clip((0.3-g)/0.3,0,1))
img = img*0.85
ov = Overlay()
rgb,a = logo_rgba(120, hexc('#FFFFFF'), hexc('#DDE6F5'))
im = Image.fromarray((np.dstack([rgb,a])*255).astype(np.uint8),'RGBA'); put(ov.im, im, (int(W/2-im.width/2), 400))
kinetic(ov,'BNYAN','Poppins-Bold',44,W/2,570,10,0,track=18)
kinetic(ov,'STEEL','Poppins-Black',300,W/2,1010,10,0,color=ORANGE,glow=0.45,track=6)
kinetic(ov,'THAT BUILDS','Poppins-ExtraBold',104,W/2,1200,10,0)
kinetic(ov,'THE UAE','Poppins-ExtraBold',104,W/2,1320,10,0)
bars(ov, W/2-60, 1420, 22, 1)
img = composite(img, ov)
img = finish(np.clip(img,0,1), 0)
Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save('cover.jpg', quality=95)
