import numpy as np, cv2, subprocess, json, math, os, sys
from PIL import Image, ImageDraw, ImageFont

D = os.path.dirname(os.path.abspath(__file__))
SRC = D + '/src/'
W, H, FPS = 1080, 1920, 30
DUR = 40.0
NF = int(DUR * FPS)

# ---------------------------------------------------------------- music grid
OFF = 11.96
BS = [11.96, 12.54, 13.12, 13.68, 14.26, 14.81, 15.39, 15.95, 16.53, 17.11, 17.67, 18.25, 18.83, 19.39, 19.95,
      20.53, 21.11, 21.69, 22.24, 22.83, 23.38, 23.96, 24.54, 25.1, 25.68, 26.26, 26.82, 27.4, 27.98, 28.54, 29.09,
      29.68, 30.26, 30.81, 31.39, 31.97, 32.53, 33.11, 33.67, 34.25, 34.83, 35.39, 35.97, 36.52, 37.11, 37.66,
      38.24, 38.82, 39.38, 39.96, 40.54, 41.1, 41.68, 42.24, 42.82, 43.37, 43.96, 44.54, 45.09, 45.67, 46.25,
      46.81, 47.39, 47.95, 48.53, 49.11, 49.67, 50.25, 50.81, 51.39, 51.97]
B = [round(b - OFF, 3) for b in BS]          # beat times in video seconds
DROP = B[8]                                    # 4.57
BRK = B[44]                                    # 25.15 breakdown
HITS = [26.84, 28.00, 29.14, 30.28, 31.41, 32.58, 33.71, 34.85, 35.99, 37.15, 38.29, 39.43]

# ---------------------------------------------------------------- colors (RGB 0..1)
def hexc(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
NAVY = hexc('#071731'); BLUE = hexc('#2B56A8'); BRIGHT = hexc('#2F6BFF'); ORANGE = hexc('#F2A81D')
WHITE = np.ones(3, np.float32)

# ---------------------------------------------------------------- easing
def clamp(x, a=0., b=1.): return max(a, min(b, x))
def eoc(x): x = clamp(x); return 1 - (1 - x) ** 3
def eoq(x): x = clamp(x); return 1 - (1 - x) ** 5
def eic(x): x = clamp(x); return x ** 3
def eio(x): x = clamp(x); return 3 * x * x - 2 * x ** 3
def eob(x, s=1.7):
    x = clamp(x); x -= 1; return x * x * ((s + 1) * x + s) + 1
def lerp(a, b, t): return a + (b - a) * t

# ---------------------------------------------------------------- fonts
_fc = {}
def font(name, size):
    k = (name, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(D + '/fonts/' + name + '.ttf', size)
    return _fc[k]

# ---------------------------------------------------------------- precomputed textures
rng = np.random.default_rng(7)
GRAIN = [cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H)) for _ in range(6)]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rr = ((xx - W / 2) / (W / 2)) ** 2 * 0.8 + ((yy - H / 2) / (H / 2)) ** 2
VIGN = (1 - 0.42 * np.clip(rr, 0, 1.6) ** 1.3)[..., None].astype(np.float32)
del rr

# ---------------------------------------------------------------- logo
LOGO = cv2.imread(D + '/logo/logo_mask.png', 0).astype(np.float32) / 255
_lab = cv2.connectedComponents((LOGO > 0.5).astype(np.uint8))[1]
LOGO_PARTS = []
for i in (1, 2, 3):
    m = cv2.dilate((_lab == i).astype(np.uint8), np.ones((5, 5), np.uint8))
    LOGO_PARTS.append(LOGO * m)

def logo_rgba(h, color_top=hexc('#4A7BE0'), color_bot=hexc('#1D3F8E'), part=None):
    src = LOGO if part is None else LOGO_PARTS[part]
    w = int(src.shape[1] * h / src.shape[0])
    a = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    g = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    rgb = color_top * (1 - g) + color_bot * g
    rgb = np.broadcast_to(rgb, (h, w, 3)).copy()
    return rgb, a

# ---------------------------------------------------------------- video decoding
def load_frames(fname, t_in, speed, n):
    path = SRC + fname
    ww, hh = map(int, subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                                'stream=width,height', '-of', 'csv=p=0', path]).decode().split(',')[:2])
    s = max(W / ww, H / hh)
    sw, sh = int(math.ceil(ww * s / 2) * 2), int(math.ceil(hh * s / 2) * 2)
    need = n / FPS * speed + 0.3
    vf = f'trim=duration={need:.3f},setpts=(PTS-STARTPTS)/{speed},fps={FPS},scale={sw}:{sh}:flags=lanczos,crop={W}:{H}'
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{t_in:.3f}', '-i', path, '-an', '-vf', vf, '-frames:v', str(n),
           '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    raw = subprocess.check_output(cmd)
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    if len(fr) < n:
        fr = np.concatenate([fr, np.repeat(fr[-1:], n - len(fr), 0)])
    return fr

# ---------------------------------------------------------------- grading
def grade(x, mode):
    lum = (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114)[..., None]
    if mode == 'night':
        x = x * 1.08 - 0.03
        x = x + np.array([-0.02, 0.025, 0.07], np.float32) * (1 - lum) ** 2 + np.array([0.08, 0.03, -0.04], np.float32) * lum ** 2
        sat = 1.18
    elif mode == 'dark':
        x = x * 0.55
        sat = 0.6
    else:
        x = x * 1.06 - 0.025
        x = x + np.array([-0.03, 0.02, 0.05], np.float32) * (1 - lum) ** 2 + np.array([0.07, 0.03, -0.035], np.float32) * lum ** 2
        sat = 1.1
    lum = (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114)[..., None]
    x = lum + (x - lum) * sat
    x = np.clip(x, 0, 1)
    sc = x * x * (3 - 2 * x)
    return x * 0.62 + sc * 0.38

def finish(x, fi):
    x = x * VIGN
    x = x + GRAIN[fi % 6][..., None] * 0.022
    return x

# ---------------------------------------------------------------- image ops
def affine(img, z, dx=0, dy=0, rot=0):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, z)
    M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def zoom_blur(img, s, k=6):
    acc = img.copy()
    for i in range(1, k):
        acc += affine(img, 1 + s * i / (k - 1))
    return acc / k

def dir_blur(img, L, vertical=False):
    L = int(L)
    if L < 3: return img
    return cv2.blur(img, (1, L) if vertical else (L, 1))

def rgb_split(img, d):
    d = int(d)
    if d < 1: return img
    o = img.copy()
    o[..., 0] = np.roll(img[..., 0], d, 1)
    o[..., 2] = np.roll(img[..., 2], -d, 1)
    return o

def glitch_slices(img, amt, seed):
    r = np.random.default_rng(seed)
    o = img.copy()
    for _ in range(int(6 * amt) + 1):
        y0 = r.integers(0, H - 40); hgt = r.integers(10, 120)
        o[y0:y0 + hgt] = np.roll(o[y0:y0 + hgt], int(r.normal(0, 60 * amt)), 1)
    return o

def light_leak(t, strength, seed=0):
    r = np.random.default_rng(seed)
    cx = W * (0.2 + 0.6 * r.random()) + math.sin(t * 2 + seed) * 200
    cy = H * (0.2 + 0.6 * r.random()) + math.cos(t * 1.5 + seed) * 300
    g = np.exp(-(((xx - cx) / 520) ** 2 + ((yy - cy) / 700) ** 2))[..., None]
    return g * np.array([1.0, 0.55, 0.15], np.float32) * strength

# ---------------------------------------------------------------- overlay (PIL) helpers
class Overlay:
    def __init__(self):
        self.im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        self.glow = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
        self.g = ImageDraw.Draw(self.glow)
        self.sh = Image.new('L', (W, H), 0)
        self.used = False; self.gused = False; self.sused = False

def rgba(c, a=1.0):
    return tuple(int(v * 255) for v in c) + (int(255 * clamp(a)),)

def put(dst, src, pos):
    x, y = pos
    l, t = max(0, -x), max(0, -y)
    r, b = min(src.width, W - x), min(src.height, H - y)
    if r <= l or b <= t: return
    dst.alpha_composite(src.crop((l, t, r, b)), dest=(x + l, y + t))

def text_width(txt, f, track):
    w = 0
    for ch in txt:
        w += f.getlength(ch) + track
    return w - track

def kinetic(ov, txt, fname, size, cx, cy, t, t0, t1=None, color=WHITE, track=0, stagger=0.035, dur=0.38,
            mode='rise', align='center', glow=0.0, out_dur=0.22, shadow=True):
    """per-letter animated text. cy = baseline-ish center."""
    if t < t0: return
    f = font(fname, size)
    asc, desc = f.getmetrics()
    tw = text_width(txt, f, track)
    x = cx - tw / 2 if align == 'center' else (cx - tw if align == 'right' else cx)
    lh = asc + desc
    band = Image.new('RGBA', (int(tw + size * 2), int(lh * 1.25)), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    gband = Image.new('RGBA', band.size, (0, 0, 0, 0)) if glow > 0 else None
    gd = ImageDraw.Draw(gband) if gband else None
    oa = 1.0
    if t1 is not None and t > t1:
        oa = 1 - clamp((t - t1) / out_dur)
        if oa <= 0: return
    px = size
    n = len(txt)
    for i, ch in enumerate(txt):
        p = clamp((t - t0 - i * stagger) / dur)
        if mode == 'rise':
            e = eoq(p); dy = (1 - e) * lh * 0.95; a = clamp(p * 2.5)
        elif mode == 'drop':
            e = eob(p, 1.3); dy = -(1 - e) * lh * 0.9; a = clamp(p * 3)
        else:  # fade
            e = eoc(p); dy = 0; a = e
        if t1 is not None and t > t1:
            dy -= eic(clamp((t - t1) / out_dur)) * lh * 0.6
        if a > 0:
            bd.text((px, int(lh * 0.1 + dy)), ch, font=f, fill=rgba(color, a * oa))
            if gd: gd.text((px, int(lh * 0.1 + dy)), ch, font=f, fill=rgba(color, a * oa * glow))
        px += f.getlength(ch) + track
    pos = (int(x - size), int(cy - lh * 0.6))
    put(ov.im, band, pos)
    if shadow:
        ov.sh.paste(255, (max(0, pos[0]), max(0, pos[1])), band.getchannel('A').crop((max(0, -pos[0]), max(0, -pos[1]), band.width, band.height)))
        ov.sused = True
    ov.used = True
    if gband:
        put(ov.glow, gband, pos); ov.gused = True

def bars(ov, x, y, s, a=1.0, prog=1.0, color=ORANGE):
    """brand double slanted bars (=)"""
    for k in range(2):
        p = clamp(prog * 2 - k * 0.6)
        if p <= 0: continue
        w = s * 2.2 * eoc(p); h = s * 0.42
        yy0 = y + k * s * 0.62
        sk = h * 0.6
        ov.d.polygon([(x + sk, yy0), (x + sk + w, yy0), (x + w, yy0 + h), (x, yy0 + h)], fill=rgba(color, a))
    ov.used = True

def pill(ov, cx, cy, txt, size, a, fill=BRIGHT, fg=WHITE, fname='Poppins-SemiBold', dot=True):
    f = font(fname, size)
    tw = f.getlength(txt)
    pad = size * 0.7
    dw = size * 1.0 if dot else 0
    w = tw + pad * 2 + dw; h = size * 1.75
    x0, y0 = cx, cy - h / 2
    ov.d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=h / 2, fill=rgba(fill, 0.92 * a),
                           outline=rgba(hexc('#9CC0FF'), 0.8 * a), width=2)
    if dot:
        r = size * 0.26
        ov.d.ellipse([x0 + pad * 0.8 - r, cy - r, x0 + pad * 0.8 + r, cy + r], fill=rgba(WHITE, a))
    ov.d.text((x0 + pad + dw * 0.55, cy - size * 0.62), txt, font=f, fill=rgba(fg, a))
    ov.used = True
    return w

def composite(img, ov, glow_strength=1.0):
    if ov.sused:
        sa = np.asarray(ov.sh, np.float32) / 255
        sm = cv2.GaussianBlur(cv2.resize(sa, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 6)
        sm = cv2.resize(sm, (W, H))
        img = img * (1 - np.clip(sm * 1.4, 0, 0.75))[..., None]
    if ov.gused:
        g = np.asarray(ov.glow, np.float32) / 255
        small = cv2.resize(g, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 7)
        gl = cv2.resize(small, (W, H))
        img = img + gl[..., :3] * gl[..., 3:4] * 1.6 * glow_strength
    if ov.used:
        o = np.asarray(ov.im, np.float32) / 255
        a = o[..., 3:4]
        img = img * (1 - a) + o[..., :3] * a
    return img

# ---------------------------------------------------------------- HUD
def hud(ov, t, a):
    if a <= 0: return
    f = font('Poppins-Medium', 24)
    ov.d.ellipse([72, 236, 86, 250], fill=rgba(ORANGE, a))
    ov.d.text((98, 228), 'BNYAN  |  STEEL SUPPLY', font=f, fill=rgba(WHITE, 0.85 * a))
    s = '25.20°N  55.27°E'
    ov.d.text((W - 72 - f.getlength(s), 228), s, font=f, fill=rgba(WHITE, 0.7 * a))
    # progress line
    p = clamp(t / DUR)
    ov.d.rectangle([72, 272, W - 72, 274], fill=rgba(WHITE, 0.18 * a))
    ov.d.rectangle([72, 272, 72 + (W - 144) * p, 274], fill=rgba(ORANGE, 0.9 * a))
    ov.used = True

# ================================================================ EDL
def S(t0, t1, f, tin, sp, mode='day', ti='cut', to=None, z=(1.04, 1.12), **kw):
    d = dict(t0=t0, t1=t1, f=f, tin=tin, sp=sp, mode=mode, ti=ti, to=to, z=z); d.update(kw); return d

b = B
SHOTS = [
    # ---- HOOK (night, moody)
    S(0.0, b[2], 'IMG_9944.MOV', 2.0, 0.8, 'night', 'fadein', z=(1.16, 1.03)),
    S(b[2], b[4], 'IMG_9949.MOV', 12.5, 1.0, 'night', 'zoom', z=(1.0, 1.1)),
    S(b[4], b[6], 'IMG_9942.MOV', 5.6, 1.0, 'night', 'whip', z=(1.05, 1.14)),
    S(b[6], b[7], 'IMG_9941.MOV', 1.0, 2.0, 'night', 'glitch', z=(1.1, 1.2)),
    S(b[7], (b[7] + b[8]) / 2, 'IMG_9944.MOV', 5.2, 1.5, 'night', 'zoom', z=(1.1, 1.25)),
    S((b[7] + b[8]) / 2, b[8], 'IMG_9949.MOV', 17.0, 1.5, 'night', 'zoom', to='zoomflash', z=(1.15, 1.45)),
    # ---- DROP
    S(b[8], b[12], 'IMG_9973.MOV', 6.95, 0.8, 'day', 'drop', z=(1.18, 1.02)),
    S(b[12], b[14], 'IMG_9972.MOV', 6.0, 1.0, 'day', 'whip'),
    S(b[14], b[16], 'IMG_9984.MOV', 12.2, 1.3, 'day', 'zoom', z=(1.12, 1.03)),
    S(b[16], b[18], 'IMG_9971.MOV', 7.3, 0.8, 'day', 'whip', label=('01', 'REBAR')),
    S(b[18], b[20], 'IMG_9986.MOV', 6.2, 1.0, 'day', 'zoom'),
    S(b[20], b[22], 'IMG_9993.MOV', 0.8, 0.9, 'day', 'whipv', label=('02', 'MS ROUND BARS')),
    S(b[22], b[24], 'IMG_9590.MOV', 5.0, 1.2, 'day', 'whip', label=('03', 'HOLLOW SECTIONS')),
    S(b[24], b[26], 'IMG_9120.MOV', 2.5, 1.0, 'day', 'zoom', label=('04', 'EQUAL ANGLES')),
    S(b[26], b[28], 'IMG_9965.MOV', 3.5, 1.0, 'day', 'flash', big='FROM PORT'),
    S(b[28], b[30], 'IMG_9983.MOV', 9.3, 1.0, 'day', 'whip', big='TO SITE.', accent=True),
    S(b[30], b[32], 'IMG_9971.MOV', 0.3, 0.9, 'day', 'zoom', big='ON TIME.'),
    S(b[32], b[34], 'IMG_9967.MOV', 1.0, 0.8, 'day', 'whip', big='EVERY TIME.', accent=True),
    S(b[34], b[36], 'IMG_9942.MOV', 0.2, 1.0, 'night', 'glitch', big='DAY'),
    S(b[36], b[38], 'IMG_9941.MOV', 3.5, 1.2, 'night', 'whip', big='& NIGHT.'),
    S(b[38], b[40], 'IMG_9944.MOV', 4.5, 1.0, 'night', 'zoom', big='NONSTOP.', accent=True),
]
_mont = [('IMG_9973.MOV', 3.3), ('IMG_9984.MOV', 17.5), ('IMG_9971.MOV', 15.5), ('IMG_9986.MOV', 12.3),
         ('IMG_9972.MOV', 12.5), ('IMG_9590.MOV', 8.0), ('IMG_9993.MOV', 3.0), ('IMG_9965.MOV', 8.0)]
for i, (f, tin) in enumerate(_mont):
    t0 = b[40] + i * (b[44] - b[40]) / 8; t1 = b[40] + (i + 1) * (b[44] - b[40]) / 8
    SHOTS.append(S(t0, t1, f, tin, 1.0, 'day', 'zoom' if i % 2 == 0 else 'flash', z=(1.25, 1.05),
                   to='zoomflash' if i == 7 else None))
SHOTS.append(dict(t0=BRK, t1=HITS[4], kind='map'))
SHOTS += [
    S(HITS[4], HITS[5], 'IMG_9973.MOV', 3.0, 0.6, 'day', 'flash', big='QUALITY.', z=(1.02, 1.1), calm=True),
    S(HITS[5], HITS[6], 'IMG_9984.MOV', 17.0, 0.6, 'day', 'whip', big='TRUST.', z=(1.02, 1.1), calm=True),
    S(HITS[6], b[59], 'IMG_9986.MOV', 12.0, 0.6, 'day', 'zoom', big='SPEED.', accent=True, z=(1.02, 1.1), calm=True),
]
SHOTS.append(dict(t0=b[59], t1=DUR, kind='end'))

def in_drop(t): return DROP <= t < BRK

# ================================================================ video shot renderer
class VideoShot:
    def __init__(self, sh):
        self.s = sh
        self.n = int(round((sh['t1'] - sh['t0']) * FPS)) + 2
        self.fr = load_frames(sh['f'], sh['tin'], sh['sp'], self.n)

    def render(self, t, fi):
        s = self.s; lt = t - s['t0']; dur = s['t1'] - s['t0']; rt = s['t1'] - t
        i = min(int(lt * FPS + 0.5), self.n - 1)
        img = self.fr[i].astype(np.float32) / 255
        z = lerp(s['z'][0], s['z'][1], eio(lt / dur))
        dx = dy = rot = 0.0
        # beat punch + shake inside drop
        if in_drop(t) and not s.get('calm'):
            lb = max([x for x in B if x <= t + 1e-6], default=0)
            k = math.exp(-(t - lb) * 9)
            z += 0.035 * k
            dx += math.sin(fi * 2.7) * 9 * k; dy += math.cos(fi * 3.1) * 9 * k
        ti, to = s['ti'], s['to']
        TI = 0.2
        if ti in ('whip',) and lt < TI:
            p = lt / TI; dx += (1 - eoc(p)) * 420; rot += (1 - eoc(p)) * -3
        if ti == 'whipv' and lt < TI:
            p = lt / TI; dy += (1 - eoc(p)) * 600
        if ti in ('zoom', 'drop') and lt < TI:
            z *= 1 + (1 - eoc(lt / TI)) * (0.35 if ti == 'drop' else 0.18)
        if to == 'zoomflash' and rt < 0.25:
            z *= 1 + eic(1 - rt / 0.25) * 0.5
        img = affine(img, z, dx, dy, rot)
        img = grade(img, s['mode'])
        # transitions on pixels
        if ti in ('whip',) and lt < TI:
            img = dir_blur(img, (1 - lt / TI) * 160)
        if ti == 'whipv' and lt < TI:
            img = dir_blur(img, (1 - lt / TI) * 200, True)
        if ti in ('zoom', 'drop') and lt < TI:
            img = zoom_blur(img, (1 - eoc(lt / TI)) * (0.22 if ti == 'drop' else 0.12))
        if to == 'zoomflash' and rt < 0.25:
            img = zoom_blur(img, eic(1 - rt / 0.25) * 0.25)
            img = img + eic(1 - rt / 0.25) * 0.9
        if ti == 'glitch' and lt < 0.25:
            a = 1 - lt / 0.25
            img = glitch_slices(rgb_split(img, a * 30), a, fi)
        if ti == 'drop':
            if lt < 0.3: img = rgb_split(img, (1 - lt / 0.3) * 26)
            img = img + max(0, 1 - lt / 0.35) ** 2 * 1.0
        if ti == 'flash' and lt < 0.2:
            img = img + (1 - lt / 0.2) ** 2 * 0.7
        if ti == 'fadein':
            img = img * eoc(lt / 0.5)
        # light leaks at a few cuts
        if in_drop(t) and lt < 0.45 and ti in ('whip', 'flash'):
            img = img + light_leak(t, 0.45 * (1 - lt / 0.45), seed=int(s['t0'] * 10))
        return img

    def overlay(self, ov, t):
        s = self.s; lt = t - s['t0']
        if 'label' in s:
            num, name = s['label']
            t0 = s['t0'] + 0.06
            x0, y0 = 84, 1250
            p = clamp((t - t0) / 0.35)
            kinetic(ov, num + ' / 04', 'Poppins-SemiBold', 30, x0, y0, t, t0, color=ORANGE, align='left', track=3)
            bars(ov, x0, y0 + 50, 26, 1, prog=clamp((t - t0) / 0.4))
            kinetic(ov, name, 'Poppins-ExtraBold', 92 if len(name) < 12 else 74, x0 + 84, y0 + 66, t, t0 + 0.05,
                    align='left', stagger=0.022, dur=0.3)
            ov.d.rectangle([x0, y0 + 140, x0 + (W - 2 * x0) * eoq(clamp((t - t0 - 0.1) / 0.5)), y0 + 143],
                           fill=rgba(WHITE, 0.7))
        if 'big' in s:
            txt = s['big']
            col = ORANGE if s.get('accent') else WHITE
            size = 150 if len(txt) <= 6 else (124 if len(txt) <= 9 else 108)
            kinetic(ov, txt, 'Poppins-Black', size, W / 2, 900, t, s['t0'] + 0.02, color=col, stagger=0.03,
                    dur=0.3, glow=0.55 if s.get('accent') else 0.25, mode='drop' if s.get('calm') else 'rise')

# ================================================================ hook text
def hook_text(ov, t):
    if t >= b[7] + 0.02: return
    tout = b[6] + 0.25
    kinetic(ov, 'EVERY SKYLINE', 'Poppins-ExtraBold', 92, W / 2, 780, t, 0.25, tout, stagger=0.03)
    kinetic(ov, 'STARTS WITH', 'Poppins-Light', 70, W / 2, 890, t, b[2] + 0.02, tout, stagger=0.03, track=6)
    kinetic(ov, 'STEEL.', 'Poppins-Black', 220, W / 2, 1060, t, b[4] + 0.02, tout + 0.1, color=ORANGE, stagger=0.05,
            dur=0.4, glow=0.8, mode='drop', track=4)

def drop_title(ov, t):
    t0 = DROP + 0.05
    if not (t0 <= t < b[12]): return
    out = b[12] - 0.3
    a = 1 - clamp((t - out) / 0.25)
    # logo mark
    p = eob(clamp((t - t0) / 0.4))
    hgt = int(150 * max(p, 0.01))
    if hgt > 4:
        rgb, al = logo_rgba(hgt, hexc('#FFFFFF'), hexc('#DDE6F5'))
        im = Image.fromarray((np.dstack([rgb, al * a]) * 255).astype(np.uint8), 'RGBA')
        put(ov.im, im, (int(W / 2 - im.width / 2), int(700 - hgt / 2)))
    kinetic(ov, 'BNYAN', 'Poppins-Black', 190, W / 2, 900, t, t0 + 0.08, out, stagger=0.045, dur=0.32, track=16,
            glow=0.12)
    kinetic(ov, 'PREMIUM STEEL  ·  UAE', 'Poppins-SemiBold', 40, W / 2, 1040, t, t0 + 0.4, out, stagger=0.02,
            color=ORANGE, track=8)

# ================================================================ map scene
GEO = json.load(open(D + '/countries.geojson'))
def merc(lon, lat):
    lat = max(-80, min(80, lat))
    return lon, math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))
COUNTRIES = []
for ft in GEO['features']:
    g = ft['geometry']; name = ft['properties']['ADMIN']
    polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    rings = []
    for pl in polys:
        ring = np.array([merc(x, y) for x, y in pl[0]], np.float32)
        if ring[:, 0].max() < -30 or ring[:, 0].min() > 160: continue
        rings.append(ring)
    if rings: COUNTRIES.append((name, rings))
MARKETS = {  # name: (lon, lat, label, label side, arrival hit index)
    'Saudi Arabia': (45.0, 24.0, 'Saudi Arabia', 'L', 0),
    'Oman': (57.0, 21.0, 'Oman', 'R', 0),
    'Kuwait': (47.6, 29.4, 'Kuwait', 'L', 0),
    'India': (78.5, 22.0, 'India', 'R', 1),
    'Bangladesh': (90.3, 23.8, 'Bangladesh', 'R', 1),
    'Malaysia': (102.0, 4.0, 'Malaysia', 'R', 1),
    'Somalia': (46.0, 6.0, 'Somalia', 'R', 2),
    'South Africa': (25.0, -29.0, 'South Africa', 'R', 2),
    'Turkey': (35.0, 39.0, 'Türkiye', 'L', 3),
    'Georgia': (43.5, 42.0, 'Georgia', 'R', 3),
    'Lebanon': (35.8, 33.9, 'Lebanon', 'L', 3),
}
DXB = merc(55.27, 25.2)
UAE_CITIES = [('DUBAI', 55.27, 25.2), ('SHARJAH', 55.41, 25.35), ('UMM AL QUWAIN', 55.55, 25.56)]
MAP_CY = 1000

def map_cam(t):
    lt = t - BRK
    c0 = (54.6, merc(55, 24.6)[1]); k0 = 120
    c1 = (63.0, merc(0, 8.0)[1]); k1 = 10.6
    p = eio(clamp((t - HITS[0] + 0.15) / 2.6))
    k = math.exp(lerp(math.log(k0), math.log(k1), p))
    cx = lerp(c0[0], c1[0], p); cy = lerp(c0[1], c1[1], p)
    k *= 1 + 0.02 * lt  # slow drift in
    return cx, cy, k

def proj(pt, cam):
    cx, cy, k = cam
    return (W / 2 + (pt[0] - cx) * k, MAP_CY - (pt[1] - cy) * k)

def render_map(t, fi):
    cam = map_cam(t)
    cx, cy, k = cam
    img = np.zeros((H, W, 3), np.float32)
    g = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img[:] = hexc('#040A18') * (1 - g) + hexc('#0A1A38') * g
    canvas = (img * 255).astype(np.uint8)
    # grid
    for lon in range(-30, 181, 10):
        x = int(W / 2 + (lon - cx) * k)
        if 0 <= x < W: cv2.line(canvas, (x, 0), (x, H), (18, 34, 66), 1, cv2.LINE_AA)
    for lat in range(-60, 81, 10):
        y = int(MAP_CY - (merc(0, lat)[1] - cy) * k)
        if 0 <= y < H: cv2.line(canvas, (0, y), (W, y), (18, 34, 66), 1, cv2.LINE_AA)
    arrived = {}
    for name, (lon, lat, lab, side, hi) in MARKETS.items():
        arrived[name] = clamp((t - (HITS[hi] + 0.55)) / 0.35) if t > HITS[hi] else 0
    land = hexc('#13305F'); hi_c = hexc('#2A6CF0')
    vx0 = cx - W / 2 / k - 5; vx1 = cx + W / 2 / k + 5
    for name, rings in COUNTRIES:
        if name == 'United Arab Emirates':
            col = land * 0.35 + ORANGE * 0.65
        elif name in arrived:
            col = land * (1 - arrived[name]) + hi_c * arrived[name]
        else:
            col = land
        c = tuple(int(v * 255) for v in col)
        for ring in rings:
            if ring[:, 0].max() < vx0 or ring[:, 0].min() > vx1: continue
            pts = np.empty_like(ring)
            pts[:, 0] = W / 2 + (ring[:, 0] - cx) * k
            pts[:, 1] = MAP_CY - (ring[:, 1] - cy) * k
            p16 = np.round(pts * 16).astype(np.int32)
            cv2.fillPoly(canvas, [p16], c, cv2.LINE_AA, shift=4)
            cv2.polylines(canvas, [p16], True, (40, 74, 128), 1, cv2.LINE_AA, shift=4)
    img = canvas.astype(np.float32) / 255
    # glow layer for arcs
    gl = np.zeros((H // 2, W // 2, 3), np.float32)
    lines = np.zeros((H, W, 3), np.uint8)
    d = proj(DXB, cam)
    for name, (lon, lat, lab, side, hi) in MARKETS.items():
        if t < HITS[hi]: continue
        p = eoc((t - HITS[hi]) / 0.6)
        tp = proj(merc(lon, lat), cam)
        mx, my = (d[0] + tp[0]) / 2, (d[1] + tp[1]) / 2
        vx, vy = tp[0] - d[0], tp[1] - d[1]
        L = math.hypot(vx, vy) + 1e-6
        ctrl = (mx - vy / L * L * 0.22 * (1 if vx > 0 else -1), my - abs(vx) / L * L * 0.22 - L * 0.08)
        ts = np.linspace(0, p, 48)
        px = (1 - ts) ** 2 * d[0] + 2 * (1 - ts) * ts * ctrl[0] + ts ** 2 * tp[0]
        py = (1 - ts) ** 2 * d[1] + 2 * (1 - ts) * ts * ctrl[1] + ts ** 2 * tp[1]
        pts = np.round(np.stack([px, py], 1) * 16).astype(np.int32)
        cv2.polylines(lines, [pts], False, (255, 190, 80), 3, cv2.LINE_AA, shift=4)
        hx, hy = px[-1], py[-1]
        cv2.circle(lines, (int(hx), int(hy)), 7, (255, 255, 255), -1, cv2.LINE_AA)
    lf = lines.astype(np.float32) / 255
    gl = cv2.GaussianBlur(cv2.resize(lf, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 4)
    img = img + cv2.resize(gl, (W, H)) * 2.2
    img = np.maximum(img, lf)
    # dubai pulse
    dd = proj(DXB, cam)
    for r_i in range(3):
        ph = ((t - BRK) * 1.1 + r_i / 3) % 1
        rad = 10 + ph * 70
        ring = np.zeros((H, W), np.uint8)
        cv2.circle(ring, (int(dd[0]), int(dd[1])), int(rad), 255, 3, cv2.LINE_AA)
        img = img + (ring.astype(np.float32) / 255)[..., None] * ORANGE * (1 - ph)
    img = grade(np.clip(img, 0, 1), 'map') if False else img
    # transition in from white flash / zoom
    lt = t - BRK
    if lt < 0.3:
        img = zoom_blur(img, (1 - eoc(lt / 0.3)) * 0.2) + (1 - lt / 0.3) ** 2 * 0.9
    rt = HITS[4] - t
    if rt < 0.22:
        img = zoom_blur(affine(img, 1 + eic(1 - rt / 0.22) * 0.6), eic(1 - rt / 0.22) * 0.25) + eic(1 - rt / 0.22) * 0.8
    return img, cam

def map_overlay(ov, t, cam):
    out = HITS[4] - 0.25
    kinetic(ov, 'BASED IN THE UAE', 'Poppins-SemiBold', 40, W / 2, 400, t, BRK + 0.15, HITS[0] + 0.6, color=ORANGE,
            track=8, stagger=0.02)
    kinetic(ov, 'BUILT FOR', 'Poppins-Light', 64, W / 2, 400, t, HITS[0] + 0.9, out, track=10, stagger=0.03)
    kinetic(ov, 'THE REGION', 'Poppins-Black', 104, W / 2, 500, t, HITS[0] + 1.05, out, stagger=0.03)
    # uae city labels in first part
    if t < HITS[0] + 0.4:
        a = clamp((t - BRK - 0.3) / 0.3) * (1 - clamp((t - HITS[0]) / 0.3))
        for i, (nm, lon, lat) in enumerate(UAE_CITIES):
            x, y = proj(merc(lon, lat), cam)
            ov.d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=rgba(WHITE, a))
            a2 = a * clamp((t - BRK - 0.3 - i * 0.25) / 0.3)
            pill(ov, x + 26, y - 40 * (i - 1) * 1.3 - 10, nm, 26, a2, fill=BRIGHT if nm != 'DUBAI' else ORANGE,
                 fg=WHITE if nm != 'DUBAI' else NAVY)
    # market labels
    for name, (lon, lat, lab, side, hi) in MARKETS.items():
        ta = HITS[hi] + 0.55
        if t < ta: continue
        a = eoc((t - ta) / 0.3) * (1 - clamp((t - out) / 0.2))
        x, y = proj(merc(lon, lat), cam)
        ph = clamp((t - ta) / 0.6)
        r = 6 + ph * 26
        ov.d.ellipse([x - r, y - r, x + r, y + r], outline=rgba(WHITE, (1 - ph) * a), width=3)
        ov.d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=rgba(WHITE, a))
        f = font('Poppins-SemiBold', 24); w = f.getlength(lab) + 24 * 0.7 * 2 + 24
        px = x + 14 if side == 'R' else x - 14 - w
        pill(ov, px, y, lab, 24, a)
    # region words at bottom
    regs = ['GCC', 'ASIA', 'AFRICA', '& BEYOND']
    f = font('Poppins-ExtraBold', 54)
    total = sum(f.getlength(r) for r in regs) + 40 * 3
    x = W / 2 - total / 2
    for i, r in enumerate(regs):
        kinetic(ov, r, 'Poppins-ExtraBold', 54, x, 1560, t, HITS[i] + 0.1, out, align='left',
                color=ORANGE if i < 3 else WHITE, stagger=0.02)
        x += f.getlength(r) + 40
    kinetic(ov, 'EXPORT MARKETS', 'Poppins-Medium', 28, W / 2, 1480, t, HITS[0] + 0.1, out, track=12,
            color=hexc('#9CB4DC'), stagger=0.015)

# ================================================================ end scene
class EndScene:
    def __init__(self, sh):
        self.s = sh
        n = int((DUR - sh['t0']) * FPS) + 2
        self.bg = load_frames('IMG_9973.MOV', 2.6, 0.45, n)
        self.n = n
        rgb, a = logo_rgba(470, hexc('#5B8BF0'), hexc('#1E428F'))
        er = cv2.erode(a, np.ones((5, 5), np.uint8))
        rim = np.clip(a - er, 0, 1)
        rim_top = rim * np.linspace(1, 0.2, a.shape[0], dtype=np.float32)[:, None]
        rgb = rgb * (1 - rim_top[..., None] * 0.7) + rim_top[..., None] * 0.7
        self.logo_rgb, self.logo_a = rgb, a
        self.parts = [logo_rgba(470, part=i)[1] for i in range(3)]
        m = (cv2.resize(LOGO, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        self.contours = [c[:, 0, :].astype(np.float32) for c in cs if len(c) > 50]
        r = np.random.default_rng(3)
        self.parts_p = [(r.uniform(0, W), r.uniform(H * 0.3, H * 1.1), r.uniform(-30, 30), r.uniform(120, 420),
                         r.uniform(0.6, 2.2), r.uniform(0, 6)) for _ in range(140)]

    def render(self, t, fi):
        lt = t - self.s['t0']
        i = min(int(lt * FPS), self.n - 1)
        bg = self.bg[i].astype(np.float32) / 255
        bg = cv2.GaussianBlur(cv2.resize(bg, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3)
        bg = cv2.resize(bg, (W, H))
        bg = bg * 0.32 + NAVY * 0.55
        img = bg
        LAND = HITS[7]
        cx, cy = W / 2, 820
        lh, lw = self.logo_a.shape
        x0, y0 = int(cx - lw / 2), int(cy - lh / 2)
        # sparks
        sp = np.zeros((H, W, 3), np.uint8)
        for (px, py, vx, vy, life, ph) in self.parts_p:
            tt = (lt * 0.9 + ph) % life
            x = px + vx * tt; y = py - vy * tt * (1.6 if t > LAND else 1)
            a = (1 - tt / life)
            cv2.line(sp, (int(x), int(y)), (int(x - vx * 0.03), int(y + vy * 0.05)), (int(255 * a), int(150 * a), int(40 * a)), 2, cv2.LINE_AA)
        spf = sp.astype(np.float32) / 255
        img = img + spf + cv2.GaussianBlur(spf, (0, 0), 6) * 1.5
        # outline tracing before landing
        if t < LAND + 0.3:
            line = np.zeros((H, W), np.uint8)
            p = eio(clamp((lt - 0.05) / (LAND - self.s['t0'] - 0.1)))
            for c in self.contours:
                n = len(c); m = int(n * p)
                if m > 2:
                    pts = (c[:m] + [x0, y0]) * 16
                    cv2.polylines(line, [pts.astype(np.int32)], False, 255, 3, cv2.LINE_AA, shift=4)
            fade = 1 - clamp((t - LAND) / 0.3)
            lf = line.astype(np.float32)[..., None] / 255 * fade
            img = img + lf * hexc('#9CC0FF') + cv2.GaussianBlur(lf[..., 0], (0, 0), 8)[..., None] * BRIGHT * 2
        # logo fill slam
        if t >= LAND - 0.12:
            p = clamp((t - LAND + 0.12) / 0.12)
            sc = lerp(1.35, 1.0, eoc(p)) * (1 + 0.035 * eio(clamp((t - LAND) / (DUR - LAND))))
            a = self.logo_a; rgb = self.logo_rgb
            # shine sweep
            st = clamp((t - LAND - 0.45) / 0.7)
            if 0 < st < 1:
                hh, ww = a.shape
                yy_, xx_ = np.mgrid[0:hh, 0:ww].astype(np.float32)
                band = np.exp(-(((xx_ + yy_ * 0.6) - lerp(-400, ww + hh * 0.6 + 400, st)) / 70) ** 2)
                rgb = rgb + band[..., None] * 0.9
            layer = np.zeros((H, W, 3), np.float32); al = np.zeros((H, W), np.float32)
            layer[y0:y0 + lh, x0:x0 + lw] = rgb; al[y0:y0 + lh, x0:x0 + lw] = a * p
            if sc != 1.0:
                M = cv2.getRotationMatrix2D((cx, cy), 0, sc)
                layer = cv2.warpAffine(layer, M, (W, H)); al = cv2.warpAffine(al, M, (W, H))
            glow = cv2.GaussianBlur(cv2.resize(al, (W // 4, H // 4)), (0, 0), 10)
            img = img + cv2.resize(glow, (W, H))[..., None] * BRIGHT * 0.45
            img = img * (1 - al[..., None]) + layer * al[..., None]
        # impact
        if t >= LAND:
            k = t - LAND
            ring = np.zeros((H, W), np.uint8)
            rad = int(240 + eoc(k / 0.8) * 700)
            cv2.circle(ring, (int(cx), int(cy)), rad, 255, 6, cv2.LINE_AA)
            img = img + (ring.astype(np.float32) / 255)[..., None] * (1 - clamp(k / 0.8)) * hexc('#BFD4FF')
            img = img + max(0, 1 - k / 0.25) ** 2 * 0.85
            if k < 0.4:
                s = (1 - k / 0.4) * 20
                img = affine(img, 1 + s * 0.003, math.sin(fi * 2.3) * s, math.cos(fi * 1.7) * s)
                img = rgb_split(img, s)
        # entry flash from previous shot
        if lt < 0.12:
            img = img + (1 - lt / 0.12) * 0.8
        # fade out
        if t > DUR - 0.45:
            img = img * (1 - eic((t - (DUR - 0.45)) / 0.45))
        return img

    def overlay(self, ov, t):
        a = 1 - clamp((t - (DUR - 0.45)) / 0.45)
        kinetic(ov, 'BNYAN', 'Poppins-Black', 150, W / 2, 1180, t, HITS[8], color=WHITE, stagger=0.05, track=26, glow=0.35)
        kinetic(ov, 'AL MUSTAGBAL BUILDING MATERIALS', 'Poppins-Medium', 34, W / 2, 1290, t, HITS[9], track=4,
                stagger=0.012, color=hexc('#C9D6EE'))
        kinetic(ov, 'TRADING CO. L.L.C.', 'Poppins-Medium', 34, W / 2, 1340, t, HITS[9] + 0.15, track=4,
                stagger=0.012, color=hexc('#C9D6EE'))
        if t >= HITS[10]:
            p = eob(clamp((t - HITS[10]) / 0.45), 1.4)
            f = font('Poppins-Bold', 46); txt = 'bnyanco.ae'
            tw = f.getlength(txt); w = (tw + 150) * p; h = 92
            x0 = W / 2 - w / 2; y0 = 1440
            if w > 20:
                ov.d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=h / 2, fill=rgba(ORANGE, a))
            if p > 0.6:
                ta = clamp((p - 0.6) / 0.4) * a
                ov.d.text((W / 2 - tw / 2 + 22, y0 + 14), txt, font=f, fill=rgba(NAVY, ta))
                bars(ov, W / 2 - tw / 2 - 52, y0 + 30, 15, ta, color=NAVY)
            ov.used = True

# ================================================================ frame renderer
def build(shot):
    if shot.get('kind') == 'map': return 'map'
    if shot.get('kind') == 'end': return EndScene(shot)
    return VideoShot(shot)

def render_frame(fi, obj, shot):
    t = fi / FPS
    ov = Overlay()
    if obj == 'map':
        img, cam = render_map(t, fi)
        map_overlay(ov, t, cam)
    else:
        img = obj.render(t, fi)
        obj.overlay(ov, t)
    hook_text(ov, t)
    drop_title(ov, t)
    if DROP <= t < BRK:
        hud(ov, t, 1.0)
    img = composite(np.clip(img, 0, None), ov)
    img = finish(np.clip(img, 0, 1), fi)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def shots_for_range(a, b_):
    out = []
    for s in SHOTS:
        f0, f1 = int(round(s['t0'] * FPS)), int(round(s['t1'] * FPS))
        if f1 > a and f0 < b_: out.append((s, max(f0, a), min(f1, b_)))
    return out

def render_range(a, b_, out_path):
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                            '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '12',
                            '-pix_fmt', 'yuv420p', out_path], stdin=subprocess.PIPE)
    for s, f0, f1 in shots_for_range(a, b_):
        obj = build(s)
        for fi in range(f0, f1):
            enc.stdin.write(render_frame(fi, obj, s).tobytes())
        del obj
    enc.stdin.close(); enc.wait()

if __name__ == '__main__':
    if sys.argv[1] == 'still':
        for tt in map(float, sys.argv[2:]):
            fi = int(round(tt * FPS))
            s, _, _ = shots_for_range(fi, fi + 1)[0]
            obj = build(s)
            cv2.imwrite(D + f'/stills/{tt:06.2f}.jpg', cv2.cvtColor(render_frame(fi, obj, s), cv2.COLOR_RGB2BGR))
    elif sys.argv[1] == 'range':
        render_range(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
