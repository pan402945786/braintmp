import json, math, pathlib
from PIL import Image, ImageDraw, ImageFont

BASE = pathlib.Path(r'D:\work\braintmp')
OUT = BASE / 'visuals_2026-11-24'
OUT.mkdir(exist_ok=True)
EDGE_PATH = BASE / 'runs' / '2026-11-23-formal-a' / 'seed-20261123' / 'joint-edges.json'
DATA = json.loads(EDGE_PATH.read_text(encoding='utf-8'))
EDGES = DATA['edges']
LIVE = [e for e in EDGES if e.get('deleted') is None]
INITIAL = [e for e in EDGES if e.get('born', 0) == 0 and e.get('state') == 'initial']
NEW_LIVE = [e for e in LIVE if e.get('born', 0) > 0]

# Fonts: use a readable fallback where available.
def get_font(size, bold=False):
    candidates = [
        r'C:\Windows\Fonts\segoeui.ttf' if not bold else r'C:\Windows\Fonts\segoeuib.ttf',
        r'C:\Windows\Fonts\arial.ttf' if not bold else r'C:\Windows\Fonts\arialbd.ttf',
    ]
    for p in candidates:
        if pathlib.Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

# ---------- SVG network diagrams ----------
W, H = 760, 720
CX, CY, R = 380, 345, 245
positions = {}
for n in range(32):
    a = -math.pi/2 + 2*math.pi*n/32
    positions[n] = (CX + R*math.cos(a), CY + R*math.sin(a))

def esc(s):
    return (str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;'))

def edge_path(e):
    x1,y1 = positions[e['i']]; x2,y2 = positions[e['j']]
    if e['i'] == e['j']:
        return f'M {x1-2:.1f},{y1-12:.1f} C {x1-35:.1f},{y1-55:.1f} {x1+35:.1f},{y1-55:.1f} {x1+2:.1f},{y1-12:.1f}'
    dx,dy=x2-x1,y2-y1; d=math.hypot(dx,dy) or 1
    nx,ny=-dy/d,dx/d
    # Bend toward the circle center for legibility; farther bends for long chords.
    mx,my=(x1+x2)/2,(y1+y2)/2
    bend=(0.10 + 0.18*min(1.0,d/(2*R))) * R
    # Alternate bend sign by edge id for parallel/opposite edges.
    sign = 1 if (e.get('id',0)%2==0) else -1
    qx,qy=mx+sign*nx*bend, my+sign*ny*bend
    return f'M {x1:.1f},{y1:.1f} Q {qx:.1f},{qy:.1f} {x2:.1f},{y2:.1f}'

def make_svg(kind):
    if kind=='initial':
        edges=INITIAL
        title='初始网络结构 / Initial network'
        subtitle='32 units · 128 directed connections · 64 backbone + 64 extra'
    else:
        edges=LIVE
        title='训练后网络结构 / Learned network'
        subtitle='32 units · 192 live connections · 64 retained additions'
    out=[]
    out.append(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
 <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#4f8bd6"/></marker>
 <marker id="arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa4b2"/></marker>
 <marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#e58a3a"/></marker>
 <marker id="arrow-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9b6bd3"/></marker>
 <style>text{{font-family:Segoe UI,Arial,sans-serif;fill:#1f2937}} .edge{{fill:none;stroke-linecap:round}} .node{{fill:#f8fafc;stroke:#334155;stroke-width:2}}</style>
</defs>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="42" font-size="26" font-weight="600">{esc(title)}</text>
<text x="32" y="68" font-size="15" fill="#64748b">{esc(subtitle)}</text>
''')
    # subtle ring guides
    out.append(f'<circle cx="{CX}" cy="{CY}" r="{R}" fill="none" stroke="#e2e8f0" stroke-width="1"/>')
    for e in edges:
        w=abs(float(e.get('birthWeight',0) if kind=='initial' else e.get('w',0)))
        width=0.7 + min(2.7, w/1.7*2.2)
        if kind=='initial':
            color='#4f8bd6' if e.get('backbone') else '#a7b0bc'
            marker='arrow-blue' if e.get('backbone') else 'arrow-gray'
            opacity='0.62' if e.get('backbone') else '0.34'
            dash=''
        else:
            if e.get('born',0)==0:
                color='#94a3b8'; marker='arrow-gray'; opacity='0.28'; dash=''
                width=min(width,1.4)
            elif e.get('state')=='mature':
                color='#e58a3a' if float(e.get('w',0))>=0 else '#9b6bd3'
                marker='arrow-orange' if float(e.get('w',0))>=0 else 'arrow-purple'
                opacity='0.82'; dash=''
            else:
                color='#64748b'; marker='arrow-gray'; opacity='0.65'; dash='5 4'
        out.append(f'<path class="edge" d="{edge_path(e)}" stroke="{color}" stroke-width="{width:.2f}" opacity="{opacity}" marker-end="url(#{marker})"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>' )
    for n,(x,y) in positions.items():
        out.append(f'<circle class="node" cx="{x:.1f}" cy="{y:.1f}" r="16"/>')
        out.append(f'<text x="{x:.1f}" y="{y+5:.1f}" font-size="12" text-anchor="middle">{n}</text>')
    # legend
    ly=H-82
    if kind=='initial':
        items=[('#4f8bd6','backbone ring'),('#a7b0bc','extra initial edge')]
    else:
        items=[('#94a3b8','surviving initial edge'),('#e58a3a','mature positive-weight edge'),('#9b6bd3','mature negative-weight edge'),('#64748b','trial edge')]
    x=32
    for color,label in items:
        dash=' stroke-dasharray="5 4"' if label=='trial edge' else ''
        out.append(f'<line x1="{x}" y1="{ly}" x2="{x+30}" y2="{ly}" stroke="{color}" stroke-width="3"{dash}/><text x="{x+40}" y="{ly+5}" font-size="13">{esc(label)}</text>')
        x += 175 if kind=='initial' else 185
    out.append('</svg>')
    return ''.join(out)

initial_svg=make_svg('initial')
final_svg=make_svg('final')
(OUT/'network_initial.svg').write_text(initial_svg, encoding='utf-8')
(OUT/'network_final.svg').write_text(final_svg, encoding='utf-8')
html=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Synaptic network structures</title><style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1f2937;background:#f8fafc}}h1{{font-size:28px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.card{{background:white;border-radius:12px;padding:10px;box-shadow:0 3px 18px #0001}}svg{{width:100%;height:auto}}@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><h1>突触网络结构变化</h1><p>同一正式实验种子 20261123：初始 128 条边，训练后保留 192 条活动边。</p><div class="grid"><div class="card">{initial_svg}</div><div class="card">{final_svg}</div></div></body></html>'''
(OUT/'network_structures.html').write_text(html, encoding='utf-8')

# ---------- Ball collision GIF ----------
scene = {"y":11.279060402186587,"v":-0.013634916997463331,"gravity":0.035,"x":20.338602244853973,"vx":0.023218952398747203,"background":0.0883274333178997,"brightness":0.6794010183541104,"texture":0.8147796944249421,"contact":31}
S=32; FLOOR=29; RADIUS=2
frames=[]
font_big=get_font(22, True); font_small=get_font(16, False); font_tiny=get_font(13, False)
for t in range(48):
    im=Image.new('RGB',(760,460),(18,24,38)); d=ImageDraw.Draw(im)
    # title and labels
    d.text((28,18),'Ball drop and impact',font=font_big,fill=(242,245,250))
    d.text((28,50),'same kinematics as the experiment scene',font=font_small,fill=(170,184,204))
    # scene viewport
    ox,oy,scale=72,92,10
    # soft backdrop and floor
    d.rounded_rectangle((ox-12,oy-12,ox+S*scale+12,oy+S*scale+12),radius=12,fill=(37,48,66),outline=(80,98,124),width=2)
    # pixel-like background texture
    for row in range(S):
        for col in range(S):
            val=max(0,min(255,int((scene['background']+0.025*math.sin(row*0.7+col*0.5+scene['texture']*6.28))*255)))
            if row>=FLOOR: val=min(255,val+30)
            c=(val,val+4 if val<250 else val,val+10 if val<245 else val)
            d.rectangle((ox+col*scale,oy+row*scale,ox+(col+1)*scale-1,oy+(row+1)*scale-1),fill=c)
    y=min(FLOOR-RADIUS, scene['y']+scene['v']*min(t,scene['contact'])+(scene['gravity']/2)*min(t,scene['contact'])**2)
    x=scene['x']+scene['vx']*min(t,scene['contact'])
    # draw ball with glow
    cx=ox+x*scale; cy=oy+y*scale; rr=RADIUS*scale
    for k in range(3,0,-1):
        alpha=int(32/k)
        col=(min(255,110+alpha*2),min(255,180+alpha*2),255)
        d.ellipse((cx-rr-k*4,cy-rr-k*4,cx+rr+k*4,cy+rr+k*4),outline=col,width=2)
    d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=(245,198,76),outline=(255,244,180),width=2)
    # collision flash and sound-wave cue
    if t==scene['contact']:
        d.line((cx-32,cy+rr+12,cx+32,cy+rr+12),fill=(255,170,70),width=4)
        d.arc((cx-42,cy-42,cx+42,cy+42),200,340,fill=(255,214,94),width=3)
        d.arc((cx-62,cy-62,cx+62,cy+62),205,335,fill=(255,214,94),width=3)
        d.text((430,150),'IMPACT',font=font_big,fill=(255,198,82))
        d.text((430,184),f'frame {t}',font=font_small,fill=(255,225,160))
    elif t>scene['contact']:
        d.text((430,150),'after impact',font=font_big,fill=(160,205,255))
        d.text((430,184),f'frame {t}',font=font_small,fill=(190,210,235))
    else:
        d.text((430,150),'falling',font=font_big,fill=(160,205,255))
        d.text((430,184),f'frame {t}',font=font_small,fill=(190,210,235))
    # timeline
    tx,ty=430,285; tw=280
    d.text((tx,ty-28),'collision timeline',font=font_small,fill=(180,195,214))
    d.line((tx,ty,tx+tw,ty),fill=(91,108,134),width=5)
    px=tx+tw*scene['contact']/47
    d.line((px,ty-16,px,ty+16),fill=(255,188,74),width=4)
    d.ellipse((tx+tw*t/47-5,ty-5,tx+tw*t/47+5,ty+5),fill=(111,198,255))
    d.text((tx,ty+20),'0',font=font_tiny,fill=(150,164,183)); d.text((px-14,ty+20),str(scene['contact']),font=font_tiny,fill=(255,210,130)); d.text((tx+tw-10,ty+20),'47',font=font_tiny,fill=(150,164,183))
    d.text((28,425),'Training signal: collision feedback is revealed at the impact frame.',font=font_small,fill=(180,195,214))
    frames.append(im)
frames[0].save(OUT/'ball_collision.gif',save_all=True,append_images=frames[1:],duration=80,loop=0,optimize=False)

meta={
 'seed':20261123,'method':'joint','initial_edges':len(INITIAL),'final_live_edges':len(LIVE),
 'new_live_edges':len(NEW_LIVE),'mature_final_edges':sum(e.get('state')=='mature' for e in LIVE),
 'contact_frame':31,'outputs':['network_initial.svg','network_final.svg','network_structures.html','ball_collision.gif']
}
(OUT/'visuals_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False))

