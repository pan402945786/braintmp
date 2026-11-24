import json, math, pathlib
from PIL import Image, ImageDraw, ImageFont
BASE=pathlib.Path(r'D:\work\braintmp'); OUT=BASE/'visuals_2026-11-24'
D=json.loads((BASE/'runs/2026-11-23-formal-a/seed-20261123/joint-edges.json').read_text()); E=D['edges']; initial=[e for e in E if e.get('born',0)==0 and e.get('state')=='initial']; live=[e for e in E if e.get('deleted') is None]
W,H=760,720; CX,CY,R=380,345,245; pos={n:(CX+R*math.cos(-math.pi/2+2*math.pi*n/32),CY+R*math.sin(-math.pi/2+2*math.pi*n/32)) for n in range(32)}
def font(sz,b=False):
 p=r'C:\Windows\Fonts\segoeuib.ttf' if b else r'C:\Windows\Fonts\segoeui.ttf'; return ImageFont.truetype(p,sz) if pathlib.Path(p).exists() else ImageFont.load_default()
def arrow(draw,a,b,color,width):
 x1,y1=a; x2,y2=b; ang=math.atan2(y2-y1,x2-x1); L=7+width*2; aw=3+width
 pts=[(x2,y2),(x2-L*math.cos(ang)+aw*math.sin(ang),y2-L*math.sin(ang)-aw*math.cos(ang)),(x2-L*math.cos(ang)-aw*math.sin(ang),y2-L*math.sin(ang)+aw*math.cos(ang))]; draw.polygon(pts,fill=color)
def draw_net(kind, outpath):
 im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im); d.text((32,34),'Initial network' if kind=='initial' else 'Learned network',font=font(26,True),fill=(31,41,55)); d.text((32,63),('32 units · 128 directed connections · 64 backbone + 64 extra' if kind=='initial' else '32 units · 192 live connections · 64 retained additions'),font=font(15),fill=(100,116,139)); d.ellipse((CX-R,CY-R,CX+R,CY+R),outline=(226,232,240),width=1)
 edges=initial if kind=='initial' else live
 for e in edges:
  x1,y1=pos[e['i']]; x2,y2=pos[e['j']];
  if e['i']==e['j']: continue
  dx,dy=x2-x1,y2-y1; dist=math.hypot(dx,dy) or 1; nx,ny=-dy/dist,dx/dist; mx,my=(x1+x2)/2,(y1+y2)/2; bend=(0.10+0.18*min(1.0,dist/(2*R)))*R; sign=1 if e.get('id',0)%2==0 else -1; qx,qy=mx+sign*nx*bend,my+sign*ny*bend
  pts=[]
  for k in range(21):
   t=k/20; xx=(1-t)**2*x1+2*(1-t)*t*qx+t*t*x2; yy=(1-t)**2*y1+2*(1-t)*t*qy+t*t*y2; pts.append((xx,yy))
  w=0.7+min(2.7,abs(float(e.get('w',0)))/1.7*2.2)
  if kind=='initial': col=(79,139,214) if e.get('backbone') else (167,176,188); alpha=160 if e.get('backbone') else 85
  else:
   if e.get('born',0)==0: col=(148,163,184); alpha=70
   elif e.get('state')=='mature': col=(229,138,58) if float(e.get('w',0))>=0 else (155,107,211); alpha=210
   else: col=(100,116,139); alpha=150
  overlay=Image.new('RGBA',im.size,(0,0,0,0)); od=ImageDraw.Draw(overlay); od.line(pts,fill=(*col,alpha),width=max(1,int(round(w)))); im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB'); d=ImageDraw.Draw(im); arrow(d,pts[-2],pts[-1],col,max(1,int(round(w))))
 for n,(x,y) in pos.items():
  d.ellipse((x-16,y-16,x+16,y+16),fill=(248,250,252),outline=(51,65,85),width=2); d.text((x,y),str(n),font=font(12),anchor='mm',fill=(31,41,55))
 ly=H-82; items=[((79,139,214),'backbone ring'),((167,176,188),'extra initial edge')] if kind=='initial' else [((148,163,184),'surviving initial edge'),((229,138,58),'mature retained edge'),((155,107,211),'mature negative-weight edge'),((100,116,139),'trial edge')]; x=32
 for c,label in items: d.line((x,ly,x+30,ly),fill=c,width=3); d.text((x+40,ly-8),label,font=font(13),fill=(31,41,55)); x += 175 if kind=='initial' else 185
 im.save(outpath)
draw_net('initial',OUT/'network_initial.png'); draw_net('final',OUT/'network_final.png'); print('done')

