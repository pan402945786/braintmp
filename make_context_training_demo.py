from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json, math

OUT = Path(r'D:\work\braintmp')
W, H = 960, 540
BG = '#0f172a'
PANEL = '#172033'
PANEL2 = '#1d2940'
FG = '#e5e7eb'
MUTED = '#9ca3af'
BLUE = '#60a5fa'
CYAN = '#22d3ee'
GREEN = '#34d399'
ORANGE = '#f59e0b'
RED = '#fb7185'
PURPLE = '#a78bfa'
GRID = '#334155'
FONT = r'C:\Windows\Fonts\NotoSansSC-VF.ttf'
FONT_B = r'C:\Windows\Fonts\msyhbd.ttc'

def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except Exception:
        return ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', size)

F12 = font(12); F14 = font(14); F16 = font(16); F18 = font(18, True); F22 = font(22, True); F28 = font(28, True)

def txt(d, xy, s, f=F14, fill=FG, anchor=None):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)

def rr(d, box, radius=12, fill=PANEL, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def lerp(a,b,t): return a+(b-a)*t

def clamp(x,a=0,b=1): return max(a,min(b,x))

def soft_probs(target, confidence):
    p = [(1-confidence)/2]*3
    p[target] = confidence
    return p

# actual run summary distilled for the explanatory animation
GROWTH = [36,67,80,88,96,104,112,120]
CHECK_E = [1,20,40,60,90,120,150,180]
CHECK_ACC = [0.471,0.700,0.694,0.700,0.913,0.883,0.913,0.927]
CHECK_LOSS = [1.059,0.591,0.552,0.444,0.330,0.575,0.225,0.182]

def interp(xs, ys, x):
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(len(xs)-1):
        if xs[i] <= x <= xs[i+1]:
            t=(x-xs[i])/(xs[i+1]-xs[i]); return lerp(ys[i],ys[i+1],t)

def active_at_epoch(e):
    return 2 + sum(e >= x for x in GROWTH)

def training_frame(epoch, phase='train', event=False):
    im = Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
    txt(d,(30,22),'十个循环神经元：从看见线索到记住路线',F22)
    txt(d,(930,28),f'训练轮次 {epoch}/180' if phase=='train' else '演示：一段路线',F14,MUTED,anchor='ra')
    # panels
    rr(d,(24,65,310,350),fill=PANEL)
    rr(d,(326,65,658,350),fill=PANEL)
    rr(d,(674,65,936,350),fill=PANEL)
    rr(d,(24,365,936,516),fill=PANEL)
    txt(d,(44,82),'1. 输入：入口线索会消失',F16)
    txt(d,(346,82),'2. 循环网络：把过去留在状态里',F16)
    txt(d,(694,82),'3. 当前预测',F16)
    # corridor timeline
    x0,y0=52,150; step=23
    txt(d,(52,112),'路线 0：0 1 0 2 0 1 0 2 0 1 0 2',F12,MUTED)
    for i in range(12):
        x=x0+i*step
        fill='#26344b' if i%2==0 else '#202b3e'
        d.rounded_rectangle((x,y0,x+19,y0+54),radius=5,fill=fill)
        label=['无','左','无','右'][i%4]
        txt(d,(x+9.5,y0+27),label,F12,FG,anchor='mm')
    t = (epoch-1)%12 if phase=='episode' else 0
    if phase=='episode':
        # show landmark only at t=0; current position highlighted
        for i in range(12):
            x=x0+i*step
            if i==t: d.rounded_rectangle((x-2,y0-2,x+21,y0+56),radius=7,outline=CYAN,width=2)
        if t==0:
            d.ellipse((160,232,204,276),fill=PURPLE)
            txt(d,(182,254),'入口',F12,BG,anchor='mm')
            txt(d,(52,290),'这一拍看到了路线标记',F14,PURPLE)
        else:
            txt(d,(52,290),'标记已消失，只剩相同噪声',F14,MUTED)
        txt(d,(52,320),f'当前时刻 t = {t}',F14,CYAN)
    else:
        txt(d,(52,290),'输入线索已经消失',F14,MUTED)
        txt(d,(52,320),'网络必须依靠记忆预测',F14,CYAN)
    # network
    n=active_at_epoch(epoch) if phase=='train' else 2
    active = n
    cx0,cy0=370,150
    for i in range(10):
        row=i//5; col=i%5
        cx=cx0+col*54; cy=cy0+row*84
        if i<n:
            # state pulse
            pulse=0.25+0.75*((math.sin((epoch+i)*0.5)+1)/2)
            fill=tuple(int(lerp(35,96,pulse)) for _ in range(1))
            d.ellipse((cx-21,cy-21,cx+21,cy+21),fill='#245b65' if i<n else '#243044',outline=CYAN,width=2)
            txt(d,(cx,cy),f'N{i+1}',F12,FG,anchor='mm')
            if i==n-1 and event:
                d.ellipse((cx-28,cy-28,cx+28,cy+28),outline=ORANGE,width=4)
        else:
            d.ellipse((cx-21,cy-21,cx+21,cy+21),fill='#222d40',outline=GRID,width=1)
            txt(d,(cx,cy),'空',F12,MUTED,anchor='mm')
    txt(d,(346,312),f'当前活动神经元：{n}/10',F16,GREEN if n>=5 else ORANGE)
    if event:
        txt(d,(346,333),'验证误差停滞 → 增加 1 个神经元',F14,ORANGE)
    else:
        txt(d,(346,333),'蓝色圆圈 = 已存在；灰色 = 尚未长出',F12,MUTED)
    # prediction panel
    target = ((epoch-1)%3)+0 if phase=='train' else ([0,1,0,2,0,1,0,2,0,1,0,2][(epoch-1)%12])
    if phase=='train':
        acc=interp(CHECK_E,CHECK_ACC,epoch)
        conf=0.36+0.62*clamp((acc-0.45)/0.55)
    else:
        conf=0.94
    probs=soft_probs(target,conf)
    names=['无地标','左侧','右侧']; colors=[MUTED,BLUE,ORANGE]
    for i,(name,p) in enumerate(zip(names,probs)):
        y=145+i*51
        txt(d,(698,y+10),name,F14,FG)
        d.rounded_rectangle((760,y,895,y+24),radius=5,fill='#243044')
        d.rounded_rectangle((760,y,760+135*p,y+24),radius=5,fill=colors[i])
        txt(d,(905,y+12),f'{p*100:.0f}%',F14,FG,anchor='rm')
    if phase=='episode':
        txt(d,(694,315),f'目标：{names[target]}',F16,GREEN)
        txt(d,(694,337),'预测正确 ✓',F14,GREEN)
    else:
        txt(d,(694,315),f'延迟准确率：{interp(CHECK_E,CHECK_ACC,epoch)*100:.1f}%',F16,GREEN if acc>.8 else ORANGE)
        txt(d,(694,337),'训练中：逐渐学会记忆路线',F12,MUTED)
    # chart
    txt(d,(44,382),'训练曲线：损失下降，神经元在停滞时增加',F16)
    gx,gy,gw,gh=48,420,680,72
    for k in range(5):
        yy=gy+gh*k/4; d.line((gx,yy,gx+gw,yy),fill=GRID,width=1)
    d.line((gx,gy,gx,gy+gh),fill=GRID,width=1); d.line((gx,gy+gh,gx+gw,gy+gh),fill=GRID,width=1)
    pts=[]
    for xx in range(0,181,4):
        loss=interp(CHECK_E,CHECK_LOSS,xx if xx>0 else 1)
        px=gx+gw*xx/180; py=gy+gh*clamp(loss/1.1)
        pts.append((px,py))
    d.line(pts,fill=CYAN,width=3)
    for ge in GROWTH:
        px=gx+gw*ge/180
        d.line((px,gy,px,gy+gh),fill=ORANGE,width=2)
    txt(d,(744,407),'橙线 = 长出新神经元',F12,ORANGE)
    txt(d,(744,432),f'数量：{n}/10',F22,GREEN)
    txt(d,(744,466),'先记住线索，再预测未来',F14,FG)
    return im

frames=[]; durations=[]
# opening
for _ in range(10): frames.append(training_frame(1,'episode')); durations.append(180)
# episode timeline
for t in range(24):
    frames.append(training_frame(t+1,'episode')); durations.append(180)
# training, include event holds
for e in range(1,181,4):
    frames.append(training_frame(e,'train',event=e in GROWTH)); durations.append(150 if e not in GROWTH else 650)
# closing replay
for e in range(12):
    frames.append(training_frame(e+1,'episode')); durations.append(180)

out_gif=OUT/'context_training_demo.gif'
frames[0].save(out_gif,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True)

# Write an HTML viewer with play/pause and next controls.
meta=[]
for j in range(len(frames)):
    if j < 24:
        meta.append({'caption':'演示：看见入口线索','message':'入口颜色消失后，网络必须靠记忆','neurons':2,'growth':False,'acc':45.0})
    else:
        epoch=min(180, 1 + (j-24)*4)
        meta.append({'caption':'训练轮次 '+str(epoch),'message':'损失下降，验证集停滞时新增神经元','neurons':active_at_epoch(epoch),'growth':epoch in GROWTH,'acc':round(interp(CHECK_E,CHECK_ACC,epoch)*100,1)})

html = '''<!doctype html><meta charset="utf-8"><title>路线记忆与神经元增长演示</title>
<style>body{margin:0;background:#0f172a;color:#e5e7eb;font-family:"Microsoft YaHei",sans-serif}main{max-width:960px;margin:20px auto}canvas{width:100%;background:#0f172a;border-radius:12px}button{font-size:16px;padding:8px 16px;margin:8px 6px 0 0;border-radius:8px;border:1px solid #475569;background:#1e293b;color:#e5e7eb;cursor:pointer}p{color:#94a3b8}</style>
<main><canvas id="c" width="960" height="540"></canvas><div><button id="play">播放 / 暂停</button><button id="next">下一帧</button><span id="status"></span></div><p>先看入口标记，等标记消失后观察网络如何依靠循环状态继续预测；训练曲线中的橙线表示新增一个神经元。</p></main>
<script>
const frames = __DATA__;
const c=document.getElementById('c'),ctx=c.getContext('2d'),status=document.getElementById('status');
let i=0,playing=true,last=0;
function draw(){ctx.fillStyle='#0f172a';ctx.fillRect(0,0,c.width,c.height);ctx.font='24px Microsoft YaHei';ctx.fillStyle='#e5e7eb';ctx.fillText('十个循环神经元：从看见线索到记住路线',30,45);ctx.font='16px Microsoft YaHei';ctx.fillStyle='#94a3b8';ctx.fillText(frames[i].caption,30,75);ctx.fillText(`${i+1} / ${frames.length}`,850,45);ctx.font='18px Microsoft YaHei';ctx.fillStyle='#60a5fa';ctx.fillText(frames[i].message,30,110);ctx.font='16px Microsoft YaHei';ctx.fillStyle='#34d399';ctx.fillText(`神经元数量：${frames[i].neurons}/10`,30,145);ctx.fillStyle='#f59e0b';ctx.fillText(frames[i].growth?'验证误差停滞 → 增加 1 个神经元':'训练：权重逐步调整',30,175);ctx.fillStyle='#e5e7eb';ctx.fillText(`延迟预测准确率：${frames[i].acc}%`,30,205);let x=40; for(let n=0;n<10;n++){ctx.beginPath();ctx.arc(x+n*80,280,24,0,Math.PI*2);ctx.fillStyle=n<frames[i].neurons?'#24606c':'#222d40';ctx.fill();ctx.strokeStyle=n===frames[i].neurons-1&&frames[i].growth?'#f59e0b':'#22d3ee';ctx.stroke();ctx.fillStyle='#e5e7eb';ctx.fillText(n<frames[i].neurons?'N'+(n+1):'空',x-12+n*80,286)}ctx.fillStyle='#94a3b8';ctx.fillText('输入线索 → 记忆状态 → 下一事件预测',280,380);ctx.fillStyle='#22d3ee';ctx.fillRect(280,405,frames[i].acc*5,18);ctx.fillStyle='#e5e7eb';ctx.fillText('预测能力',280,450);status.textContent=frames[i].growth?' 现在正在长出新神经元':' 观察训练过程';}
function tick(t){if(t-last>150){if(playing){i=(i+1)%frames.length;draw()}last=t}requestAnimationFrame(tick)}
document.getElementById('play').onclick=()=>playing=!playing;document.getElementById('next').onclick=()=>{i=(i+1)%frames.length;draw()};draw();requestAnimationFrame(tick);
</script>'''.replace('__DATA__', json.dumps(meta, ensure_ascii=False))
(OUT/'context_training_demo.html').write_text(html,encoding='utf-8')
print(out_gif, len(frames), out_gif.stat().st_size)
