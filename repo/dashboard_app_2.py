"""
Data Vortex A'26 — Social Engine Dashboard (v5 · Aurora Glass)
Run:  streamlit run dashboard_app.py
Requires: streamlit>=1.36, plotly>=5.18, scikit-learn, pandas, numpy

What's new vs v4
  * Bento layout modelled on the reference: glass top bar, KPI cards with animated
    count-up + drawing sparklines, big chart tiles, 3D Plotly surfaces (with orbit button)
  * Real JS layer (injected into the parent page): animated social-graph canvas that reacts
    to the mouse, 3D card tilt + spotlight glare, count-up numbers
  * Social touches: floating reactions in the hero, live ticker, LIVE pulse, social-style post feed
  * Dark / Light theme toggle (like the 3 variants in the reference)
"""
import os
import json
import html as _html
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# DATA CONFIG  (extension-agnostic: .xlsx -> .csv -> .json)
# ---------------------------------------------------------------------------
FILES = {
    "r1_cleaned": "datasets/Social_Engine_Posts_Cleaned",
    "r1_corrupted": "datasets/Social_Engine_Posts_Corrupted",
    "r1_users": "datasets/Social_Engine_Users_Cleaned",
    "r2_labeled": "datasets/Labeled_Social_NLP_Training_Data",
    "r2_sent_preds": "outputs/round2_sentiment_predictions.csv",
    "r2_topic_preds": "outputs/round2_topic_predictions.csv",
    "r3_live": "datasets/live_dataset",
    "r3_translated": "datasets/translated_checkpoint",
    "r3_processed": "datasets/round3_processed",
    "r4_live_processed": "datasets/live_processing",
}
ENTITIES = ["vijay", "trisha", "udhayanidhi", "kangana", "tvk", "dmk", "sangeetha"]


def read_any(stem):
    for ext, reader in [(".xlsx", pd.read_excel), (".csv", pd.read_csv), (".json", pd.read_json)]:
        path = stem + ext
        if os.path.exists(path):
            try:
                return reader(path)
            except Exception as e:
                st.warning(f"Found `{path}` but couldn't read it: {e}")
                return None
    return None


st.set_page_config(page_title="Social Engine — Data Vortex A'26", page_icon="🌀",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------
LIGHT = st.session_state.get("light", False)
T = (dict(bgfx="linear-gradient(160deg,#eef6fd 0%,#d8e8f7 38%,#bfd7f0 72%,#a9c9ea 100%)", bg1="#d8e8f7",
          card="rgba(255,255,255,.56)", edge="rgba(255,255,255,.88)", shadow="rgba(58,110,170,.24)",
          text="#17345c", sub="#48668c", muted="#7f98b8", glow="rgba(40,185,210,.42)", track="rgba(50,100,160,.14)",
          tip="#f4f9ff", pill="rgba(255,255,255,.66)", side="linear-gradient(180deg,rgba(226,240,252,.92),rgba(188,214,240,.92))",
          orb1="rgba(80,205,228,.55)", orb2="rgba(90,140,225,.45)", ln1="#35c3d3", ln2="#4a86d0",
          g1="#25aec6", g2="#3f7bc8", g3="#7fb6e8", onacc="#ffffff",
          nodes="47,188,208|74,134,208|127,182,232|123,140,240")
     if LIGHT else
     dict(bgfx="linear-gradient(160deg,#020817,#0a2242 55%,#020817)", bg1="#020817",
          card="rgba(13,34,68,.68)", edge="rgba(125,211,252,.20)", shadow="rgba(0,0,0,.55)",
          text="#eaf6ff", sub="#8fb0d0", muted="#5b7a99", glow="rgba(34,211,238,.40)", track="rgba(255,255,255,.08)",
          tip="#0f2545", pill="rgba(255,255,255,.05)", side="linear-gradient(180deg,#071226,#050b1a)",
          orb1="rgba(34,211,238,.30)", orb2="rgba(167,139,250,.28)", ln1="#22d3ee", ln2="#f472b6", g1="#22d3ee", g2="#a78bfa", g3="#f472b6", onacc="#04212e",
          nodes="34,211,238|167,139,250|244,114,182|251,191,36"))
CYAN, BLUE, VIOLET, PINK, AMBER, GREEN, CORAL, SLATE = (
    ("#22b5cf", "#4a86d0", "#7b8cf0", "#f08fb3", "#f4b860", "#3fc59a", "#f0768a", "#8aa0bd") if LIGHT else
    ("#22d3ee", "#3b82f6", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#fb7185", "#94a3b8"))
ACCENT = [CYAN, VIOLET, PINK, AMBER, GREEN, BLUE, CORAL]
SENT = {"Positive": GREEN, "Neutral": SLATE, "Negative": CORAL}
SENT_EMOJI = {"Positive": "❤️", "Neutral": "💬", "Negative": "😡"}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif}
html,body{background:var(--bg1)!important}
.stApp{background:var(--bgfx) fixed!important}
.stMain,[data-testid="stMain"],[data-testid="stMainBlockContainer"],.main,.block-container,[data-testid="stBottom"]{background:transparent!important}
[data-testid="stAppViewContainer"]{position:relative;z-index:1;background:transparent!important}
#MainMenu,footer,[data-testid="stDecoration"],[data-testid="stToolbar"]{display:none!important}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1480px;padding:1.1rem 2rem 4rem}
body::before,body::after{content:'';position:fixed;border-radius:50%;filter:blur(50px);pointer-events:none;z-index:0}
body::before{width:640px;height:640px;left:-180px;top:-220px;background:radial-gradient(circle,var(--orb1),transparent 65%);animation:drift 20s ease-in-out infinite alternate}
body::after{width:700px;height:700px;right:-220px;bottom:-260px;background:radial-gradient(circle,var(--orb2),transparent 70%);animation:drift 24s ease-in-out infinite alternate-reverse}
@keyframes drift{to{transform:translate(90px,60px) scale(1.15)}}
@keyframes rise{from{opacity:0;transform:translateY(18px)}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes grow{from{width:0!important}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{50%{opacity:.25}}
@keyframes shimmer{to{background-position:200% 0}}
@keyframes floatUp{0%{transform:translateY(0) scale(.6);opacity:0}15%{opacity:.95}100%{transform:translate(24px,-280px) scale(1.15);opacity:0}}
@keyframes marquee{to{transform:translateX(-50%)}}

/* ---------- glass ---------- */
.glass,div[data-testid="stVerticalBlockBorderWrapper"]{position:relative;background:linear-gradient(145deg,rgba(255,255,255,.07),rgba(255,255,255,.01)),var(--card);
 border:1px solid var(--edge);border-radius:24px;backdrop-filter:blur(24px) saturate(160%);-webkit-backdrop-filter:blur(24px) saturate(160%);
 box-shadow:0 22px 50px var(--shadow),inset 0 1px 0 rgba(255,255,255,.14),inset 0 0 44px color-mix(in srgb,var(--g1) 5%,transparent);
 transition:transform .25s ease,box-shadow .3s,border-color .3s;animation:rise .8s cubic-bezier(.2,.8,.2,1) backwards;will-change:transform}
.glass::after,div[data-testid="stVerticalBlockBorderWrapper"]::after{content:'';position:absolute;inset:0;border-radius:inherit;pointer-events:none;
 background:radial-gradient(420px circle at var(--mx,50%) var(--my,0%),rgba(255,255,255,.13),transparent 45%);opacity:0;transition:opacity .3s}
.glass:hover::after,div[data-testid="stVerticalBlockBorderWrapper"]:hover::after{opacity:1}
.glass:hover,div[data-testid="stVerticalBlockBorderWrapper"]:hover{border-color:color-mix(in srgb,var(--g1) 55%,transparent);box-shadow:0 28px 60px var(--shadow),0 0 32px var(--glow),inset 0 1px 0 rgba(255,255,255,.16)}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"]{box-shadow:none;background:none;backdrop-filter:none}

/* ---------- top bar / nav ---------- */
.brand{display:flex;align-items:center;gap:12px;font-weight:800;font-size:1.12rem;color:var(--text)}
.logo{width:42px;height:42px;border-radius:13px;display:grid;place-items:center;font-size:1.3rem;background:linear-gradient(135deg,var(--g1),var(--g2) 60%,var(--g3));box-shadow:0 0 26px color-mix(in srgb,var(--g1) 55%,transparent)}
.logo span{display:inline-block;animation:spin 9s linear infinite}
div[role="radiogroup"]{gap:8px;justify-content:center}
div[role="radiogroup"] label div:not(:has(p)),div[role="radiogroup"] label>span:first-child{display:none!important}
div[role="radiogroup"] label{background:var(--pill);border:1px solid var(--edge);border-radius:999px;padding:8px 20px!important;transition:all .2s}
div[role="radiogroup"] label:hover{border-color:color-mix(in srgb,var(--g1) 60%,transparent);box-shadow:0 0 16px var(--glow)}
div[role="radiogroup"] label p{color:var(--sub)!important;font-weight:600!important;font-size:.85rem!important}
div[role="radiogroup"] label:has(input:checked){background:linear-gradient(135deg,color-mix(in srgb,var(--g1) 32%,transparent),color-mix(in srgb,var(--g2) 26%,transparent));border-color:color-mix(in srgb,var(--g1) 85%,transparent);box-shadow:0 0 22px var(--glow)}
div[role="radiogroup"] label:has(input:checked) p{color:var(--text)!important;font-weight:800!important}
[data-testid="stToggle"] label,[data-testid="stToggle"] label *{color:var(--text)!important;font-size:.82rem;font-weight:600}

/* ---------- hero ---------- */
.hero{overflow:hidden;padding:30px 36px 26px;margin:16px 0 20px}
.hero-row{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;position:relative;z-index:2}
.eyebrow{color:var(--sub);font-size:.78rem;font-weight:600;margin-bottom:6px}
.title{font-size:2.7rem;font-weight:800;letter-spacing:-.02em;line-height:1.1;background:linear-gradient(100deg,var(--text),var(--g1),var(--g2),var(--g3),var(--text));background-size:200% 100%;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shimmer 8s linear infinite}
.narr{color:var(--sub);font-size:.92rem;max-width:720px;line-height:1.6;margin-top:10px}
.avatar{display:flex;align-items:center;gap:10px;background:var(--pill);border:1px solid var(--edge);border-radius:999px;padding:6px 18px 6px 6px;height:fit-content}
.avatar i{width:38px;height:38px;border-radius:50%;display:grid;place-items:center;font-style:normal;font-weight:800;color:var(--onacc);background:linear-gradient(135deg,var(--g1),var(--g3))}
.avatar b{display:block;font-size:.85rem;color:var(--text)}.avatar small{color:var(--sub);font-size:.72rem}
.steps{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px;position:relative;z-index:2}
.step{padding:8px 16px;border-radius:999px;border:1px solid var(--edge);font-size:.76rem;color:var(--sub);background:var(--pill)}
.step.on{color:var(--onacc);font-weight:800;background:linear-gradient(90deg,var(--g1),var(--g2));border-color:transparent;box-shadow:0 0 22px var(--glow)}
.react{position:absolute;bottom:-30px;font-size:1.5rem;opacity:0;animation:floatUp 7s linear infinite;z-index:1}

/* ---------- KPI ---------- */
.kpi{padding:18px 20px 0;min-height:150px;overflow:hidden;margin-bottom:16px}
.kpi-top{display:flex;justify-content:space-between;align-items:center}
.kpi-label{font-size:.78rem;color:var(--sub);font-weight:600}
.kpi-ic{width:36px;height:36px;border-radius:12px;display:grid;place-items:center;font-size:1.05rem;background:linear-gradient(135deg,var(--a),var(--b));box-shadow:0 8px 20px var(--glow)}
.kpi-val{font-size:2.05rem;font-weight:800;color:var(--text);letter-spacing:-.02em;margin-top:8px;line-height:1.15}
.kpi-delta{display:inline-block;margin-top:4px;font-size:.72rem;font-weight:700;padding:2px 9px;border-radius:999px}
.up{color:#34d399;background:rgba(52,211,153,.15)}.down{color:#fb7185;background:rgba(251,113,133,.15)}.flat{color:var(--sub);background:var(--track)}
.spark{position:absolute;left:0;bottom:0;width:100%;height:48px}
.spark .line{stroke-dasharray:400;stroke-dashoffset:400;animation:draw 2s ease .2s forwards}

/* ---------- section head / pills / ticker ---------- */
.sec{display:flex;align-items:center;gap:12px;margin:6px 0 16px}
.sec-t{font-size:1.4rem;font-weight:800;color:var(--text);letter-spacing:-.01em}
.sec small{color:var(--sub);font-size:.85rem}
.live{display:inline-flex;align-items:center;gap:7px;padding:4px 12px;border-radius:999px;font-size:.72rem;font-weight:800;color:#ff9c9c;background:rgba(255,107,107,.14);border:1px solid rgba(255,107,107,.4)}
.live i{width:7px;height:7px;border-radius:50%;background:#ff6b6b;box-shadow:0 0 10px #ff6b6b;animation:pulse 1.4s infinite}
.ticker{overflow:hidden;white-space:nowrap;padding:12px 0;margin-bottom:16px;-webkit-mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent);mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent)}
.ticker-in{display:inline-block;animation:marquee 34s linear infinite}
.tk{display:inline-flex;gap:8px;align-items:center;margin-right:34px;font-size:.84rem;color:var(--sub)}.tk b{color:var(--text)}.tk em{font-style:normal;color:var(--a,var(--g1));font-weight:700}

/* ---------- feed / progress / callout ---------- */
.ptitle{font-size:.85rem;font-weight:700;color:var(--sub);margin:4px 8px 14px}
.post{display:flex;gap:12px;padding:12px 14px;border-radius:16px;background:var(--pill);border:1px solid var(--edge);margin:0 6px 10px;animation:rise .6s backwards}
.av{width:38px;height:38px;border-radius:50%;flex:none;display:grid;place-items:center;font-weight:800;color:var(--onacc);background:linear-gradient(135deg,var(--a),var(--b))}
.pmeta{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:.72rem;margin-bottom:3px}.pmeta b{color:var(--text)}
.ptext{color:var(--text);font-size:.86rem;line-height:1.45}
.chip{padding:1px 9px;border-radius:999px;font-size:.66rem;font-weight:700}
.prow{margin:0 8px 16px}.prow-top{display:flex;justify-content:space-between;font-size:.83rem;color:var(--sub);margin-bottom:6px}.prow-top b{color:var(--text)}
.ptrack{height:8px;border-radius:99px;background:var(--track);overflow:hidden}
.pfill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--g1),var(--g2),var(--g3));box-shadow:0 0 12px var(--glow);animation:grow 1.4s cubic-bezier(.2,.8,.2,1) backwards}
.callout{border-left:3px solid var(--g1);border-radius:14px;padding:14px 20px;margin-top:8px;color:var(--sub);font-size:.92rem;line-height:1.6;background:linear-gradient(135deg,color-mix(in srgb,var(--g1) 10%,transparent),color-mix(in srgb,var(--g2) 6%,transparent))}
.callout b{color:var(--text)}
.stButton>button{background:linear-gradient(135deg,color-mix(in srgb,var(--g1) 25%,transparent),color-mix(in srgb,var(--g2) 22%,transparent));border:1px solid color-mix(in srgb,var(--g1) 55%,transparent);color:var(--text);border-radius:999px;font-weight:700;padding:.45rem 1.3rem;transition:all .2s}
.stButton>button:hover{box-shadow:0 0 22px var(--glow);border-color:var(--g1);color:var(--text)}
div[data-testid="stAlert"]{background:var(--card);border:1px solid var(--edge);border-radius:14px}
[data-testid="stCaptionContainer"],.stCaption{color:var(--sub)!important}
hr{border-color:var(--edge)!important}
.up{color:color-mix(in srgb,#10b981 78%,var(--text))}.down{color:color-mix(in srgb,#f43f5e 78%,var(--text))}
.live{color:color-mix(in srgb,#ff5a5a 75%,var(--text))}
/* neon hairline on every card + tinted KPI + colourful hero */
.glass::before,div[data-testid="stVerticalBlockBorderWrapper"]::before{content:'';position:absolute;top:0;left:14%;right:14%;height:2px;border-radius:2px;pointer-events:none;background:linear-gradient(90deg,transparent,var(--ln1),var(--ln2),transparent);opacity:.85}
.glass.kpi{background:linear-gradient(150deg,color-mix(in srgb,var(--a) 20%,transparent),transparent 62%),var(--card)}
.glass.kpi::before{left:0;right:0;height:3px;border-radius:0;background:linear-gradient(90deg,var(--a),var(--b))}
.glass.hero{background:radial-gradient(620px 260px at 92% 0%,color-mix(in srgb,var(--g3) 26%,transparent),transparent 62%),radial-gradient(640px 280px at 0% 100%,color-mix(in srgb,var(--g1) 24%,transparent),transparent 62%),radial-gradient(500px 240px at 55% 50%,color-mix(in srgb,var(--g2) 14%,transparent),transparent 70%),var(--card)}
.avs{display:flex}.avs i{width:38px;height:38px;border-radius:50%;display:grid;place-items:center;font-style:normal;font-weight:800;color:#fff;border:2px solid var(--card);background:linear-gradient(135deg,var(--g1),var(--g2))}
.avs i+i{margin-left:-12px;background:linear-gradient(135deg,var(--g2),var(--g3))}
/* sidebar */
section[data-testid="stSidebar"]{background:var(--side)!important;border-right:1px solid var(--edge);backdrop-filter:blur(20px)}
.sb-h{font-size:.72rem;font-weight:800;color:var(--muted);margin:22px 4px 10px;letter-spacing:.08em}
.pres{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:16px;background:var(--pill);border:1px solid var(--edge);margin-bottom:8px}
.pres i{width:38px;height:38px;border-radius:50%;display:grid;place-items:center;font-style:normal;font-weight:800;color:#fff;flex:none;background:linear-gradient(135deg,var(--a),var(--b));box-shadow:0 6px 16px var(--glow)}
.pres b{color:var(--text);font-size:.88rem;display:block}.pres small{color:var(--muted);font-size:.72rem}
.rd{display:flex;gap:10px;align-items:center;padding:9px 12px;border-radius:12px;font-size:.82rem;color:var(--sub);margin-bottom:6px}
.rd.on{background:linear-gradient(90deg,color-mix(in srgb,var(--g1) 28%,transparent),color-mix(in srgb,var(--g2) 26%,transparent));color:var(--text);font-weight:800;box-shadow:0 0 16px var(--glow)}
.ds{display:flex;align-items:center;justify-content:space-between;font-size:.76rem;color:var(--sub);padding:5px 6px}
.ds i{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:9px}
.sb-f{margin-top:22px;padding:12px;border-radius:14px;font-size:.72rem;color:var(--muted);text-align:center;border:1px dashed var(--edge)}

[data-testid="stToggle"] label:has(input:checked)>div:first-child{background:var(--g1)!important}
[data-testid="stToggle"] p,[data-testid="stWidgetLabel"] p,label[data-baseweb="checkbox"] *{color:var(--text)!important;font-weight:600}
"""
LCSS = """
.kpi-ic{background:linear-gradient(145deg,#ffffff,#fff3ea)!important;border:1.5px solid color-mix(in srgb,var(--a) 55%,#fff);box-shadow:0 8px 18px color-mix(in srgb,var(--a) 38%,transparent)!important}
.logo{background:linear-gradient(145deg,#ffffff,#fff0e6)!important;border:1.5px solid var(--g1);box-shadow:0 8px 20px var(--glow)!important}
.glass.kpi{background:linear-gradient(150deg,color-mix(in srgb,var(--a) 24%,#fff),rgba(255,255,255,.55) 72%),var(--card)}
.stButton>button{background:linear-gradient(135deg,var(--g1),var(--g2));color:#fff;border-color:transparent;box-shadow:0 8px 20px var(--glow)}
.glass,div[data-testid="stVerticalBlockBorderWrapper"]{backdrop-filter:blur(26px) saturate(170%);-webkit-backdrop-filter:blur(26px) saturate(170%);box-shadow:0 18px 44px var(--shadow),inset 0 1px 0 rgba(255,255,255,.95),inset 0 0 40px rgba(255,255,255,.28)}
.glass.hero{background:radial-gradient(620px 260px at 92% 0%,rgba(90,215,235,.32),transparent 62%),radial-gradient(640px 280px at 0% 100%,rgba(110,160,235,.28),transparent 62%),var(--card)}
.stButton>button:hover{color:#fff;border-color:transparent;filter:brightness(1.05)}
.step.on,.rd.on{box-shadow:0 6px 18px var(--glow)}
.title{filter:saturate(1.15)}
""" if LIGHT else ""
TOG = """
[data-testid="stToggle"] label:has(input:checked)>div:not(:has(p)),[data-testid="stCheckbox"] label:has(input:checked)>div:not(:has(p)){background:var(--g1)!important}
"""
st.markdown(f"<style>:root{{color-scheme:{'light' if LIGHT else 'dark'};{''.join(f'--{k}:{v};' for k, v in T.items())}}}{CSS}{LCSS}{TOG}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# JS LAYER — runs in the parent page (injected once): social-graph canvas,
# 3D tilt + glare on glass cards, count-up numbers.
# ---------------------------------------------------------------------------
JS = r"""
(function(){
if(window.__vx)return;window.__vx=true;
var cv=document.createElement('canvas');cv.style.cssText='position:fixed;inset:0;width:100vw;height:100vh;z-index:0;pointer-events:none;opacity:.65';
document.body.appendChild(cv);var x=cv.getContext('2d'),W,H,N=[],M={x:-999,y:-999};
var cols=['34,211,238'];function C(a){return cols[a.k%cols.length]}
function rs(){W=cv.width=innerWidth;H=cv.height=innerHeight}rs();addEventListener('resize',rs);
for(var i=0;i<64;i++)N.push({x:Math.random()*innerWidth,y:Math.random()*innerHeight,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,r:Math.random()*2+1,k:i});
addEventListener('mousemove',function(e){M.x=e.clientX;M.y=e.clientY});
(function draw(){x.clearRect(0,0,W,H);cols=(getComputedStyle(document.documentElement).getPropertyValue('--nodes')||'34,211,238').trim().split('|');
 for(var i=0;i<N.length;i++){var a=N[i];a.x+=a.vx;a.y+=a.vy;if(a.x<0||a.x>W)a.vx*=-1;if(a.y<0||a.y>H)a.vy*=-1;
  var dm=Math.hypot(a.x-M.x,a.y-M.y);if(dm<150){a.x+=(a.x-M.x)/dm*.9;a.y+=(a.y-M.y)/dm*.9}
  x.beginPath();x.arc(a.x,a.y,a.r,0,7);x.fillStyle='rgba('+C(a)+',.8)';x.fill();
  for(var j=i+1;j<N.length;j++){var b=N[j],d=Math.hypot(a.x-b.x,a.y-b.y);
   if(d<135){x.strokeStyle='rgba('+C(a)+','+(.2*(1-d/135))+')';x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke()}}}
 requestAnimationFrame(draw)})();
var SEL='.glass,div[data-testid="stVerticalBlockBorderWrapper"]',last=null;
function rst(e){if(e)e.style.transform=''}
document.addEventListener('mousemove',function(ev){
 var el=ev.target.closest?ev.target.closest(SEL):null;
 if(last&&last!==el)rst(last);last=el;if(!el)return;
 var r=el.getBoundingClientRect(),px=(ev.clientX-r.left)/r.width,py=(ev.clientY-r.top)/r.height;
 el.style.setProperty('--mx',px*100+'%');el.style.setProperty('--my',py*100+'%');
 el.style.transform='perspective(1100px) rotateX('+((.5-py)*4)+'deg) rotateY('+((px-.5)*5)+'deg)'});
function cu(el){if(el.dataset.done)return;el.dataset.done=1;
 var t=parseFloat(el.dataset.count),dc=+el.dataset.dec||0,sf=el.dataset.suffix||'',s=performance.now();
 (function f(n){var k=Math.min((n-s)/1500,1),e=1-Math.pow(1-k,3);
  el.textContent=(t*e).toLocaleString(undefined,{minimumFractionDigits:dc,maximumFractionDigits:dc})+sf;
  if(k<1)requestAnimationFrame(f)})(s)}
function scan(){document.querySelectorAll('[data-count]').forEach(cu)}
scan();new MutationObserver(scan).observe(document.body,{childList:true,subtree:true});
})();
"""
components.html(
    "<script>var p=window.parent;if(!p.__vx){var s=p.document.createElement('script');"
    f"s.textContent={json.dumps(JS)};p.document.head.appendChild(s);}}</script>", height=0)


# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------
def esc(s, n=200):
    s = _html.escape(str(s))
    return s if len(s) <= n else s[:n] + "…"


def _lerp(c1, c2, t):
    a, b = c1.lstrip("#"), c2.lstrip("#")
    return "#" + "".join(f"{round(int(a[i:i+2], 16) + (int(b[i:i+2], 16) - int(a[i:i+2], 16)) * t):02x}" for i in (0, 2, 4))


def ramp(n, c1=CYAN, c2=VIOLET):
    return [c1] if n <= 1 else [_lerp(c1, c2, i / (n - 1)) for i in range(n)]


def spark(vals, color=CYAN):
    v = [float(x) for x in vals if pd.notna(x)]
    if len(v) < 2:
        return ""
    lo, rng = min(v), (max(v) - min(v)) or 1
    d = "M" + " L".join(f"{i / (len(v) - 1) * 200:.1f},{40 - (x - lo) / rng * 32:.1f}" for i, x in enumerate(v))
    g = f"g{abs(hash(d)) % 99999}"
    return (f'<svg class="spark" viewBox="0 0 200 46" preserveAspectRatio="none"><defs><linearGradient id="{g}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".38"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="{d} L200,46 L0,46Z" fill="url(#{g})"/><path class="line" d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="2.2" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>')


def kpi(col, label, value, icon, ac=(CYAN, BLUE), delta=None, tone="up", series=None, suffix="", dec=0):
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        v = f'<span data-count="{float(value)}" data-dec="{dec}" data-suffix="{suffix}">{value:,.{dec}f}{suffix}</span>'
    else:
        v = esc(value)
    d = f'<div><span class="kpi-delta {tone}">{esc(delta)}</span></div>' if delta else ""
    sp = spark(series, ac[0]) if series is not None else ""
    col.markdown(f'<div class="glass kpi" style="--a:{ac[0]};--b:{ac[1]}"><div class="kpi-top"><span class="kpi-label">{esc(label)}</span>'
                 f'<span class="kpi-ic">{icon}</span></div><div class="kpi-val">{v}</div>{d}{sp}</div>', unsafe_allow_html=True)


def kpi_row(items):
    for c, it in zip(st.columns(len(items)), items):
        kpi(c, **it)


def head(title, sub="", live=False):
    lv = '<span class="live"><i></i>LIVE</span>' if live else ""
    st.markdown(f'<div class="sec"><span class="sec-t">{title}</span>{lv}<small>{sub}</small></div>', unsafe_allow_html=True)


def callout(t):
    st.markdown(f'<div class="callout">{t}</div>', unsafe_allow_html=True)


def ticker(items):
    row = "".join(f'<span class="tk" style="--a:{ACCENT[i % 7]}"><b>{esc(k)}</b><em>{v}</em></span>' for i, (k, v) in enumerate(items))
    st.markdown(f'<div class="glass ticker"><div class="ticker-in">{row}{row}</div></div>', unsafe_allow_html=True)


def style_fig(fig, h=340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h, colorway=ACCENT,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=T["sub"], size=12),
        title=dict(font=dict(size=15, color=T["text"]), x=0.03, y=0.96),
        margin=dict(l=14, r=14, t=54, b=12),
        hoverlabel=dict(bgcolor=T["tip"], font_color=T["text"], bordercolor=CYAN),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.1))
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=T["track"], zeroline=False, showline=False)
    return fig


def card(fig, h=340):
    style_fig(fig, h)
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def wave(df, x, cols, title, cmap=None, h=340):
    fig = go.Figure()
    for i, c in enumerate(cols):
        if c not in df.columns:
            continue
        col = (cmap or {}).get(c, ACCENT[i % 7])
        fig.add_trace(go.Scatter(x=df[x], y=df[c], name=c, mode="lines", line=dict(width=2.6, color=col, shape="spline"),
                                 fill="tozeroy", fillcolor=col + "24"))
    fig.update_layout(title=title, hovermode="x unified")
    card(fig, h)


def donut(df, names, title, h=330, cmap=None, values=None):
    fig = px.pie(df, names=names, values=values, hole=0.72, title=title, color_discrete_sequence=ACCENT, color_discrete_map=cmap)
    fig.update_traces(textinfo="percent", textfont=dict(color="#0d2b4d" if LIGHT else "#04212e", size=11), pull=0.02,
                      marker=dict(line=dict(color="rgba(0,0,0,0)", width=3)))
    total = df[values].sum() if values else df[names].dropna().shape[0]
    fig.add_annotation(text=f"<b>{total:,}</b>", x=0.5, y=0.54, showarrow=False, font=dict(size=24, color=T["text"]))
    fig.add_annotation(text="Total", x=0.5, y=0.44, showarrow=False, font=dict(size=11, color=T["muted"]))
    card(fig, h)


def gauge(value, title, h=270):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title, "font": {"size": 14, "color": T["sub"]}},
        number={"suffix": "%", "font": {"size": 34, "color": T["text"]}},
        gauge={"axis": {"range": [0, 100], "tickcolor": T["muted"], "tickfont": {"color": T["muted"], "size": 10}},
               "bar": {"color": CYAN, "thickness": 0.3}, "bgcolor": T["track"], "borderwidth": 0,
               "steps": [{"range": [0, 50], "color": "rgba(251,113,133,.10)"}, {"range": [50, 80], "color": "rgba(251,191,36,.10)"},
                         {"range": [80, 100], "color": "rgba(52,211,153,.12)"}]}))
    card(fig, h)


def radar(cats, vals, title, h=330):
    fig = go.Figure(go.Scatterpolar(r=list(vals) + [vals[0]], theta=list(cats) + [cats[0]], fill="toself",
                                    line=dict(color=CYAN, width=2.5), fillcolor=CYAN + "38", marker=dict(size=6, color=PINK)))
    ax = dict(gridcolor=T["track"], linecolor=T["track"])
    fig.update_layout(title=title, showlegend=False, polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], tickfont=dict(size=8, color=T["muted"]), **ax),
                                                                 angularaxis=dict(tickfont=dict(size=11, color=T["sub"]), **ax)))
    card(fig, h)


def bars(labels, values, title, h=340, horiz=False):
    labels, values = list(labels), list(values)
    fig = go.Figure(go.Bar(x=values if horiz else labels, y=labels if horiz else values, orientation="h" if horiz else "v",
                           marker=dict(color=ramp(len(labels), CYAN, VIOLET)), text=values, texttemplate="%{text:,}",
                           textposition="outside", cliponaxis=False, textfont=dict(color=T["sub"], size=11)))
    try:
        fig.update_traces(marker_cornerradius=10)
    except Exception:
        pass
    if horiz:
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(visible=False)
    fig.update_layout(title=title)
    card(fig, h)


def surface3d(piv, title, xl="", yl="", h=470):
    """piv: DataFrame, index -> y axis, columns -> x axis. Adds an orbit animation button."""
    fig = go.Figure(go.Surface(z=piv.values, x=[str(c)[:10] for c in piv.columns], y=[str(i) for i in piv.index],
                               colorscale=[[0, "#a9c9ea" if LIGHT else "#0b3a6b"], [0.35, CYAN], [0.7, VIOLET], [1, PINK]], showscale=False,
                               contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1))))
    ax = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=T["track"], color=T["sub"], showbackground=False)
    fig.update_layout(title=title, scene=dict(xaxis=dict(title=xl, **ax), yaxis=dict(title=yl, **ax), zaxis=dict(title="Posts", **ax),
                                              aspectratio=dict(x=1.3, y=1, z=0.55), camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9))))
    fig.frames = [go.Frame(layout=dict(scene=dict(camera=dict(eye=dict(x=2.1 * np.cos(a), y=2.1 * np.sin(a), z=0.9)))))
                  for a in np.linspace(-0.8, 5.5, 70)]
    fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, x=0.03, y=0.06, bgcolor=T["tip"], bordercolor=CYAN,
                                        font=dict(color=T["text"]),
                                        buttons=[dict(label="▶  Orbit", method="animate",
                                                      args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True,
                                                                       transition=dict(duration=0), mode="immediate")])])])
    card(fig, h)


def progress(items, title):
    rows = "".join(f'<div class="prow"><div class="prow-top"><span>{esc(l)}</span><b>{v}</b></div>'
                   f'<div class="ptrack"><div class="pfill" style="width:{max(0, min(100, p))}%"></div></div></div>' for l, v, p in items)
    with st.container(border=True):
        st.markdown(f'<div class="ptitle">{title}</div>{rows}', unsafe_allow_html=True)


def feed(df, n=8):
    rows = []
    for i, (_, r) in enumerate(df.head(n).iterrows()):
        s = str(r.get("sentiment", ""))
        c = SENT.get(s, SLATE)
        src = str(r.get("source", "post"))
        txt = r.get("text") if pd.notna(r.get("text")) else r.get("clean_text", "")
        chip = f'<span class="chip" style="color:{c};background:{c}22">{SENT_EMOJI.get(s, "")} {esc(s)}</span>' if s in SENT else ""
        tp = f'<span class="chip" style="color:{VIOLET};background:{VIOLET}22">#{esc(r.get("topic"), 24)}</span>' if pd.notna(r.get("topic", np.nan)) else ""
        rows.append(f'<div class="post" style="--a:{ACCENT[i % 7]};--b:{ACCENT[(i + 2) % 7]};animation-delay:{i * .07}s"><div class="av">{esc(src[:1]).upper()}</div>'
                    f'<div style="flex:1"><div class="pmeta"><b>{esc(src)}</b><span>{esc(str(r.get("created_utc", ""))[:16])}</span>{chip}{tp}</div>'
                    f'<div class="ptext">{esc(txt, 220)}</div></div></div>')
    with st.container(border=True):
        st.markdown('<div class="ptitle">🕐 Latest posts</div>' + "".join(rows), unsafe_allow_html=True)


def entity_counts(df, tcol):
    return pd.Series({e: int(df[tcol].astype(str).str.lower().str.count(e).sum()) for e in ENTITIES}).sort_values(ascending=False)


def num(v):
    return "n/a" if pd.isna(v) else v


# ---------------------------------------------------------------------------
# PAGE 1 — Raw Signal
# ---------------------------------------------------------------------------
def page_round1():
    head("Raw Signal", "Round 1 · broken data in, trustworthy data out")
    cleaned, corrupted, users = (read_any(FILES[k]) for k in ("r1_cleaned", "r1_corrupted", "r1_users"))
    if cleaned is None and corrupted is None:
        st.warning(f"Couldn't find `{FILES['r1_cleaned']}` or `{FILES['r1_corrupted']}` (.xlsx/.csv/.json).")
        return
    posts = pd.concat([d for d in (cleaned, corrupted) if d is not None], ignore_index=True)
    total = len(posts)
    pct = lambda c: posts[c].astype(bool).mean() * 100 if c in posts.columns else np.nan
    p_text, p_time, p_like, p_plat = pct("text_missing"), pct("timestamp_missing"), pct("likes_anomalous"), pct("platform_missing")
    p_clean = len(cleaned) / total * 100 if cleaned is not None else np.nan
    kpi_row([
        dict(label="Total posts", value=total, icon="🗂️", ac=(CYAN, BLUE)),
        dict(label="Missing text", value=num(p_text), icon="❓", ac=(PINK, VIOLET), suffix="%", dec=1, delta="needs repair", tone="down"),
        dict(label="Anomalous likes", value=num(p_like), icon="⚠️", ac=(AMBER, CORAL), suffix="%", dec=1, delta="flagged", tone="down"),
        dict(label="Missing timestamps", value=num(p_time), icon="🕒", ac=(VIOLET, BLUE), suffix="%", dec=1),
        dict(label="Cleaned share", value=num(p_clean), icon="✅", ac=(GREEN, CYAN), suffix="%", dec=1, delta="recovered", tone="up"),
    ])
    c1, c2, c3 = st.columns(3)
    with c1:
        if "platform" in posts.columns:
            vc = posts["platform"].value_counts()
            bars(vc.index, vc.values, "Posts by platform", 340)
    with c2:
        dims, vals = [], []
        for l, p in [("Text", p_text), ("Time", p_time), ("Likes", p_like), ("Platform", p_plat), ("Cleaned", 100 - p_clean if not pd.isna(p_clean) else np.nan)]:
            if not pd.isna(p):
                dims.append(l)
                vals.append(round(100 - p, 1))
        if len(dims) >= 3:
            radar(dims, vals, "Data health radar", 340)
    with c3:
        if cleaned is not None and corrupted is not None:
            donut(pd.DataFrame({"status": ["Cleaned", "Corrupted"], "n": [len(cleaned), len(corrupted)]}), "status", "Cleaned vs corrupted",
                  340, {"Cleaned": GREEN, "Corrupted": CORAL}, "n")
    if users is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            if "language" in users.columns:
                donut(users, "language", "User language mix")
        with c2:
            if "location" in users.columns:
                tl = users["location"].value_counts().head(8)
                bars(tl.index, tl.values, "Top locations", 330, horiz=True)
        with c3:
            if "follower_count" in users.columns:
                fig = go.Figure(go.Histogram(x=users["follower_count"], nbinsx=40, marker=dict(color=VIOLET, line=dict(color=CYAN, width=.5)), opacity=.9))
                fig.update_layout(title="Follower-count distribution")
                card(fig, 330)
    if not pd.isna(p_text):
        callout(f"<b>Say out loud:</b> Real social data arrives broken — {p_text:.1f}% had missing text and "
                f"{p_like:.1f}% had corrupted engagement numbers. Cleaning it was step one of rebuilding the engine.")


# ---------------------------------------------------------------------------
# PAGE 2 — Understanding
# ---------------------------------------------------------------------------
def page_round2():
    head("Understanding", "Round 2 · sentiment and topic models")
    c1, c2 = st.columns(2)
    with c1:
        gauge(64.6, "Sentiment accuracy")
        st.caption("macro-F1 0.647 — from your Round 2 report")
    with c2:
        gauge(96.8, "Topic accuracy")
        st.caption("macro-F1 0.831 — from your Round 2 report")
    lab = read_any(FILES["r2_labeled"])
    if lab is not None:
        c1, c2 = st.columns(2)
        with c1:
            if "sentiment_label" in lab.columns:
                donut(lab, "sentiment_label", "Sentiment balance (training data)", cmap=SENT)
        with c2:
            if "topic_category" in lab.columns:
                donut(lab, "topic_category", "Topic balance (training data)")
        c1, c2 = st.columns(2)
        with c1:
            if "sentiment_label" in lab.columns:
                sc = lab["sentiment_label"].dropna().value_counts()
                fig = go.Figure(go.Bar(x=sc.index.tolist(), y=sc.values.tolist(), marker_color=[SENT.get(k, CYAN) for k in sc.index],
                                       text=sc.values.tolist(), texttemplate="%{text:,}", textposition="outside", cliponaxis=False))
                try:
                    fig.update_traces(marker_cornerradius=10)
                except Exception:
                    pass
                fig.update_layout(title="Posts by sentiment")
                card(fig, 330)
        with c2:
            if "topic_category" in lab.columns:
                tc = lab["topic_category"].dropna().value_counts().head(8)
                bars(tc.index, tc.values, "Top discussion topics", 330, horiz=True)
    else:
        st.info(f"No `{FILES['r2_labeled']}` found — label-balance charts skipped.")
    tp = read_any(FILES["r2_topic_preds"])
    if tp is not None and {"true", "pred"}.issubset(tp.columns):
        labels = sorted(tp["true"].unique())
        f1s = f1_score(tp["true"], tp["pred"], labels=labels, average=None)
        progress([(l, f"{v * 100:.1f}%", v * 100) for l, v in zip(labels, f1s)], "Topic per-class F1")
    callout("<b>Say out loud:</b> Topic accuracy looks high mainly because 86% of posts are one class — "
            "macro-F1 is the fairer number, and it shows the model is really learning, not exploiting imbalance.")


# ---------------------------------------------------------------------------
# PAGE 3 — Application
# ---------------------------------------------------------------------------
def page_round3():
    head("Application", "Round 3 · Vijay–Trisha live case study")
    df, label = read_any(FILES["r3_processed"]), "round3_processed"
    if df is None:
        df, label = read_any(FILES["r3_translated"]), "translated_checkpoint"
    if df is None:
        df, label = read_any(FILES["r3_live"]), "live_dataset (raw — no sentiment/topic columns expected)"
    if df is None:
        st.warning("No Round 3 file found (checked round3_processed, translated_checkpoint, live_dataset).")
        return
    st.caption(f"Using: {label}")
    dc = "date" if "date" in df.columns else ("created_utc" if "created_utc" in df.columns else None)
    if dc:
        df[dc] = pd.to_datetime(df[dc], errors="coerce", utc=True)
        df["_date"] = df[dc].dt.floor("D")
    if "source" in df.columns:
        opts = sorted(df["source"].dropna().unique())
        with st.expander("🎛️ Source filter"):
            df = df[df["source"].isin(st.multiselect("Sources", opts, default=opts))]
    tcol = "clean_text" if "clean_text" in df.columns else ("text" if "text" in df.columns else None)
    daily = df.groupby("_date").size().reset_index(name="volume") if "_date" in df.columns else None
    ec = entity_counts(df, tcol) if tcol else None
    span = int((df["_date"].max() - df["_date"].min()).days + 1) if daily is not None and df["_date"].notna().any() else "n/a"
    kpi_row([
        dict(label="Posts collected", value=len(df), icon="📥", ac=(CYAN, BLUE), series=daily["volume"] if daily is not None else None),
        dict(label="Days tracked", value=span, icon="📅", ac=(VIOLET, PINK)),
        dict(label="Languages", value=df["lang"].nunique() if "lang" in df.columns else "n/a", icon="🌐", ac=(GREEN, CYAN)),
        dict(label="Most mentioned", value=ec.index[0].title() if ec is not None else "n/a", icon="👤", ac=(AMBER, CORAL)),
    ])
    if ec is not None:
        ticker([(e.title(), f"{v:,} mentions") for e, v in ec.items()])
    if daily is not None:
        wave(daily, "_date", ["volume"], "Daily post volume", h=350)
    if "sentiment" in df.columns and "lang" in df.columns:
        en = df[(df["lang"] == "en") & df["sentiment"].notna()]
        if not en.empty:
            sd = en.groupby("_date")["sentiment"].value_counts(normalize=True).unstack(fill_value=0).reset_index()
            wave(sd, "_date", [c for c in ("Negative", "Neutral", "Positive") if c in sd.columns], "Sentiment mix over time (English subset)", SENT)
    else:
        st.info("No `sentiment`/`lang` columns yet — run the Round 3 notebook's language-flag and model-scoring cells, "
                "or point this at translated_checkpoint if it already carries sentiment.")
    if tcol and "_date" in df.columns:
        ent = df[tcol].astype(str).str.lower().apply(lambda t: {e: t.count(e) for e in ENTITIES}).apply(pd.Series)
        ed = pd.concat([df["_date"], ent], axis=1).groupby("_date").sum().reset_index()
        wave(ed, "_date", ENTITIES, "Entity mentions over time")
        if len(ed) >= 2:
            surface3d(ed.set_index("_date")[ENTITIES].T, "Entity × day mention landscape (3D)", "Day", "Entity")
    c1, c2 = st.columns(2)
    with c1:
        if "lang" in df.columns:
            donut(df, "lang", "Language mix (en / ta / mixed)")
    with c2:
        if "query" in df.columns:
            q = df["query"].value_counts().head(8)
            bars(q.index, q.values, "Posts by search query", 330, horiz=True)
    callout("<b>Say out loud:</b> <i>(fill in once you've checked round3_processed against real dates)</i> — "
            "sentiment shifted on [date] following [event], entity mentions for [name] spiked around [date].")


# ---------------------------------------------------------------------------
# PAGE 4 — Live Monitoring
# ---------------------------------------------------------------------------
def page_round4():
    head("Live Monitoring", "Scraper → Round 3 analytics pipeline", live=True)
    if "last_refreshed" not in st.session_state:
        st.session_state.last_refreshed = None
    if st.button("🔄 Refresh live"):
        st.session_state.last_refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()
    df = read_any(FILES["r4_live_processed"])
    if df is None:
        st.error("Live processed dataset not found. Run the Round 3 notebook and create `datasets/live_processing_processed.csv`.")
        return
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce", utc=True)
        df = df.dropna(subset=["created_utc"])
        df["_date"] = df["created_utc"].dt.date
    total = len(df)
    n_src = df["source"].nunique() if "source" in df.columns else 0
    pos = int((df["sentiment"] == "Positive").sum()) if "sentiment" in df.columns else 0
    neg = int((df["sentiment"] == "Negative").sum()) if "sentiment" in df.columns else 0
    daily = df.groupby("_date").size() if "_date" in df.columns else None
    vd, tone = None, "flat"
    if daily is not None and len(daily) >= 2 and daily.iloc[-2] > 0:
        chg = (daily.iloc[-1] - daily.iloc[-2]) / daily.iloc[-2] * 100
        vd, tone = f"{chg:+.0f}% vs prev day", "up" if chg >= 0 else "down"
    sd_ = lambda lbl: df[df["sentiment"] == lbl].groupby("_date").size().reindex(daily.index, fill_value=0) if daily is not None and "sentiment" in df.columns else None
    kpi_row([
        dict(label="Live posts", value=total, icon="🔴", ac=(CYAN, BLUE), delta=vd, tone=tone, series=daily),
        dict(label="Sources", value=n_src, icon="🌐", ac=(VIOLET, PINK)),
        dict(label="Positive", value=pos, icon="😊", ac=(GREEN, CYAN), delta=f"{pos / total * 100 if total else 0:.0f}% of total", tone="up", series=sd_("Positive")),
        dict(label="Negative", value=neg, icon="😡", ac=(CORAL, AMBER), delta=f"{neg / total * 100 if total else 0:.0f}% of total", tone="down", series=sd_("Negative")),
    ])
    tcol = "clean_text" if "clean_text" in df.columns else ("text" if "text" in df.columns else None)
    ec = entity_counts(df, tcol) if tcol else None
    if ec is not None:
        ticker([(f"#{e}", f"{v:,}") for e, v in ec.items()])
    if daily is not None:
        wave(daily.reset_index(name="posts"), "_date", ["posts"], "Live post volume over time", h=380)
    c1, c2 = st.columns(2)
    with c1:
        if "sentiment" in df.columns:
            sc = df["sentiment"].dropna().value_counts().reset_index()
            sc.columns = ["sentiment", "count"]
            donut(sc, "sentiment", "Sentiment distribution", cmap=SENT, values="count")
    with c2:
        if "lang" in df.columns:
            lc = df["lang"].dropna().value_counts()
            bars(lc.index, lc.values, "Language distribution", 330)
    if "sentiment" in df.columns and daily is not None:
        sd = df.dropna(subset=["sentiment"]).groupby(["_date", "sentiment"]).size().unstack(fill_value=0).reset_index()
        wave(sd, "_date", ["Positive", "Neutral", "Negative"], "Sentiment trend over time", SENT, h=380)
    if "created_utc" in df.columns and daily is not None and len(daily) >= 2:
        piv = df.assign(hour=df["created_utc"].dt.hour).pivot_table(index="hour", columns="_date", values="created_utc", aggfunc="count", fill_value=0)
        surface3d(piv, "Activity landscape — hour of day × date (3D)", "Date", "Hour (UTC)")
    c1, c2 = st.columns(2)
    with c1:
        if "topic" in df.columns:
            tc = df["topic"].dropna().value_counts().head(8)
            bars(tc.index, tc.values, "Top discussion topics", 400, horiz=True)
    with c2:
        if ec is not None:
            bars(ec.index, ec.values, "Entity mentions", 400)
    c1, c2 = st.columns([1, 1])
    with c1:
        if "query" in df.columns:
            qc = df["query"].dropna().value_counts().head(8)
            bars(qc.index, qc.values, "Posts by search query", 470, horiz=True)
    with c2:
        if "created_utc" in df.columns:
            feed(df.sort_values("created_utc", ascending=False))
    st.caption(f"Last dashboard refresh: {st.session_state.last_refreshed or 'Not refreshed yet'}")


# ---------------------------------------------------------------------------
# TOP BAR + HERO + ROUTING
# ---------------------------------------------------------------------------
PAGES = {"📡 Raw Signal": page_round1, "🧠 Understanding": page_round2,
         "🎯 Application": page_round3, "🔴 Live Monitoring": page_round4}

with st.container(border=True):
    b, n, t = st.columns([1.5, 4.6, 1.1], vertical_alignment="center")
    b.markdown('<div class="brand"><div class="logo"><span>🌀</span></div>Social Engine</div>', unsafe_allow_html=True)
    page = n.radio("Stage", list(PAGES), horizontal=True, label_visibility="collapsed")
    t.toggle("☀️ Light", key="light")

NARRATIVE = ("The Social Engine went blind (Round 1: raw, messy data) → we gave it comprehension (Round 2: sentiment + topics) → "
             "we proved it on a live event (Round 3: Vijay–Trisha) → now it's fully rebuilt and monitoring in real time (Round 4).")
reacts = "".join(f'<span class="react" style="left:{l}%;animation-delay:{d}s">{e}</span>'
                 for l, d, e in [(56, 0, "❤️"), (64, 1.3, "💬"), (72, 2.6, "🔁"), (80, .7, "👍"), (88, 3.4, "🔥"), (60, 4.2, "📈"), (76, 5.1, "✨"), (93, 2, "🎯")])
steps = "".join(f'<span class="step {"on" if i == list(PAGES).index(page) else ""}">{s}</span>'
                for i, s in enumerate(["📡 R1 · Raw signal", "🧠 R2 · Understanding", "🎯 R3 · Application", "🔴 R4 · Live"]))
st.markdown(f'''<div class="glass hero">{reacts}<div class="hero-row"><div><div class="eyebrow">Data Vortex A'26 · Aaruush, SRM · Final round</div>
<div class="title">Social Engine</div><div class="narr">{NARRATIVE}</div></div>
<div class="avatar"><div class="avs"><i>S</i><i>K</i></div><div><b>Sanjana &amp; Keerthana</b><small>Presenting · Round 4</small></div></div></div><div class="steps">{steps}</div></div>''',
            unsafe_allow_html=True)

def has(stem):
    return any(os.path.exists(stem + e) for e in ("", ".xlsx", ".csv", ".json"))


with st.sidebar:
    st.markdown('<div class="brand"><div class="logo"><span>🌀</span></div>Social Engine</div>', unsafe_allow_html=True)
    pres = "".join(f'<div class="pres" style="--a:{a};--b:{b}"><i>{n[0]}</i><div><b>{n}</b><small>Presenting · Round 4</small></div></div>'
                   for n, a, b in [("Sanjana", CYAN, BLUE), ("Keerthana", VIOLET, PINK)])
    rds = "".join(f'<div class="rd {"on" if i == list(PAGES).index(page) else ""}">{ic}&nbsp; {tx}</div>'
                  for i, (ic, tx) in enumerate([("📡", "Round 1 · Raw Signal"), ("🧠", "Round 2 · Understanding"),
                                                ("🎯", "Round 3 · Application"), ("🔴", "Round 4 · Live Monitoring")]))
    ok = {k: has(v.replace(".csv", "")) for k, v in FILES.items()}
    ds = "".join(f'<div class="ds"><span><i style="background:{GREEN if v else CORAL};box-shadow:0 0 8px {GREEN if v else CORAL}"></i>'
                 f'{k.replace("_", " ").upper()}</span><span>{"✓" if v else "missing"}</span></div>' for k, v in ok.items())
    st.markdown(f'<div class="sb-h">PRESENTERS</div>{pres}<div class="sb-h">PIPELINE</div>{rds}'
                f'<div class="sb-h">DATA STATUS · {sum(ok.values())}/{len(ok)}</div>{ds}'
                '<div class="sb-f">Data Vortex A\'26<br>Aaruush · SRM</div>', unsafe_allow_html=True)

PAGES[page]()