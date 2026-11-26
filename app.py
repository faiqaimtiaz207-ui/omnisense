import pandas as pd, collections
from rouge_score import rouge_scorer
import os, time, html, json, tempfile, subprocess
import cv2, torch, whisper, string, re
import streamlit as st
import streamlit.components.v1 as components
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

st.set_page_config(page_title="OmniSense", page_icon="🎞️", layout="wide")
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# ---------------------------------------------------------------- MODELS
@st.cache_resource(show_spinner=False)
def load_models():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto" if dev == "cuda" else None)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    asr = whisper.load_model("base", device=dev)
    return vlm, proc, asr, dev

# ---------------------------------------------------------------- PIPELINE
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def extract_audio(video_path, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
                    out_path, "-loglevel", "error"])
    return out_path if os.path.exists(out_path) and os.path.getsize(out_path) > 1000 else None

def transcribe(asr, dev, audio_path):
    if audio_path is None:
        return "", []
    r = asr.transcribe(audio_path, fp16=(dev == "cuda"))
    return r["text"].strip(), r["segments"]

def predict_timestamp(question, segments):
    if not segments:
        return None
    q = set(normalize(question).split())
    best = max(segments, key=lambda s: len(q & set(normalize(s["text"]).split())))
    return best["end"]

def build_prompt(question, transcript):
    ctx = f'Audio transcript of the video: "{transcript}"\n' if transcript else ""
    return ("You are watching a video. " + ctx +
            "Answer the question using what is visible and audible in the video. "
            "Do not refuse or say you cannot answer. Reply with a short answer.\n"
            f"Question: {question}")

def answer_question(vlm, proc, video_path, prompt, fps=2.0, max_pixels=360 * 420):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": f"file://{video_path}", "max_pixels": max_pixels, "fps": fps},
        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img, vid = process_vision_info(messages)
    inputs = proc(text=[text], images=img, videos=vid, padding=True,
                  return_tensors="pt").to(vlm.device)
    t0 = time.time()
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=64)
    secs = round(time.time() - t0, 2)
    out = out[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(out, skip_special_tokens=True)[0].strip(), secs

def video_duration(path, fallback=10.0):
    cap = cv2.VideoCapture(path)
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps and n else fallback

def fmt(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"

# ---------------------------------------------------------------- STYLE
# Palette: #5D001E burgundy | #E3E2DF stone | #E3AFBC blush | #9A1750 plum | #EE4C7C rose
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;800&family=Newsreader:ital,wght@0,400;1,400;1,600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--bg:#E3E2DF;--panel:#EFEEEB;--ink:#5D001E;--plum:#9A1750;--rose:#EE4C7C;--blush:#E3AFBC;--line:#C9BFC2}
header[data-testid="stHeader"],#MainMenu,footer{display:none!important}
.stApp{background:radial-gradient(circle,rgba(154,23,80,.10) 1px,transparent 1.4px) 0 0/16px 16px,var(--bg);
 color:var(--ink);font-family:'IBM Plex Mono',monospace}
.block-container{padding-top:1rem;max-width:1250px}
.stApp p,.stApp label,.stApp span,.stApp li{color:var(--ink)}
.slate{font-family:'Barlow Condensed',sans-serif;font-weight:800;letter-spacing:.2em;font-size:15px;color:var(--ink);
 margin:6px 0 12px;display:flex;align-items:center;gap:10px;text-transform:uppercase}
.slate::before{content:"";width:22px;height:8px;background:var(--rose)}
.slate::after{content:"";flex:1;height:2px;background:var(--ink)}
[data-testid="stFileUploader"] section{background:var(--panel);border:2px dashed var(--plum);border-radius:2px;padding:26px;transition:.25s}
[data-testid="stFileUploader"] section:hover{border-color:var(--rose);background:#F6E6EB;transform:translateY(-2px)}
[data-testid="stFileUploader"] *{color:var(--ink)!important;font-family:'IBM Plex Mono',monospace!important}
[data-testid="stFileUploader"] button{background:var(--ink)!important;border:0!important;border-radius:2px!important}
[data-testid="stFileUploader"] button *{color:var(--bg)!important}
.stTextInput input{background:var(--panel)!important;color:var(--ink)!important;border:2px solid var(--ink)!important;
 border-radius:2px!important;font-family:'IBM Plex Mono',monospace!important;padding:14px!important}
.stTextInput input::placeholder{color:#9A1750!important;opacity:.7}
.stTextInput input:focus{border-color:var(--rose)!important;box-shadow:5px 5px 0 var(--rose)!important}
.stButton>button{width:100%;background:var(--ink);color:var(--bg);border:2px solid var(--ink);border-radius:2px;
 font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:24px;letter-spacing:.14em;padding:10px;
 box-shadow:6px 6px 0 var(--rose);transition:.15s}
.stButton>button *{color:var(--bg)!important}
.stButton>button:hover{transform:translate(3px,3px);box-shadow:3px 3px 0 var(--rose);background:var(--plum);border-color:var(--plum)}
.stButton>button:active{transform:translate(6px,6px);box-shadow:0 0 0 var(--rose)}
video{border-radius:2px;border:2px solid var(--ink)}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:4px 0 16px}
.step{border:1.5px solid var(--line);padding:10px;font-size:11px;letter-spacing:.12em;color:#8a7b80;background:var(--panel);transition:.3s}
.step b{display:block;font-family:'Barlow Condensed';font-weight:800;font-size:28px;color:var(--line)}
.step.active{border-color:var(--rose);color:var(--plum);animation:pulse 1s infinite}
.step.active b{color:var(--rose)}
.step.done{color:var(--ink);border-color:var(--ink)}
.step.done b{color:var(--plum)}
@keyframes pulse{50%{box-shadow:0 0 18px rgba(238,76,124,.45)}}
.card{background:var(--panel);border:2px solid var(--ink);margin-bottom:14px}
.clap{height:14px;background:repeating-linear-gradient(115deg,var(--ink) 0 18px,var(--rose) 18px 36px)}
.card .in{padding:14px 18px}
.card h4{margin:0 0 8px;font-family:'Barlow Condensed',sans-serif;font-size:15px;letter-spacing:.22em;color:var(--plum);font-weight:800}
.seg{display:flex;gap:12px;font-size:13px;padding:6px 0;border-bottom:1px dashed var(--blush);color:var(--ink)}
.seg i{color:var(--plum);font-style:normal;white-space:nowrap}
.empty{border:2px dashed var(--plum);padding:60px 20px;text-align:center;color:var(--plum);letter-spacing:.2em;font-size:12px;background:rgba(227,175,188,.25)}
.stats{display:flex;gap:10px;font-size:12px;color:var(--ink);margin-top:6px;flex-wrap:wrap}
.stats span{border:1.5px solid var(--ink);padding:4px 10px;background:var(--blush)}
[data-testid="stMetric"]{background:var(--panel);border:2px solid var(--ink);padding:12px 14px;border-left:8px solid var(--rose)}
[data-testid="stMetricLabel"] *{color:var(--plum)!important;font-family:'Barlow Condensed',sans-serif;font-weight:800;letter-spacing:.12em;font-size:15px}
[data-testid="stMetricValue"]{color:var(--ink)!important;font-family:'Barlow Condensed',sans-serif;font-weight:800}
[data-testid="stExpander"] details{border:2px solid var(--ink);background:var(--panel)}
.credits{text-align:center;color:var(--plum);font-size:11px;letter-spacing:.25em;margin-top:40px}
</style>
"""

HERO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;800&family=Newsreader:ital,wght@0,400;1,400;1,600&family=IBM+Plex+Mono:wght@400&display=swap');
html,body{margin:0;background:transparent;color:#5D001E;font-family:'IBM Plex Mono',monospace;overflow:hidden}
.top{display:flex;justify-content:space-between;font-size:12px;letter-spacing:.22em;color:#5D001E;padding-top:12px}
.rec{color:#9A1750}.rec i{display:inline-block;width:9px;height:9px;border-radius:50%;background:#EE4C7C;margin-right:8px;animation:b 1s infinite}
@keyframes b{50%{opacity:.1}}
h1{font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:clamp(40px,5.5vw,64px);line-height:1;
 margin:10px 0 6px;letter-spacing:.01em;color:#5D001E;text-transform:uppercase}
.m{display:inline-block;overflow:hidden;vertical-align:top;padding-bottom:4px}
.m span{display:inline-block}.a .m span{color:#EE4C7C}
.sub{font-family:'Newsreader',serif;font-style:italic;font-size:clamp(21px,2.5vw,30px);line-height:1.35;color:#5D001E;
 max-width:900px;opacity:0;margin-top:2px}
.hl{font-weight:600;padding:0 5px;background:linear-gradient(#E3AFBC,#E3AFBC) no-repeat left 88%/0% 55%}
.strip{margin-top:16px;background:#5D001E;position:relative;height:46px;overflow:hidden}
.h{position:absolute;left:0;width:100%;height:6px;background:repeating-linear-gradient(90deg,#E3E2DF 0 12px,transparent 12px 26px)}
.h.t{top:4px}.h.bt{bottom:4px}
.track{position:absolute;top:13px;left:0;display:flex;white-space:nowrap}
.f{width:96px;height:20px;border:1px solid #9A1750;margin-right:10px;font-size:9px;letter-spacing:.2em;color:#E3AFBC;
 display:flex;align-items:center;justify-content:center;flex:none}
.f.hot{background:#EE4C7C;color:#5D001E;border-color:#EE4C7C}
</style>
<div class="top"><div class="rec"><i></i>REC &nbsp;<span id="tc">00:00:00:00</span></div>
<div>VIDEO Q&amp;A &middot; MK-I</div></div>
<h1><span id="l1"></span><span id="l2" class="a"></span></h1>
<div class="sub">Ask any clip a question. It <span class="hl">watches</span>, it <span class="hl">listens</span>, and it shows you the <span class="hl">exact second</span> the answer lives.</div>
<div class="strip"><div class="h t"></div><div class="h bt"></div><div class="track" id="tr"></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script>
function split(id,txt){var e=document.getElementById(id);
 txt.split('').forEach(function(c){var m=document.createElement('span');m.className='m';
 var s=document.createElement('span');s.textContent=c;m.appendChild(s);e.appendChild(m);});}
split('l1','OMNI');split('l2','SENSE');
var tr=document.getElementById('tr');
for(var k=0;k<2;k++){for(var i=1;i<=30;i++){var d=document.createElement('div');d.className='f'+(i%7==0?' hot':'');
 d.textContent=('0000'+i).slice(-4);tr.appendChild(d);}}
gsap.from('.m span',{yPercent:115,duration:1,stagger:.06,ease:'power4.out'});
gsap.to('.sub',{opacity:1,duration:.9,delay:.7});
gsap.to('.hl',{backgroundSize:'100% 55%',duration:.8,delay:1.3,stagger:.35,ease:'power2.out'});
gsap.from('.strip',{scaleX:0,transformOrigin:'left',duration:1.1,delay:.3,ease:'power3.out'});
gsap.to('#tr',{x:-(96+10)*30,duration:40,ease:'none',repeat:-1});
var t0=Date.now();setInterval(function(){var t=(Date.now()-t0)/1000,p=function(n){return('0'+n).slice(-2)};
 document.getElementById('tc').textContent=p(Math.floor(t/3600))+':'+p(Math.floor(t/60)%60)+':'+p(Math.floor(t)%60)+':'+p(Math.floor((t%1)*24));},40);
</script>
"""

ANSWER = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;800&family=IBM+Plex+Mono&display=swap');
html,body{margin:0;background:transparent}
.card{background:#EFEEEB;border:2px solid #5D001E;color:#5D001E}
.clap{height:14px;background:repeating-linear-gradient(115deg,#5D001E 0 18px,#EE4C7C 18px 36px)}
.in{padding:14px 18px 20px}
h4{margin:0 0 10px;font:800 15px 'Barlow Condensed',sans-serif;letter-spacing:.22em;color:#9A1750}
#a{font:800 40px/1.1 'Barlow Condensed',sans-serif;min-height:50px;color:#5D001E;text-transform:uppercase}
.cur{display:inline-block;width:14px;height:32px;background:#EE4C7C;vertical-align:-4px;margin-left:4px;animation:b .8s infinite}
@keyframes b{50%{opacity:0}}
</style>
<div class="card" id="c"><div class="clap"></div><div class="in"><h4>ANSWER &middot; TAKE 01</h4>
<div id="a"></div></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script>
var txt=__TEXT__, el=document.getElementById('a'), o={n:0};
gsap.from('#c',{y:30,opacity:0,duration:.6,ease:'power3.out'});
gsap.to(o,{n:txt.length,duration:Math.min(2.5,txt.length*0.03),delay:.5,ease:'none',
 onUpdate:function(){el.innerHTML=txt.slice(0,Math.round(o.n)).replace(/</g,'&lt;')+'<span class="cur"></span>';}});
</script>
"""

TIMELINE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@800&family=IBM+Plex+Mono&display=swap');
html,body{margin:0;background:transparent;font-family:'IBM Plex Mono',monospace;color:#5D001E}
.card{background:#EFEEEB;border:2px solid #5D001E;padding:14px 18px 16px}
h4{margin:0 0 22px;font:800 15px 'Barlow Condensed',sans-serif;letter-spacing:.22em;color:#9A1750}
.track{position:relative;height:34px;background:#E3E2DF;border:1.5px solid #5D001E}
.tick{position:absolute;top:0;width:1px;height:10px;background:#9A1750}
.fill{position:absolute;left:0;top:0;height:100%;width:0;background:linear-gradient(90deg,transparent,rgba(238,76,124,.45))}
.mark{position:absolute;top:-9px;width:0;height:0;border-left:7px solid transparent;border-right:7px solid transparent;
 border-top:11px solid #9A1750;transform:translateX(-7px) scale(0)}
.pin{position:absolute;top:0;width:2px;height:100%;background:#9A1750;opacity:0}
.head{position:absolute;top:-6px;width:3px;height:46px;background:#EE4C7C;box-shadow:0 0 12px #EE4C7C}
.r{display:flex;justify-content:space-between;margin-top:12px;font-size:12px;color:#5D001E}
.r b{color:#9A1750;font-weight:500}#cur{color:#EE4C7C}
.none{color:#9A1750;font-size:13px;padding:6px 0}
</style>
<div class="card"><h4>EVIDENCE TIMELINE</h4><div id="w"></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script>
var dur=__DUR__, ts=__TS__, w=document.getElementById('w');
function f(t){var m=Math.floor(t/60),s=(t%60).toFixed(1);return('0'+m).slice(-2)+':'+(s.length<4?'0':'')+s;}
if(ts<0){w.innerHTML='<div class="none">No speech detected in this clip, so no spoken-evidence timestamp.</div>';}
else{
 w.innerHTML='<div class="track" id="t"><div class="fill" id="fi"></div><div class="pin" id="pn"></div><div class="mark" id="mk"></div><div class="head" id="hd"></div></div>'+
 '<div class="r"><span id="cur">00:00.0</span><span>EVIDENCE @ <b>'+f(ts)+'</b></span><span>'+f(dur)+'</span></div>';
 var t=document.getElementById('t');
 for(var s=0;s<=Math.floor(dur);s++){var k=document.createElement('div');k.className='tick';k.style.left=(s/dur*100)+'%';t.appendChild(k);}
 var pct=Math.min(100,ts/dur*100),o={p:0};
 document.getElementById('mk').style.left=pct+'%';document.getElementById('pn').style.left=pct+'%';
 gsap.to(o,{p:pct,duration:1.8,delay:.4,ease:'power3.inOut',onUpdate:function(){
  document.getElementById('hd').style.left=o.p+'%';document.getElementById('fi').style.width=o.p+'%';
  document.getElementById('cur').textContent=f(o.p/100*dur);},
  onComplete:function(){gsap.to('#mk',{scale:1,duration:.7,ease:'elastic.out(1,.4)'});gsap.to('#pn',{opacity:.9,duration:.3});}});
}
</script>
"""

def stepper(active):
    names = ["EXTRACT AUDIO", "WHISPER ASR", "QWEN2-VL", "LOCATE EVIDENCE"]
    out = '<div class="steps">'
    for i, n in enumerate(names):
        c = "done" if i < active else ("active" if i == active else "")
        out += f'<div class="step {c}"><b>0{i+1}</b>{n}</div>'
    return out + "</div>"

# ---------------------------------------------------------------- UI
st.markdown(CSS, unsafe_allow_html=True)
components.html(HERO, height=320)

left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown('<div class="slate">01 · INPUT</div>', unsafe_allow_html=True)
    up = st.file_uploader("clip", type=["mp4", "mov", "avi", "mkv", "webm"],
                          label_visibility="collapsed")
    if up:
        st.video(up)
    q = st.text_input("question", placeholder="Ask something about the clip...",
                      label_visibility="collapsed")
    go = st.button("▶  ROLL CAMERA")

with right:
    st.markdown('<div class="slate">02 · OUTPUT</div>', unsafe_allow_html=True)
    if not go:
        st.markdown('<div class="empty">AWAITING FOOTAGE<br><br>UPLOAD A CLIP · ASK A QUESTION · ROLL CAMERA</div>',
                    unsafe_allow_html=True)
    elif not up or not q.strip():
        st.warning("Upload a video and type a question first.")
    else:
        steps = st.empty()
        steps.markdown(stepper(0), unsafe_allow_html=True)
        with st.spinner("Warming up the models (first run only)..."):
            vlm, proc, asr, dev = load_models()

        tmp = tempfile.mkdtemp()
        vpath = os.path.join(tmp, "clip" + os.path.splitext(up.name)[1])
        with open(vpath, "wb") as f:
            f.write(up.getbuffer())

        t_start = time.time()
        ap = extract_audio(vpath, os.path.join(tmp, "audio.wav"))
        steps.markdown(stepper(1), unsafe_allow_html=True)
        transcript, segs = transcribe(asr, dev, ap)
        steps.markdown(stepper(2), unsafe_allow_html=True)
        if dev == "cpu":
            ans, secs = answer_question(
                vlm, proc, vpath, build_prompt(q, transcript),
                fps=1.0, max_pixels=224 * 224
            )
        else:
            ans, secs = answer_question(
                vlm, proc, vpath, build_prompt(q, transcript)
            )
        steps.markdown(stepper(3), unsafe_allow_html=True)
        ts = predict_timestamp(q, segs)
        dur = video_duration(vpath, fallback=(segs[-1]["end"] if segs else 10.0))
        steps.markdown(stepper(4), unsafe_allow_html=True)

        components.html(ANSWER.replace("__TEXT__", json.dumps(ans).replace("</", "<\\/")), height=150)
        components.html(TIMELINE.replace("__DUR__", str(round(dur, 2)))
                        .replace("__TS__", str(round(ts, 2)) if ts is not None else "-1"), height=135)

        if segs:
            rows = "".join(f'<div class="seg"><i>{fmt(s["start"])} → {fmt(s["end"])}</i>'
                           f'<span>{html.escape(s["text"].strip())}</span></div>' for s in segs[:8])
        else:
            rows = '<div class="seg"><span style="color:#9A1750">(no speech detected)</span></div>'
        st.markdown(f'<div class="card"><div class="clap"></div><div class="in"><h4>AUDIO TRANSCRIPT</h4>{rows}</div></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="stats"><span>VLM {secs}s</span><span>TOTAL {round(time.time()-t_start,1)}s</span>'
                    f'<span>{dev.upper()}</span><span>{round(dur,1)}s CLIP</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- EVALUATION
@st.cache_data(show_spinner=False)
def eval_table():
    p = "omnisense_full_results.csv"
    if not os.path.exists(p):
        return None, None, None
    d = pd.read_csv(p)
    for c in ["base", "enh", "gt"]:
        d[c] = d[c].fillna("").astype(str)
    rg = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def em(p, g): return float(normalize(p) == normalize(g))
    def cm(p, g): return float(normalize(g) in normalize(p))
    def f1(p, g):
        a, b = normalize(p).split(), normalize(g).split()
        s = sum((collections.Counter(a) & collections.Counter(b)).values())
        if s == 0: return 0.0
        pr, rc = s / len(a), s / len(b)
        return 2 * pr * rc / (pr + rc)

    out = {}
    for name, col in [("Baseline", "base"), ("Enhanced", "enh")]:
        out[name] = {
            "Exact Match": d.apply(lambda r: em(r[col], r["gt"]), axis=1).mean(),
            "Contains Match": d.apply(lambda r: cm(r[col], r["gt"]), axis=1).mean(),
            "Token F1": d.apply(lambda r: f1(r[col], r["gt"]), axis=1).mean(),
            "ROUGE-L": d.apply(lambda r: rg.score(r["gt"], r[col])["rougeL"].fmeasure, axis=1).mean(),
        }
    ts = d.dropna(subset=["pred_timestamp"]) if "pred_timestamp" in d else None
    return d, pd.DataFrame(out), ts

st.markdown('<div class="slate">03 · EVALUATION · QIVD PROTOTYPE RUN</div>', unsafe_allow_html=True)
d, tbl, ts = eval_table()
if d is None:
    st.info("Results file not found.")
else:
    st.markdown(f'<div class="stats"><span>{len(d)} QIVD SAMPLES</span>'
                f'<span>BASELINE vs ENHANCED</span></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for c, m in zip(cols, tbl.index):
        gain = tbl.loc[m, "Enhanced"] - tbl.loc[m, "Baseline"]
        c.metric(m, f'{tbl.loc[m, "Enhanced"]:.3f}', f"{gain:+.3f} vs baseline")
    st.bar_chart(tbl, color=["#9A1750", "#EE4C7C"])
    if ts is not None and len(ts):
        err = (ts["pred_timestamp"] - ts["timestamp_sec"]).abs().mean()
        st.caption(f"Timestamp coverage: {len(ts)}/{len(d)} clips · mean absolute error {err:.2f} s")
    with st.expander("Browse predictions"):
        st.dataframe(d[["question", "gt", "base", "enh"]].head(300), use_container_width=True)

st.markdown('<div class="credits">QWEN2-VL-2B · WHISPER BASE · TRANSFORMER PIPELINE</div>', unsafe_allow_html=True)
