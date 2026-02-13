import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import MultiModalConversation, ImageSynthesis 
from zhipuai import ZhipuAI
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io
import zipfile
import time
import os
import urllib.parse
import requests 
from http import HTTPStatus
import uuid 
import re

# ==========================================
# 0. 初始化系统缓存 (防止 AttributeError)
# ==========================================
if 'results' not in st.session_state:
    st.session_state.results = []

# ==========================================
# 1. 核心配置与脱敏读取
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    ALI_API_KEY = st.secrets["ALI_API_KEY"]
    ZHIPU_API_KEY = st.secrets["ZHIPU_API_KEY"]
except:
    GOOGLE_API_KEY = ""
    ALI_API_KEY = ""
    ZHIPU_API_KEY = ""

BIZ_CONFIG = {
    "logistics": {
        "name": "VastLog",
        "full_name": "VastLog International Logistics",
        "website": "www.vastlog.com",
        "phone": "+86 13780685000",
        "keywords": "international logistics, ddp shipping, sea freight",
        "context": "Professional DDP and international shipping provider."
    },
    "house": {
        "name": "WelluckyHouse",
        "full_name": "Wellucky Container House",
        "website": "www.welluckyhouse.com",
        "phone": "+86 18615329580",
        "keywords": "expandable container house, folding house, luxury cabin",
        "context": "Manufacturer of high-end expandable container houses."
    }
}

# ==========================================
# 2. 核心处理工具
# ==========================================
def clean_text(text):
    if not text: return ""
    return text.replace("**", "").replace("##", "").replace("###", "").strip()

def convert_image(image):
    buf = io.BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(buf, format='WEBP', quality=80, optimize=True)
    return buf.getvalue()

def apply_style(image, configs):
    W, H = image.size
    draw = ImageDraw.Draw(image)
    total_h = 0
    lines = []
    for c in configs:
        if not c['text']: continue
        fs = int(H * c['size'])
        try: font = ImageFont.truetype("impact.ttf", fs)
        except: font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), c['text'], font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        line_h = h * 1.2
        total_h += line_h
        lines.append({"t":c['text'], "f":font, "col":c['color'], "w":w, "lh":line_h, "fs":fs})
    
    curr_y = H - total_h - (H * 0.05)
    for l in lines:
        x = (W - l['w']) / 2
        stroke = int(l['fs'] * 0.08)
        draw.text((x, curr_y), l['t'], font=l['f'], fill=l['col'], stroke_width=stroke, stroke_fill="black")
        curr_y += l['lh']
    return image

def run_ai(engine, img, prompt, key, model):
    if engine == "google":
        try:
            genai.configure(api_key=key)
            m = genai.GenerativeModel(model.replace("models/", ""))
            res = m.generate_content([prompt, img] if img else [prompt])
            return clean_text(res.text)
        except: return "Error_Google"
    elif engine == "zhipu":
        try:
            client = ZhipuAI(api_key=key)
            res = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            return clean_text(res.choices[0].message.content)
        except: return "Error_Zhipu"
    else:
        try:
            dashscope.api_key = key
            if img:
                p = f"t_{uuid.uuid4().hex}.png"
                img.save(p)
                url = f"file://{os.path.abspath(p).replace('\\', '/')}"
                res = MultiModalConversation.call(model=model, messages=[{"role":"user","content":[{"image":url},{"text":prompt}]}])
                if os.path.exists(p): os.remove(p)
                return clean_text(res.output.choices[0].message.content[0]['text'])
            else:
                from dashscope import Generation
                res = Generation.call(model="qwen-max", prompt=prompt)
                return clean_text(res.output.text)
        except: return "Error_Ali"

# ==========================================
# 3. UI 布局
# ==========================================
st.set_page_config(page_title="狮子营销助手", layout="wide")

with st.sidebar:
    st.title("⚙️ 配置中心")
    engine = st.radio("文案引擎", ("Google Gemini", "阿里通义", "智谱清言"))
    if "Google" in engine:
        etype, mlist, ekey = "google", ["gemini-1.5-flash", "gemini-1.5-pro"], GOOGLE_API_KEY
    elif "阿里" in engine:
        etype, mlist, ekey = "ali", ["qwen-vl-max", "qwen-max"], ALI_API_KEY
    else:
        etype, mlist, ekey = "zhipu", ["glm-4v", "glm-4-flash"], ZHIPU_API_KEY
    
    sel_mod = st.selectbox("选择模型", mlist)
    biz = st.radio("选择业务", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"))
    cbiz = "logistics" if "VastLog" in biz else "house"
    cinfo = BIZ_CONFIG[cbiz]
    plat = st.selectbox("发布平台", ["Facebook", "LinkedIn", "YouTube"])

st.header(f"🦁 {cinfo['name']} 数字化中心")
tab1, tab2, tab3 = st.tabs(["✍️ 智能文案", "🎨 封面工厂", "🌍 GEO 优化"])

with tab1:
    col_f, col_d = st.columns(2)
    files = col_f.file_uploader("📂 上传素材", accept_multiple_files=True, key="file_u")
    draft = col_d.text_area("📝 补充描述", placeholder="AI分析图片生成文案...", key="draft_in")
    
    b1, b2 = st.columns(2)
    if b1.button("🖼️ 仅识图命名", use_container_width=True, key="btn_name"):
        if files:
            st.session_state.results = []
            for f in files:
                img = Image.open(f)
                pn = f"Generate a 3-word SEO filename for this image. Include '{cinfo['name'].lower()}'. Hyphens only."
                rn = run_ai(etype, img, pn, ekey, sel_mod)
                cn = re.sub(r'[^a-z0-9\-]', '', rn.lower().replace(" ","-")) if "Error" not in rn else f"img-{uuid.uuid4().hex[:5]}"
                st.session_state.results.append({"img": img, "name": f"{cn}.webp", "data": convert_image(img), "text": ""})

    if b2.button("🚀 全套处理", type="primary", use_container_width=True, key="btn_all"):
        if files:
            st.session_state.results = []
            for f in files:
                img = Image.open(f)
                pn = f"SEO filename for '{cinfo['name'].lower()}'. 3 keywords."
                rn = run_ai(etype, img, pn, ekey, sel_mod)
                cn = re.sub(r'[^a-z0-9\-]', '', rn.lower().replace(" ","-")) if "Error" not in rn else f"img-{uuid.uuid4().hex[:5]}"
                pt = f"Professional {plat} post for {cinfo['full_name']}. Based on: {draft}. Web: {cinfo['website']}."
                txt = run_ai(etype, img, pt, ekey, sel_mod)
                st.session_state.results.append({"img": img, "name": f"{cn}.webp", "data": convert_image(img), "text": txt})

    # 显示结果区
    for i, res in enumerate(st.session_state.results):
        r_l, r_r = st.columns([1, 2])
        r_l.image(res['img'], use_container_width=True)
        r_r.code(res['name'])
        if res['text']: r_r.text_area("文案", res['text'], height=150, key=f"t_{i}")
        r_r.download_button(f"下载图片", res['data'], res['name'], key=f"dl_{i}")

with tab2:
    st.subheader("🛠️ 封面制作")
    u_c = st.file_uploader("上传图", type=['jpg', 'png'], key="u_c_tab2")
    if u_c:
        bg = Image.open(u_c)
        cl, cr = st.columns([1, 2])
        t1 = cl.text_input("标题1", "WELLUCKY HOUSE", key="t1_tab2")
        t2 = cl.text_input("标题2", "FACTORY PRICE", key="t2_tab2")
        col = cl.color_picker("颜色", "#FFDD00", key="col_tab2")
        out = apply_style(bg.copy(), [{"text":t1,"color":"#FFF","size":0.1}, {"text":t2,"color":col,"size":0.15}])
        cr.image(out, use_container_width=True)
        bio = io.BytesIO(); out.save(bio, format="PNG")
        cr.download_button("保存封面", bio.getvalue(), "cover.png", key="dl_tab2")

with tab3:
    st.subheader("🌍 GEO 专家 (中译英 + HTML + Schema)")
    raw_tx = st.text_area("输入发货记录或中文草稿", height=250, key="tx_tab3")
    geo_u = st.file_uploader("上传对应实拍图 (可选)", type=['jpg', 'png'], key="u_tab3")
    if st.button("✨ 执行深度优化", type="primary", key="btn_tab3"):
        if raw_tx:
            with st.spinner("AI 专家正在处理..."):
                gp = f"As a Senior SEO expert, translate/refine this into professional English. Format in HTML with <h2>, <ul>. Provide FAQ Schema. Content: {raw_tx}"
                res = run_ai(etype, Image.open(geo_u) if geo_u else None, gp, ekey, sel_mod)
                st.markdown("### 💎 优化结果")
                st.write(res)
