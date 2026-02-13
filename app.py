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
# 0. 配置区 (从 Secrets 读取)
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    ALI_API_KEY = st.secrets["ALI_API_KEY"]
    ZHIPU_API_KEY = st.secrets["ZHIPU_API_KEY"]
except:
    GOOGLE_API_KEY = ""
    ALI_API_KEY = ""
    ZHIPU_API_KEY = ""

# ==========================================
# 1. 业务配置
# ==========================================
BIZ_CONFIG = {
    "logistics": {
        "name": "VastLog",
        "full_name": "VastLog International Logistics",
        "website": "www.vastlog.com",
        "phone": "+86 13780685000",
        "keywords": "international logistics, ddp shipping, sea freight",
        "context": "Reliable DDP shipping and international freight services."
    },
    "house": {
        "name": "WelluckyHouse",
        "full_name": "Wellucky Container House",
        "website": "www.welluckyhouse.com",
        "phone": "+86 18615329580",
        "keywords": "expandable container house, folding house, modular cabin",
        "context": "Professional manufacturer of premium expandable container houses."
    }
}

# ==========================================
# 2. 核心函数
# ==========================================
def clean_text(text):
    if not text: return ""
    return text.replace("**", "").replace("##", "").replace("###", "").strip()

def generate_utm(base_url, platform, biz_key):
    if not base_url: return ""
    if not base_url.startswith("http"): base_url = "https://" + base_url
    params = {"utm_source": platform.lower(), "utm_medium": "social", "utm_campaign": f"{biz_key}_ai"}
    return f"{base_url}?{urllib.parse.urlencode(params)}"

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

# ==========================================
# 3. AI 引擎 (修复 Gemini 模型名)
# ==========================================
def run_ai(engine, img, prompt, key, model):
    if engine == "google":
        try:
            genai.configure(api_key=key)
            # 修复：移除可能导致 404 的前缀
            m_name = model.replace("models/", "")
            m = genai.GenerativeModel(m_name)
            res = m.generate_content([prompt, img] if img else [prompt])
            return clean_text(res.text)
        except Exception as e: return f"Error_{engine}"
    elif engine == "zhipu":
        try:
            client = ZhipuAI(api_key=key)
            res = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            return clean_text(res.choices[0].message.content)
        except: return f"Error_{engine}"
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
        except: return f"Error_{engine}"

# ==========================================
# 4. UI 布局 (修复侧边栏与重复 ID)
# ==========================================
st.set_page_config(page_title="狮子营销助手", layout="wide")

# 侧边栏只定义一次
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

# --- TAB 1 ---
with tab1:
    c1, c2 = st.columns(2)
    files = c1.file_uploader("📂 上传素材", accept_multiple_files=True)
    draft = c2.text_area("📝 补充描述", placeholder="AI 自动分析图片内容...")
    
    b1, b2 = st.columns(2)
    do_img = b1.button("🖼️ 仅识图命名", use_container_width=True)
    do_all = b2.button("🚀 全套处理", type="primary", use_container_width=True)

    if (do_img or do_all) and files:
        st.session_state.results = []
        utm = generate_utm(cinfo['website'], plat, cbiz)
        for f in files:
            img = Image.open(f)
            # 识图命名
            p_n = f"Describe this image in 3 SEO keywords for a filename. Include '{cinfo['name'].lower()}'. Hyphens only, no ext."
            raw_n = run_ai(etype, img, p_n, ekey, sel_mod)
            # 容错处理：如果 AI 报错，使用随机名
            if "Error" in raw_n or len(raw_n) > 60:
                clean_n = f"{cinfo['name'].lower()}-{uuid.uuid4().hex[:5]}"
            else:
                clean_n = re.sub(r'[^a-z0-9\-]', '', raw_n.lower().replace(" ","-"))
            
            fname = f"{clean_n}.webp"
            # 写文案
            text = ""
            if do_all:
                p_t = f"Write a {plat} post for {cinfo['full_name']}. Content should focus on: {draft if draft else 'this image'}. Link: {utm}. Max 2 emojis."
                text = run_ai(etype, img, p_t, ekey, sel_mod)
            
            st.session_state.results.append({"img": img, "name": fname, "data": convert_image(img), "text": text})

    # 显示结果并修复 Duplicate ID
    for i, res in enumerate(st.session_state.results):
        l, r = st.columns([1, 2])
        l.image(res['img'], use_container_width=True)
        r.code(res['name'])
        if res['text']: r.text_area("文案", res['text'], height=150, key=f"txt_{i}")
        # 核心修复：为每个下载按钮增加唯一 key
        r.download_button(f"下载图片", res['data'], res['name'], key=f"dl_{i}")

# --- TAB 2 ---
with tab2:
    st.subheader("🛠️ 封面制作")
    u_c = st.file_uploader("上传背景图", type=['jpg', 'png'], key="cover_u")
    if u_c:
        bg = Image.open(u_c)
        col_l, col_r = st.columns([1, 2])
        t1 = col_l.text_input("标题1", "PREMIUM QUALITY")
        t2 = col_l.text_input("标题2", "EXPANDABLE HOUSE")
        color = col_l.color_picker("颜色", "#FFDD00")
        out = apply_style(bg.copy(), [{"text":t1,"color":"#FFF","size":0.1}, {"text":t2,"color":color,"size":0.15}])
        col_r.image(out, use_container_width=True)
        # 这里的 key 也是唯一的
        b_io = io.BytesIO(); out.save(b_io, format="PNG")
        col_r.download_button("保存封面", b_io.getvalue(), "cover.png", key="save_cover")

# --- TAB 3 ---
with tab3:
    st.subheader("🌍 GEO 专家")
    raw = st.text_area("输入中文草稿", height=200, key="geo_in")
    if st.button("✨ 执行优化", type="primary"):
        p_g = f"As a Senior SEO expert, translate/refine this into professional English with high EEAT. Content: {raw}"
        res = run_ai(etype, None, p_g, ekey, sel_mod)
        st.write(res)
