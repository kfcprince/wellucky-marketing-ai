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
# 1. 业务大脑
# ==========================================
BIZ_CONFIG = {
    "logistics": {
        "name": "VastLog",
        "full_name": "VastLog International Logistics",
        "website": "www.vastlog.com",
        "email": "info@vastlog.com",
        "phone": "+86 13780685000",
        "keywords": "international logistics, ddp shipping, sea freight, air cargo",
        "context": "We provide reliable international shipping services, focusing on DDP.",
        "buffer_url": "https://publish.buffer.com/profile/你的物流ID"
    },
    "house": {
        "name": "WelluckyHouse",
        "full_name": "Wellucky Container House",
        "website": "www.welluckyhouse.com",
        "email": "info@welluckyhouse.com",
        "phone": "+86 18615329580",
        "keywords": "expandable container house, folding house, apple cabin",
        "context": "We manufacture high-quality expandable container houses.",
        "buffer_url": "https://publish.buffer.com/profile/你的房屋ID"
    }
}

# ==========================================
# 2. 核心工具函数
# ==========================================
def clean_text(text):
    if not text: return ""
    return text.replace("**", "").replace("##", "").replace("###", "").strip()

def generate_utm(base_url, platform, biz_key):
    if not base_url: return ""
    if not base_url.startswith("http"): base_url = "https://" + base_url
    params = {"utm_source": platform.lower(), "utm_medium": "social", "utm_campaign": f"{biz_key}_ai_batch"}
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def convert_image(image, quality=80):
    img_byte_arr = io.BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(img_byte_arr, format='WEBP', quality=quality, optimize=True)
    return img_byte_arr.getvalue()

# ==========================================
# 3. 图像处理 (支持 3 标题预览)
# ==========================================
def load_font_safe(size):
    try: return ImageFont.truetype("impact.ttf", size)
    except: return ImageFont.load_default()

def apply_youtube_style(image, text_configs):
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.4) 
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    W, H = image.size
    draw = ImageDraw.Draw(image)
    lines_to_draw = []
    total_h = 0
    for cfg in text_configs:
        if not cfg['text']: continue
        f_size = int(H * cfg['size'])
        font = load_font_safe(f_size)
        bbox = draw.textbbox((0, 0), cfg['text'], font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        line_h = h * 1.2
        total_h += line_h
        lines_to_draw.append({"text": cfg['text'], "font": font, "color": cfg['color'], "w": w, "line_h": line_h, "fs": f_size})
    
    curr_y = H - total_h - (H * 0.05)
    for line in lines_to_draw:
        x = (W - line['w']) / 2
        stroke = int(line['fs'] * 0.08)
        draw.text((x, curr_y), line['text'], font=line['font'], fill=line['color'], stroke_width=stroke, stroke_fill="black")
        curr_y += line['line_h']
    return image

# ==========================================
# 4. AI 引擎 (修复命名与 GEO 逻辑)
# ==========================================
def get_prompt(info, platform, user_draft, link, task_type):
    contact = f"Web: {info['website']}, WhatsApp: {info['phone']}"
    if task_type == "content":
        return f"Role: Social Media Manager. Write a post for {platform} about {info['full_name']}. Draft: {user_draft}. Link: {link}. Contact: {contact}. Rules: Professional, Max 2 emojis."
    elif task_type == "geo":
        return f"""
        Role: Senior SEO & GEO Specialist. 
        Task: Translate/Refine to professional English for {info['full_name']}.
        Requirements:
        1. EEAT & SEO: Authoritative tone.
        2. HTML Layout: Output in a <div> with <h2>(blue left border), <p>, <ul>, and <img> placeholders.
        3. Schema: Provide JSON-LD FAQ Schema at the end.
        Content: {user_draft}
        """
    else:
        return f"""
        Task: Generate an SEO filename (3-5 keywords) based on this image.
        Rules: Lowercase, hyphens only, include brand '{info['name'].lower()}', no extension.
        Example: welluckyhouse-expandable-home-solar-roof
        """

def run_text_engine(engine, img_obj, prompt, api_key, model):
    if engine == "zhipu":
        try:
            client = ZhipuAI(api_key=api_key)
            # 智谱识图需要特定模型
            res = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            return clean_text(res.choices[0].message.content)
        except Exception as e: return f"Error: {e}"
    elif engine == "google":
        try:
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel(model)
            res = m.generate_content([prompt, img_obj] if img_obj else [prompt])
            return clean_text(res.text)
        except Exception as e: return f"Error: {e}"
    else:
        try:
            dashscope.api_key = api_key
            if img_obj:
                path = f"tmp_{uuid.uuid4().hex}.png"
                img_obj.save(path)
                f_url = f"file://{os.path.abspath(path).replace('\\', '/')}"
                msgs = [{"role": "user", "content": [{"image": f_url}, {"text": prompt}]}]
                res = MultiModalConversation.call(model=model, messages=msgs)
                if os.path.exists(path): os.remove(path)
                return clean_text(res.output.choices[0].message.content[0]['text'])
            else:
                from dashscope import Generation
                res = Generation.call(model="qwen-max", prompt=prompt)
                return clean_text(res.output.text)
        except Exception as e: return f"Error: {e}"

# ==========================================
# 5. UI 布局
# ==========================================
st.set_page_config(page_title="狮子营销大脑", layout="wide", page_icon="🦁")

if 'results' not in st.session_state: st.session_state.results = []

with st.sidebar:
    st.header("1. 配置")
    eng_choice = st.radio("文案引擎", ("Google Gemini", "阿里通义", "智谱清言 (GLM)"))
    if "Google" in eng_choice:
        eng_type, mod_list, cur_key = "google", ["gemini-1.5-flash", "gemini-1.5-pro"], GOOGLE_API_KEY
    elif "阿里" in eng_choice:
        eng_type, mod_list, cur_key = "ali", ["qwen-vl-max", "qwen-max"], ALI_API_KEY
    else:
        eng_type, mod_list, cur_key = "zhipu", ["glm-4v", "glm-4-flash"], ZHIPU_API_KEY
    sel_mod = st.selectbox("选择模型", mod_list)
    st.divider()
    biz_sel = st.radio("模式", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"))
    cur_biz = "logistics" if "VastLog" in biz_sel else "house"
    cur_info = BIZ_CONFIG[cur_biz]
    platform = st.selectbox("发布平台", ["Facebook", "LinkedIn", "YouTube", "TikTok"])

st.title(f"🦁 {cur_info['name']} 数字化中心")

tab1, tab2, tab3 = st.tabs(["✍️ 智能文案", "🎨 封面工厂", "🌍 SEO/GEO 深度优化"])

# --- Tab 1 ---
with tab1:
    c1, c2 = st.columns(2)
    u_files = c1.file_uploader("📂 上传素材", accept_multiple_files=True)
    draft = c2.text_area("📝 描述 (选填)", placeholder="AI 自动分析图片写文案...")
    b1, b2 = st.columns(2)
    btn_img = b1.button("🖼️ 仅处理图片 (识图命名)", use_container_width=True)
    btn_all = b2.button("🚀 全套处理 (识图+写文案)", type="primary", use_container_width=True)
    
    if (btn_img or btn_all) and u_files:
        st.session_state.results = []
        link = generate_utm(cur_info['website'], platform, cur_biz)
        for f in u_files:
            img = Image.open(f)
            # 识图命名
            raw_name = run_text_engine(eng_type, img, get_prompt(cur_info, platform, "", "", "name"), cur_key, sel_mod)
            clean_name = re.sub(r'[^a-z0-9\-]', '', raw_name.lower().replace(" ", "-").replace(".webp", ""))
            if len(clean_name) < 3: clean_name = f"{cur_info['name'].lower()}-{uuid.uuid4().hex[:5]}"
            res_name = f"{clean_name}.webp"
            # 文案
            text = run_text_engine(eng_type, img, get_prompt(cur_info, platform, draft, link, "content"), cur_key, sel_mod) if btn_all else ""
            st.session_state.results.append({"img": img, "name": res_name, "data": convert_image(img), "text": text})

    for res in st.session_state.results:
        l, r = st.columns([1, 2])
        l.image(res['img'], use_container_width=True)
        r.code(res['name'])
        if res['text']: r.text_area("文案", res['text'], height=150)
        r.download_button(f"下载 {res['name']}", res['data'], res['name'])

# --- Tab 2 ---
with tab2:
    st.subheader("🛠️ 封面工厂")
    u_c = st.file_uploader("上传背景", type=['jpg', 'png'])
    if u_c:
        bg = Image.open(u_c)
        ctrl, prev = st.columns([1, 2])
        with ctrl:
            t1 = st.text_input("标题 1", "PREMIUM QUALITY")
            c1, s1 = st.columns(2); col1 = c1.color_picker("颜色 1", "#FFFFFF"); siz1 = s1.slider("大小 1", 0.05, 0.3, 0.1)
            t2 = st.text_input("标题 2", "EXPANDABLE HOUSE")
            c2, s2 = st.columns(2); col2 = c2.color_picker("颜色 2", "#FFDD00"); siz2 = s2.slider("大小 2", 0.05, 0.3, 0.15)
            t3 = st.text_input("标题 3", "FACTORY DIRECT")
            c3, s3 = st.columns(2); col3 = c3.color_picker("颜色 3", "#FF0000"); siz3 = s3.slider("大小 3", 0.05, 0.3, 0.1)
        conf = [{'text':t1,'color':col1,'size':siz1},{'text':t2,'color':col2,'size':siz2},{'text':t3,'color':col3,'size':siz3}]
        out = apply_youtube_style(bg.copy(), conf)
        prev.image(out, use_container_width=True)
        buf = io.BytesIO(); out.save(buf, format="PNG")
        prev.download_button("⬇️ 下载封面", buf.getvalue(), "cover.png", type="primary")

# --- Tab 3 ---
with tab3:
    st.subheader("🌍 GEO 深度优化 (中译英 + HTML + Schema)")
    raw_tx = st.text_area("输入发货记录或中文草稿", height=250)
    geo_img = st.file_uploader("上传对应实拍图 (可选，AI会参考图片内容)", type=['jpg', 'png', 'webp'])
    if st.button("✨ 执行深度优化", type="primary") and raw_tx:
        with st.spinner("AI 专家正在处理..."):
            res = run_text_engine(eng_type, Image.open(geo_img) if geo_image else None, get_prompt(cur_info, "", raw_tx, "", "geo"), cur_key, sel_mod)
            st.markdown("### 💎 优化结果")
            pv, sc = st.columns(2)
            pv.markdown("#### 预览"); pv.components.v1.html(res, height=500, scrolling=True)
            sc.markdown("#### HTML 源代码"); sc.code(res, language="html")
