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
# 0. 初始化
# ==========================================
if 'results' not in st.session_state:
    st.session_state.results = []

# ==========================================
# 1. 核心配置
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    ALI_API_KEY = st.secrets["ALI_API_KEY"]
    ZHIPU_API_KEY = st.secrets["ZHIPU_API_KEY"]
except:
    GOOGLE_API_KEY = ALI_API_KEY = ZHIPU_API_KEY = ""

BIZ_CONFIG = {
    "logistics": {"name": "VastLog", "full_name": "VastLog Logistics", "website": "www.vastlog.com", "phone": "+86 13780685000"},
    "house": {"name": "WelluckyHouse", "full_name": "Wellucky Container House", "website": "www.welluckyhouse.com", "phone": "+86 18615329580"}
}

# ==========================================
# 2. 核心清洗逻辑 (精修版)
# ==========================================
def get_clean_seo_name(ai_res, brand):
    """确保输出：brand-keyword-keyword 格式"""
    if not ai_res or "Error" in ai_res:
        return f"{brand.lower()}-{uuid.uuid4().hex[:5]}"
    
    # 移除文件后缀和废话
    name = ai_res.lower()
    name = re.sub(r'\.(jpg|jpeg|png|webp|gif|bmp)', '', name)
    name = re.sub(r'[^a-z0-9]', ' ', name) # 标点变空格
    
    words = [w for w in name.split() if len(w) > 1 and w not in ['image', 'photo', 'picture', 'here', 'is']]
    
    # 确保品牌词在最前面且不重复
    brand_low = brand.lower()
    if brand_low in words: words.remove(brand_low)
    words.insert(0, brand_low)
    
    return "-".join(words[:6]) # 最多保留6个词

# ==========================================
# 3. 工具函数
# ==========================================
def convert_image(image):
    buf = io.BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(buf, format='WEBP', quality=80)
    return buf.getvalue()

def run_ai_vision(engine, img, prompt, key, model):
    """专用识图引擎"""
    if engine == "google":
        try:
            genai.configure(api_key=key)
            # 强制使用最新稳定的模型路径
            m_name = "gemini-1.5-flash" if "flash" in model else "gemini-1.5-pro"
            m = genai.GenerativeModel(m_name)
            res = m.generate_content([prompt, img])
            return res.text
        except: return "Error"
    elif engine == "zhipu":
        try:
            client = ZhipuAI(api_key=key)
            # 识图必须用 glm-4v
            res = client.chat.completions.create(model="glm-4v", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}])
            return res.choices[0].message.content
        except: return "Error"
    else:
        try:
            dashscope.api_key = key
            p = f"t_{uuid.uuid4().hex}.png"; img.save(p)
            url = f"file://{os.path.abspath(p).replace('\\', '/')}"
            res = MultiModalConversation.call(model=model, messages=[{"role":"user","content":[{"image":url},{"text":prompt}]}])
            if os.path.exists(p): os.remove(p)
            return res.output.choices[0].message.content[0]['text']
        except: return "Error"

# ==========================================
# 4. UI 界面
# ==========================================
st.set_page_config(page_title="狮子营销大脑", layout="wide")

with st.sidebar:
    st.title("⚙️ 配置")
    engine = st.radio("文案引擎", ("Google Gemini", "阿里通义", "智谱清言"))
    if "Google" in engine:
        etype, mlist, ekey = "google", ["gemini-1.5-flash", "gemini-1.5-pro"], GOOGLE_API_KEY
    elif "阿里" in engine:
        etype, mlist, ekey = "ali", ["qwen-vl-max", "qwen-vl-plus"], ALI_API_KEY
    else:
        etype, mlist, ekey = "zhipu", ["glm-4v"], ZHIPU_API_KEY
    
    sel_mod = st.selectbox("选择模型", mlist)
    biz = st.radio("业务模式", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"))
    cbiz = "logistics" if "VastLog" in biz else "house"
    cinfo = BIZ_CONFIG[cbiz]
    plat = st.selectbox("平台", ["Facebook", "LinkedIn", "YouTube"])

st.header(f"🦁 {cinfo['name']} 数字化助手")
tab1, tab2, tab3 = st.tabs(["✍️ 智能文案", "🎨 封面工厂", "🌍 GEO 专家"])

with tab1:
    c1, c2 = st.columns(2)
    files = c1.file_uploader("📂 上传图片", accept_multiple_files=True, key="u_tab1")
    draft = c2.text_area("📝 描述", placeholder="想要AI重点写的内容...", key="d_tab1")
    
    b1, b2 = st.columns(2)
    process_img = b1.button("🖼️ 仅识图起名 (WebP转换)", use_container_width=True)
    process_all = b2.button("🚀 全套处理 (写文案)", type="primary", use_container_width=True)

    if (process_img or process_all) and files:
        st.session_state.results = []
        for f in files:
            img = Image.open(f)
            # 强力识图提示词
            prompt_name = "Look at this image. What is the main product? Provide 3 specific English keywords. No punctuation, no sentences. Just keywords."
            raw_ai_name = run_ai_vision(etype, img, prompt_name, ekey, sel_mod)
            
            clean_name = get_clean_seo_name(raw_ai_name, cinfo['name'])
            fname = f"{clean_name}.webp"
            
            text = ""
            if process_all:
                prompt_text = f"Write a professional {plat} post for {cinfo['full_name']}. Content about: {draft if draft else 'this product'}. Include {cinfo['website']}. Professional tone."
                text = run_ai_vision(etype, img, prompt_text, ekey, sel_mod)
            
            st.session_state.results.append({"img": img, "name": fname, "data": convert_image(img), "text": text})

    for i, res in enumerate(st.session_state.results):
        l, r = st.columns([1, 2])
        l.image(res['img'], use_container_width=True)
        r.code(res['name'])
        if res['text']: r.text_area("文案", res['text'], height=150, key=f"t_{i}")
        r.download_button(f"下载 WebP 图片", res['data'], res['name'], key=f"dl_{i}")

# ... (Tab 2 和 Tab 3 保持逻辑简洁，重点修复 Tab 1)
