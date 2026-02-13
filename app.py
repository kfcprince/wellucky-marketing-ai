import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import MultiModalConversation, ImageSynthesis 
from zhipuai import ZhipuAI
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io
import base64
import zipfile
import time
import os
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
    "logistics": {"name": "VastLog", "website": "www.vastlog.com"},
    "house": {"name": "WelluckyHouse", "website": "www.welluckyhouse.com"}
}

# ==========================================
# 2. 图像转换工具 (核心修复：Base64转换)
# ==========================================
def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def convert_to_webp(image):
    buf = io.BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(buf, format='WEBP', quality=80)
    return buf.getvalue()

# ==========================================
# 3. 增强型命名清洗逻辑
# ==========================================
def get_clean_seo_name(ai_res, brand):
    if not ai_res or "Error" in ai_res:
        return f"{brand.lower()}-product-{uuid.uuid4().hex[:4]}"
    
    # 清理AI回复中的无关文字
    name = ai_res.lower()
    name = re.sub(r'(\.jpg|\.png|\.webp|file name|seo name|is:|here:)', '', name)
    name = re.sub(r'[^a-z0-9]', ' ', name) # 标点符号变空格
    
    words = [w for w in name.split() if len(w) > 1]
    # 强制加上品牌名
    brand_low = brand.lower()
    if brand_low not in words:
        words.insert(0, brand_low)
    
    return "-".join(words[:6])

# ==========================================
# 4. 万能识图引擎 (修复各家API调用姿势)
# ==========================================
def run_ai_vision(engine, img, prompt, key, model):
    if not key: return "Error: No API Key"
    
    # --- Google Gemini 识图修复 ---
    if engine == "google":
        try:
            genai.configure(api_key=key)
            m = genai.GenerativeModel(model)
            # 这里的 img 必须直接传 PIL 对象
            res = m.generate_content([prompt, img])
            return res.text if res.text else "Error: Empty Response"
        except Exception as e: return f"Error: {str(e)}"
    
    # --- 阿里通义 识图修复 ---
    elif engine == "ali":
        try:
            dashscope.api_key = key
            tmp_p = f"v_{uuid.uuid4().hex}.png"
            img.save(tmp_p)
            abs_p = os.path.abspath(tmp_p).replace('\\', '/')
            url = f"file://{abs_p}"
            res = MultiModalConversation.call(model=model, messages=[{"role":"user","content":[{"image":url},{"text":prompt}]}])
            if os.path.exists(tmp_p): os.remove(tmp_p)
            if res.status_code == 200:
                return res.output.choices[0].message.content[0]['text']
            return f"Error: {res.message}"
        except Exception as e: return f"Error: {str(e)}"

    # --- 智谱清言 识图修复 ---
    else:
        try:
            client = ZhipuAI(api_key=key)
            img_b64 = pil_to_base64(img)
            res = client.chat.completions.create(
                model="glm-4v",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_b64}}]}]
            )
            return res.choices[0].message.content
        except Exception as e: return f"Error: {str(e)}"

# ==========================================
# 5. UI 界面
# ==========================================
st.set_page_config(page_title="狮子营销助手", layout="wide")

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

st.header(f"🦁 {cinfo['name']} 数字化助手")
tab1, tab2, tab3 = st.tabs(["✍️ 智能文案", "🎨 封面工厂", "🌍 GEO 专家"])

with tab1:
    c1, c2 = st.columns(2)
    files = c1.file_uploader("📂 上传图片", accept_multiple_files=True, key="u_tab1")
    draft = c2.text_area("📝 文案重点 (选填)", key="d_tab1")
    
    b1, b2 = st.columns(2)
    process_img = b1.button("🖼️ 仅识图起名 (WebP转换)", use_container_width=True)
    process_all = b2.button("🚀 全套处理 (写文案)", type="primary", use_container_width=True)

    if (process_img or process_all) and files:
        st.session_state.results = []
        for f in files:
            img = Image.open(f)
            # --- 核心改进：极其直白的提示词 ---
            prompt_name = "What is this? Provide 3 English keywords separated by spaces. Example: solar panel house."
            
            with st.spinner(f"AI正在分析图片: {f.name}"):
                raw_ai_res = run_ai_vision(etype, img, prompt_name, ekey, sel_mod)
                
                # 如果AI返回了错误，直接在界面显示，不再偷偷用随机数
                if "Error" in raw_ai_res:
                    st.error(f"识图失败: {raw_ai_res}")
                    clean_name = f"{cinfo['name'].lower()}-fallback-{uuid.uuid4().hex[:4]}"
                else:
                    clean_name = get_clean_seo_name(raw_ai_res, cinfo['name'])
                
                fname = f"{clean_name}.webp"
                text = ""
                if process_all:
                    prompt_text = f"Write a social media post for {cinfo['name']}. Based on this image. Professional tone."
                    text = run_ai_vision(etype, img, prompt_text, ekey, sel_mod)
                
                st.session_state.results.append({"img": img, "name": fname, "data": convert_to_webp(img), "text": text})

    for i, res in enumerate(st.session_state.results):
        l, r = st.columns([1, 2])
        l.image(res['img'], use_container_width=True)
        r.code(res['name'])
        if res['text']: r.text_area("文案", res['text'], height=150, key=f"t_{i}")
        r.download_button(f"下载 WebP 图片", res['data'], res['name'], key=f"dl_{i}")

# Tab 2, Tab 3 保持简洁，代码略...
