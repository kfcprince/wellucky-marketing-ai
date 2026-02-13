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

# ==========================================
# 0. 配置区 (已脱敏：从 Streamlit Secrets 读取)
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    ALI_API_KEY = st.secrets["ALI_API_KEY"]
    ZHIPU_API_KEY = st.secrets["ZHIPU_API_KEY"]
except:
    GOOGLE_API_KEY = ""
    ALI_API_KEY = ""
    ZHIPU_API_KEY = ""

BUFFER_LOGISTICS_URL = "https://publish.buffer.com/profile/你的物流ID"
BUFFER_HOUSE_URL = "https://publish.buffer.com/profile/你的房屋ID"

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
        "buffer_url": BUFFER_LOGISTICS_URL
    },
    "house": {
        "name": "WelluckyHouse",
        "full_name": "Wellucky Container House",
        "website": "www.welluckyhouse.com",
        "email": "info@welluckyhouse.com",
        "phone": "+86 18615329580",
        "keywords": "expandable container house, folding house, apple cabin",
        "context": "We manufacture high-quality expandable container houses.",
        "buffer_url": BUFFER_HOUSE_URL
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
# 3. 图像处理 (实时预览核心)
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
    total_block_height = 0
    for cfg in text_configs:
        if not cfg['text']: continue
        font_size = int(H * cfg['size'])
        font = load_font_safe(font_size)
        bbox = draw.textbbox((0, 0), cfg['text'], font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        line_h = h * 1.2
        total_block_height += line_h
        lines_to_draw.append({"text": cfg['text'], "font": font, "color": cfg['color'], "w": w, "line_h": line_h, "f_size": font_size})
    
    current_y = H - total_block_height - (H * 0.05)
    for line in lines_to_draw:
        x = (W - line['w']) / 2
        stroke = int(line['f_size'] * 0.08)
        draw.text((x, current_y), line['text'], font=line['font'], fill=line['color'], stroke_width=stroke, stroke_fill="black")
        current_y += line['line_h']
    return image

# ==========================================
# 4. AI 引擎 (整合 SEO/GEO 逻辑)
# ==========================================
def get_prompt(info, platform, user_draft, link, task_type):
    contact = f"Web: {info['website']}, WhatsApp: {info['phone']}"
    if task_type == "content":
        return f"Role: Social Media Manager for {info['full_name']}. Platform: {platform}. Draft: {user_draft}. Link: {link}. Contact: {contact}. Rules: Professional, Max 2 emojis, NO markdown."
    elif task_type == "geo":
        return f"Role: Senior SEO & GEO Specialist. Task: Translate or Refine into professional, authoritative English. Target: Enhance EEAT (Expertise, Authoritativeness, Trustworthiness). Provide content and a JSON-LD FAQ Schema. Content: {user_draft}"
    else:
        return f"Task: Google SEO filename for {info['keywords']}. Rule: Lowercase, hyphens only, include brand '{info['name'].lower()}'."

def run_text_engine(engine, image_obj_or_path, prompt, api_key, model):
    if engine == "zhipu":
        try:
            client = ZhipuAI(api_key=api_key)
            res = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            return clean_text(res.choices[0].message.content)
        except Exception as e: return f"智谱错误: {e}"
    elif engine == "google":
        try:
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel(model)
            img = image_obj_or_path if not isinstance(image_obj_or_path, str) else Image.open(image_obj_or_path)
            res = m.generate_content([prompt, img] if img else [prompt])
            return clean_text(res.text)
        except Exception as e: return f"Google错误: {e}"
    else:
        try:
            dashscope.api_key = api_key
            if not os.path.exists("temp"): os.makedirs("temp")
            path = os.path.join("temp", f"{uuid.uuid4().hex}.png")
            if image_obj_or_path:
                (image_obj_or_path if not isinstance(image_obj_or_path, str) else Image.open(image_obj_or_path)).save(path)
                file_url = f"file://{os.path.abspath(path).replace('\\', '/')}"
                msgs = [{"role": "user", "content": [{"image": file_url}, {"text": prompt}]}]
                res = MultiModalConversation.call(model=model, messages=msgs)
                os.remove(path)
                return clean_text(res.output.choices[0].message.content[0]['text']) if res.status_code == HTTPStatus.OK else res.message
            else:
                # 纯文本处理
                from dashscope import Generation
                res = Generation.call(model="qwen-max", prompt=prompt)
                return clean_text(res.output.text) if res.status_code == HTTPStatus.OK else res.message
        except Exception as e: return f"阿里错误: {e}"

# ==========================================
# 5. 页面布局 V20.0 (支持 GEO 专家)
# ==========================================
st.set_page_config(page_title="VastLog & Wellucky 营销大脑", layout="wide", page_icon="🦁")

with st.sidebar:
    st.header("1. 配置")
    engine_choice = st.radio("文案引擎", ("Google Gemini", "阿里通义", "智谱清言 (GLM)"), key="eng_radio")
    if "Google" in engine_choice:
        eng_type, mod_list, cur_key = "google", ["gemini-1.5-flash", "gemini-1.5-pro"], GOOGLE_API_KEY
    elif "阿里" in engine_choice:
        eng_type, mod_list, cur_key = "ali", ["qwen-vl-max", "qwen-max"], ALI_API_KEY
    else:
        eng_type, mod_list, cur_key = "zhipu", ["glm-4v", "glm-4-plus", "glm-4-flash"], ZHIPU_API_KEY
    
    sel_mod = st.selectbox("选择模型", mod_list, key="mod_select")
    st.divider()
    biz_sel = st.radio("模式", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"), key="biz_radio")
    cur_biz = "logistics" if "VastLog" in biz_sel else "house"
    cur_info = BIZ_CONFIG[cur_biz]
    platform = st.selectbox("发布平台", ["Facebook", "LinkedIn", "YouTube", "TikTok"])

st.title(f"🦁 {cur_info['name']} 数字化营销中心")

tab1, tab2, tab3 = st.tabs(["✍️ 智能文案", "🎨 封面工厂", "🌍 SEO/GEO 深度优化"])

# --- Tab 1: 智能文案 ---
with tab1:
    c1, c2 = st.columns(2)
    u_files = c1.file_uploader("📂 上传素材", accept_multiple_files=True)
    draft = c2.text_area("📝 描述", placeholder="AI 自动写文案...")
    if st.button("🚀 批量处理", type="primary") and u_files:
        st.session_state.results = []
        link = generate_utm(cur_info['website'], platform, cur_biz)
        for f in u_files:
            img = Image.open(f)
            name = run_text_engine(eng_type, img, get_prompt(cur_info, platform, "", "", "name"), cur_key, sel_mod)
            name = f"{cur_info['name'].lower()}-{uuid.uuid4().hex[:5]}.webp" if not name or len(name)>50 else name+".webp"
            text = run_text_engine(eng_type, img, get_prompt(cur_info, platform, draft, link, "content"), cur_key, sel_mod)
            st.session_state.results.append({"img": img, "name": name, "data": convert_image(img), "text": text})
        st.success("处理完成！")
    
    if 'results' in st.session_state:
        for res in st.session_state.results:
            col_l, col_r = st.columns([1, 2])
            col_l.image(res['img'], use_container_width=True)
            col_r.code(res['name'])
            col_r.text_area("文案", res['text'], height=150)
            col_r.download_button("下载图片", res['data'], res['name'])

# --- Tab 2: 封面工厂 ---
with tab2:
    st.subheader("🛠️ YouTube 视频封面制作")
    u_cover = st.file_uploader("上传背景图", type=['jpg', 'png'])
    if u_cover:
        t_img = Image.open(u_cover)
        col_c, col_p = st.columns([1, 2])
        txt1 = col_c.text_input("标题 1", "TOP QUALITY")
        txt2 = col_c.text_input("标题 2", "CONTAINER HOUSE")
        color = col_c.color_picker("文字颜色", "#FFDD00")
        conf = [{'text': txt1, 'color': color, 'size': 0.15}, {'text': txt2, 'color': '#FFFFFF', 'size': 0.1}]
        prev_img = apply_youtube_style(t_img.copy(), conf)
        col_p.image(prev_img, use_container_width=True)
        buf = io.BytesIO()
        prev_img.save(buf, format="PNG")
        col_p.download_button("保存封面", buf.getvalue(), "cover.png")

# --- Tab 3: SEO/GEO 深度优化 (新功能!) ---
with tab3:
    st.subheader("🌍 内容深度加工 (中译英 + EEAT + Schema)")
    raw_text = st.text_area("粘贴你的中文草稿或原始英文", height=250, placeholder="例如：我们今天发货了，包装非常专业...")
    
    if st.button("✨ 执行深度优化", type="primary"):
        if raw_text:
            geo_prompt = get_prompt(cur_info, "", raw_text, "", "geo")
            with st.spinner("专家正在润色并生成 Schema..."):
                # 这里不传图片，只传文本
                refined_content = run_text_engine(eng_type, None, geo_prompt, cur_key, sel_mod)
                st.markdown("### 💎 优化后的权威文案")
                st.info("此文案已根据 EEAT 准则润色，适合直接发布在官网。")
                st.write(refined_content)
        else:
            st.warning("请先输入内容")
