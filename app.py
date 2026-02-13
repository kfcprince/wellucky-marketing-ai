from zhipuai import ZhipuAI
import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import MultiModalConversation, ImageSynthesis 
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

# 网页版部署时，会自动从后台设置中读取这些 Key，不再暴露在代码里
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    ALI_API_KEY = st.secrets["ALI_API_KEY"]
    ZHIPU_API_KEY = st.secrets["ZHIPU_API_KEY"]
except:
    # 如果本地运行没有配置 secrets，则留空提醒
    GOOGLE_API_KEY = ""
    ALI_API_KEY = ""
    ZHIPU_API_KEY = ""

# Buffer 配置也可以存入 Secrets，或者保持现状（因为不涉及敏感扣费）
BUFFER_LOGISTICS_URL = "https://publish.buffer.com/profile/你的物流ID"
BUFFER_HOUSE_URL = "https://publish.buffer.com/profile/你的房屋ID"

# ... (后面代码保持不变，仅需将开头的 Key 赋值部分改为上面这样)

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
    cleaned = text.replace("**", "").replace("##", "").replace("###", "")
    return cleaned.strip()

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
    except:
        try: return ImageFont.truetype("arialbd.ttf", size)
        except: return ImageFont.load_default()

def apply_youtube_style(image, text_configs):
    # 1. 滤镜增强 (饱和度+对比度)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.4) 
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)

    W, H = image.size
    draw = ImageDraw.Draw(image)
    
    # 2. 计算排版
    total_block_height = 0
    lines_to_draw = []

    for cfg in text_configs:
        text = cfg['text']
        if not text or not text.strip(): continue
        font_size = int(H * cfg['size'])
        font = load_font_safe(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_height = h * 1.2
        total_block_height += line_height
        lines_to_draw.append({
            "text": text, "font": font, "color": cfg['color'],
            "w": w, "h": h, "line_height": line_height, "font_size": font_size
        })

    current_y = H - total_block_height - (H * 0.05) # 底部堆叠

    for line in lines_to_draw:
        x = (W - line['w']) / 2
        stroke_width = int(line['font_size'] * 0.08)
        if stroke_width < 1: stroke_width = 1
        draw.text((x, current_y), line['text'], font=line['font'], 
                  fill=line['color'], stroke_width=stroke_width, stroke_fill="black")
        current_y += line['line_height']

    return image

# ==========================================
# 4. AI 引擎
# ==========================================

def generate_ai_cover(prompt, ratio, api_key):
    dashscope.api_key = api_key
    size = "1280*720" if ratio == "16:9" else "720*1280"
    refined_prompt = f"High quality, photorealistic, 4k, cinematic lighting, western aesthetics, {prompt}"
    try:
        rsp = ImageSynthesis.call(model=ImageSynthesis.Models.wanx_v1, prompt=refined_prompt, n=1, size=size)
        if rsp.status_code == HTTPStatus.OK:
            img_data = requests.get(rsp.output.results[0].url).content
            return Image.open(io.BytesIO(img_data))
        return None
    except:
        return None

def get_prompt(info, platform, user_draft, link, task_type):
    contact = f"Web: {info['website']}, WhatsApp: {info['phone']}"
    if task_type == "content":
        return f"""
        Role: Social Media Manager for {info['full_name']}.
        Business Focus: {info['context']}
        Platform: {platform}.
        Task: Write a post description.
        Input: {user_draft if user_draft else "General promotion"}
        Link: {link}
        Contact: {contact}
        Rules: Professional Business English. MINIMIZE EMOJIS (Max 1-2). NO Markdown.
        """
    else:
        return f"""
        Task: Google SEO filename.
        Keywords: {info['keywords']}
        Rule: Lowercase, hyphens only, include brand '{info['name'].lower()}', no extension.
        """
# 2. 完整替换这个函数
def run_text_engine(engine, image_obj_or_path, prompt, api_key, model):
    """核心文案生成引擎 - 支持 Google, 阿里, 智谱"""
    
    # === 智谱清言 (Zhipu) 分支 ===
    if engine == "zhipu":
        try:
            client = ZhipuAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return clean_text(response.choices[0].message.content)
        except Exception as e:
            return f"智谱引擎错误: {str(e)}"

    # === Google Gemini 分支 ===
    elif engine == "google":
        try:
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel(model)
            if isinstance(image_obj_or_path, str): 
                img = Image.open(image_obj_or_path)
            else: 
                img = image_obj_or_path
            res = m.generate_content([prompt, img])
            return clean_text(res.text)
        except Exception as e:
            return f"Google引擎错误: {str(e)}"

    # === 阿里通义 (Ali) 分支 ===
    else:
        try:
            dashscope.api_key = api_key
            if not os.path.exists("temp"): os.makedirs("temp")
            safe_name = f"img_{uuid.uuid4().hex[:8]}.png"
            safe_path = os.path.join("temp", safe_name)
            
            if not isinstance(image_obj_or_path, str): 
                image_obj_or_path.save(safe_path)
            else: 
                Image.open(image_obj_or_path).save(safe_path)
                
            abs_path = os.path.abspath(safe_path).replace("\\", "/")
            file_url = f"file://{abs_path}"
            msgs = [{"role": "user", "content": [{"image": file_url}, {"text": prompt}]}]
            res = MultiModalConversation.call(model=model, messages=msgs)
            
            try: os.remove(safe_path)
            except: pass
            
            if res.status_code == HTTPStatus.OK:
                return clean_text(res.output.choices[0].message.content[0]['text'])
            return f"阿里引擎错误: {res.message}"
        except Exception as e:
            return f"系统处理错误: {str(e)}"

# ==========================================
# 5. 页面布局 V18.0 (实时预览版)
# ==========================================

st.set_page_config(page_title="VastLog & Wellucky 旗舰版", layout="wide", page_icon="🦁")

if 'results' not in st.session_state: st.session_state.results = []
if 'edited_cover' not in st.session_state: st.session_state.edited_cover = None

with st.sidebar:
    st.header("1. 配置")
    
    # 引擎选择
    engine_choice = st.radio(
        "文案引擎", 
        ("Google Gemini", "阿里通义", "智谱清言 (GLM)"), 
        key="eng_radio"
    )
    
    # 根据选择动态切换模型列表和 Key
    if engine_choice == "Google Gemini":
        eng_type = "google"
        mod_list = ["gemini-1.5-flash", "gemini-1.5-pro"]
        cur_key = GOOGLE_API_KEY
    elif engine_choice == "阿里通义":
        eng_type = "ali"
        mod_list = ["qwen-vl-max", "qwen-vl-plus"]
        cur_key = ALI_API_KEY
    else: # 智谱清言
        eng_type = "zhipu"
        # glm-4v 是智谱最强的识图模型，glm-4-flash 是速度最快的
        mod_list = ["glm-4v", "glm-4-plus", "glm-4-flash"] 
        cur_key = ZHIPU_API_KEY
    
    # 模型选择框会根据上面的 mod_list 实时变化
    sel_mod = st.selectbox("选择模型", mod_list, key="mod_select")
    
    st.divider()
    st.header("2. 业务")
    biz_sel = st.radio("模式", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"), key="biz_radio")
    cur_biz = "logistics" if "VastLog" in biz_sel else "house"
    cur_info = BIZ_CONFIG[cur_biz]
    
    st.divider()
    platform = st.selectbox("发布平台", ["Facebook", "LinkedIn", "YouTube", "TikTok"], key="plat_select")

st.title(f"🦁 营销助手 - {cur_info['name']}")

if len(cur_key) < 5:
    st.error("⚠️ 请先在代码中填入 API Key！")
    st.stop()

tab1, tab2 = st.tabs(["✍️ 智能文案 & 配图处理", "🎨 YouTube 封面工厂"])

# === Tab 1 ===
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        u_files = st.file_uploader(f"📂 上传素材", accept_multiple_files=True, key="u_files")
    with c2:
        draft = st.text_area("📝 (选填) 描述", height=100, placeholder="留空AI自动写...", key="draft_area")
    
    col_b1, col_b2 = st.columns([1, 1])
    btn_img_only = col_b1.button("🖼️ 仅处理图片 (快)", use_container_width=True, key="btn_img")
    btn_all = col_b2.button("🚀 全套处理 (写文案)", type="primary", use_container_width=True, key="btn_all")
    
    if (btn_img_only or btn_all) and u_files:
        st.session_state.results = []
        bar = st.progress(0)
        link = generate_utm(cur_info['website'], platform, cur_biz)
        for i, f in enumerate(u_files):
            try:
                img = Image.open(f)
                p_name = get_prompt(cur_info, platform, "", "", "name")
                name = run_text_engine(eng_type, img, p_name, cur_key, sel_mod)
                if not name or "Error" in name or len(name) > 50: 
                    name = f"{cur_info['name'].lower()}-{int(time.time())}.webp"
                else: name = name.replace(".webp", "").replace(".", "") + ".webp"
                text = ""
                if btn_all:
                    p_text = get_prompt(cur_info, platform, draft, link, "content")
                    text = run_text_engine(eng_type, img, p_text, cur_key, sel_mod)
                data = convert_image(img)
                st.session_state.results.append({"img": img, "name": name, "data": data, "text": text})
            except Exception as e: st.error(f"出错: {e}")
            bar.progress((i+1)/len(u_files))
            
    if st.session_state.results:
        st.divider()
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for res in st.session_state.results: zf.writestr(res['name'], res['data'])
        st.download_button(f"📦 批量下载", zip_buf.getvalue(), "assets.zip", type="primary")
        st.markdown("---")
        for res in st.session_state.results:
            if not res['text']:
                c_l, c_r = st.columns([1, 4])
                with c_l: st.image(res['img'], use_container_width=True)
                with c_r: 
                    st.code(res['name'], language="text")
                    st.download_button(f"⬇️ 下载", res['data'], res['name'], key=f"d_{res['name']}")
                st.divider()
            else:
                c_l, c_r = st.columns([1, 2])
                with c_l:
                    st.image(res['img'], use_container_width=True)
                    st.code(res['name'], language="text")
                    st.download_button(f"⬇️ 下载图片", res['data'], res['name'], key=f"d_{res['name']}")
                with c_r:
                    st.info(f"📄 {platform} 文案")
                    st.text_area("内容", res['text'], height=250, key=f"t_{res['name']}")
                st.divider()
        if cur_info['buffer_url']: st.link_button("🚀 去 Buffer 发布", cur_info['buffer_url'])

# === Tab 2: 封面工厂 (实时预览版) ===
with tab2:
    st.subheader("🛠️ YouTube 封面工厂")
    
    # 1. 来源选择
    editor_mode = st.radio("来源", ("📤 上传截图", "🎨 AI 画图"), horizontal=True, key="ed_mode")
    target_img = None
    
    if "上传" in editor_mode:
        u_cover = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'], key="u_cover")
        if u_cover: target_img = Image.open(u_cover)
    else:
        c_p, c_r = st.columns([3, 1])
        with c_p: ai_prompt = st.text_input("画面描述", placeholder="例如：集装箱船在海上", key="ai_p")
        with c_r: ai_ratio = st.selectbox("比例", ["16:9", "9:16"], key="ai_r")
        if st.button("✨ AI 画图", key="btn_draw"):
            if not ALI_API_KEY or "填入" in ALI_API_KEY: st.error("需阿里云Key")
            else:
                with st.spinner("AI 正在绘画..."):
                    target_img = generate_ai_cover(ai_prompt, ai_ratio, ALI_API_KEY)
                    if target_img: st.session_state.edited_cover = target_img
    
    if st.session_state.edited_cover and "AI" in editor_mode:
        target_img = st.session_state.edited_cover
        
    # 2. 实时编辑与预览区
    if target_img:
        st.divider()
        
        # 左右分栏：左边调参，右边实时预览
        col_ctrl, col_prev = st.columns([1, 1.5])
        
        with col_ctrl:
            st.markdown("#### ✏️ 编辑参数")
            
            with st.expander("标题 1 (顶部)", expanded=True):
                t1_text = st.text_input("内容", placeholder="BIG SALE", key="t1_txt")
                c1, c2 = st.columns([1, 1])
                t1_color = c1.color_picker("颜色", "#FFFFFF", key="t1_col")
                t1_size = c2.slider("大小", 0.05, 0.5, 0.1, key="t1_siz")

            with st.expander("标题 2 (中间)", expanded=True):
                t2_text = st.text_input("内容", placeholder="50% OFF", key="t2_txt")
                c1, c2 = st.columns([1, 1])
                t2_color = c1.color_picker("颜色", "#FFDD00", key="t2_col")
                t2_size = c2.slider("大小", 0.05, 0.5, 0.25, key="t2_siz")

            with st.expander("标题 3 (底部)", expanded=True):
                t3_text = st.text_input("内容", placeholder="Limited Time", key="t3_txt")
                c1, c2 = st.columns([1, 1])
                t3_color = c1.color_picker("颜色", "#FF0000", key="t3_col")
                t3_size = c2.slider("大小", 0.05, 0.5, 0.1, key="t3_siz")

            st.caption("💡 提示：修改左侧参数，右侧图片会实时更新！")

        with col_prev:
            st.markdown("#### 👁️ 实时预览")
            
            # 实时计算合成图
            # 只要上面的 text_input 或 slider 一变，这里就会重新运行
            configs = [
                {'text': t1_text, 'color': t1_color, 'size': t1_size},
                {'text': t2_text, 'color': t2_color, 'size': t2_size},
                {'text': t3_text, 'color': t3_color, 'size': t3_size},
            ]
            
            # 使用原图的副本进行处理，不破坏原图
            preview_img = apply_youtube_style(target_img.copy(), configs)
            
            st.image(preview_img, use_container_width=True)
            
            # 下载按钮
            buf = io.BytesIO()
            preview_img.save(buf, format="PNG")

            st.download_button("⬇️ 下载这张封面", buf.getvalue(), "cover.png", "image/png", type="primary", use_container_width=True)



