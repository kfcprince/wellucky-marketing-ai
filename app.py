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
# 3. 图像处理 (支持 3 标题独立参数)
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
# 4. AI 引擎
# ==========================================
def generate_ai_cover(prompt, ratio, api_key):
    dashscope.api_key = api_key
    size = "1280*720" if ratio == "16:9" else "720*1280"
    try:
        rsp = ImageSynthesis.call(model=ImageSynthesis.Models.wanx_v1, prompt=f"Professional logistics photography, 4k, {prompt}", n=1, size=size)
        if rsp.status_code == HTTPStatus.OK:
            return Image.open(io.BytesIO(requests.get(rsp.output.results[0].url).content))
    except: return None

def get_prompt(info, platform, user_draft, link, task_type):
    """ AI 指令中心：增加识图命名逻辑 """
    contact = f"Web: {info['website']}, WhatsApp: {info['phone']}"
    
    # 任务 A: 社交媒体文案
    if task_type == "content":
        return f"Role: Social Media Manager for {info['full_name']}. Platform: {platform}. Draft: {user_draft}. Link: {link}. Contact: {contact}. Rules: Professional, Max 2 emojis, NO markdown."
    
    # 任务 B: 深度 SEO/GEO 专家 (新增 HTML 强制排版逻辑)
    elif task_type == "geo":
        return f"""
        Role: Senior Digital Marketing & HTML Specialist.
        Business Focus: {info['full_name']} ({info['context']})
        Task: Translate or Refine the content into authoritative, professional English.

        Strict Requirements:
        1. **EEAT & SEO**: Use industry-specific terms to enhance Expertise and Trustworthiness.
        2. **HTML Layout**: Output the content in ONE single <div> block with:
           - Container: max-width 900px, font-family Arial, line-height 1.6.
           - Headings: Use <h2> with a blue left border (5px solid #0056b3) and 15px padding-left.
           - Images: If images are mentioned, wrap them in: <img src='IMAGE_URL' alt='[Describe the image with SEO keywords here]' style='width:100%; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.1); margin-bottom:15px;'>.
           - Structure: Use <p> for paragraphs and <ul>/<li> for features/benefits.
        3. **Schema**: At the very end, provide a JSON-LD FAQ Schema code block separately.

        Content to process:
        {user_draft}
        """
    
    # 任务 C: 图片 SEO 命名 (修改这里)
    else:
        return f"""
        Task: Describe this image in 3-5 keywords for a Google SEO filename. 
        Rules:
        1. Only output keywords separated by hyphens.
        2. Must include brand '{info['name'].lower()}'.
        3. No spaces, no capital letters, no file extension.
        Example: welluckyhouse-solar-panel-expandable-home
    
def run_text_engine(engine, image_obj_or_path, prompt, api_key, model):
    if engine == "zhipu":
        client = ZhipuAI(api_key=api_key)
        res = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return clean_text(res.choices[0].message.content)
    elif engine == "google":
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        img = image_obj_or_path if not isinstance(image_obj_or_path, str) else Image.open(image_obj_or_path)
        res = m.generate_content([prompt, img] if img else [prompt])
        return clean_text(res.text)
    else:
        dashscope.api_key = api_key
        if image_obj_or_path:
            path = f"temp_{uuid.uuid4().hex}.png"
            (image_obj_or_path if not isinstance(image_obj_or_path, str) else Image.open(image_obj_or_path)).save(path)
            file_url = f"file://{os.path.abspath(path).replace('\\', '/')}"
            msgs = [{"role": "user", "content": [{"image": file_url}, {"text": prompt}]}]
            res = MultiModalConversation.call(model=model, messages=msgs)
            os.remove(path)
            return clean_text(res.output.choices[0].message.content[0]['text']) if res.status_code == HTTPStatus.OK else "Error"
        else:
            from dashscope import Generation
            res = Generation.call(model="qwen-max", prompt=prompt)
            return clean_text(res.output.text)

# ==========================================
# 5. UI 布局 (全功能回归)
# ==========================================
st.set_page_config(page_title="狮子营销大脑", layout="wide", page_icon="🦁")

if 'results' not in st.session_state: st.session_state.results = []
if 'edited_cover' not in st.session_state: st.session_state.edited_cover = None

with st.sidebar:
    st.header("1. 配置")
    engine_choice = st.radio("文案引擎", ("Google Gemini", "阿里通义", "智谱清言 (GLM)"))
    if "Google" in engine_choice:
        eng_type, mod_list, cur_key = "google", ["gemini-1.5-flash", "gemini-1.5-pro"], GOOGLE_API_KEY
    elif "阿里" in engine_choice:
        eng_type, mod_list, cur_key = "ali", ["qwen-vl-max", "qwen-max"], ALI_API_KEY
    else:
        eng_type, mod_list, cur_key = "zhipu", ["glm-4v", "glm-4-plus", "glm-4-flash"], ZHIPU_API_KEY
    sel_mod = st.selectbox("选择模型", mod_list)
    st.divider()
    biz_sel = st.radio("模式", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"))
    cur_biz = "logistics" if "VastLog" in biz_sel else "house"
    cur_info = BIZ_CONFIG[cur_biz]
    platform = st.selectbox("发布平台", ["Facebook", "LinkedIn", "YouTube", "TikTok"])

st.title(f"🦁 {cur_info['name']} 数字化中心")

tab1, tab2, tab3 = st.tabs(["✍️ 智能文案", "🎨 封面工厂", "🌍 SEO/GEO 深度优化"])

# --- Tab 1: 回归“仅处理图片”和“全套处理” ---
with tab1:
    c1, c2 = st.columns(2)
    u_files = c1.file_uploader("📂 上传素材", accept_multiple_files=True)
    draft = c2.text_area("📝 描述 (选填)", placeholder="AI 自动写文案...")
    
    b1, b2 = st.columns(2)
    btn_img = b1.button("🖼️ 仅处理图片 (快)", use_container_width=True)
    btn_all = b2.button("🚀 全套处理 (写文案)", type="primary", use_container_width=True)
    
    if (btn_img or btn_all) and u_files:
        st.session_state.results = []
        link = generate_utm(cur_info['website'], platform, cur_biz)
        import re # 导入正则用于清理文件名
        
        for f in u_files:
            img = Image.open(f)
            # 1. 获取 AI 生成的 SEO 文件名
            p_name = get_prompt(cur_info, platform, "", "", "name")
            ai_name = run_text_engine(eng_type, img, p_name, cur_key, sel_mod)
            
            # 2. 清洗文件名逻辑
            if not ai_name or "Error" in ai_name or len(ai_name) > 100:
                clean_name = f"{cur_info['name'].lower()}-{uuid.uuid4().hex[:5]}"
            else:
                # 变成小写、空格转连字符、去掉非法字符
                clean_name = ai_name.lower().strip()
                clean_name = clean_name.replace(" ", "-").replace("_", "-").replace(".webp", "")
                clean_name = re.sub(r'[^a-z0-9\-]', '', clean_name) # 只保留字母数字和连字符
            
            res_name = f"{clean_name}.webp"
            
            # 3. 生成文案 (如果是全套处理)
            text = ""
            if btn_all:
                p_text = get_prompt(cur_info, platform, draft, link, "content")
                text = run_text_engine(eng_type, img, p_text, cur_key, sel_mod)
            
            st.session_state.results.append({
                "img": img, 
                "name": res_name, 
                "data": convert_image(img), 
                "text": text
            })
        st.success("图片 SEO 优化完成！")

# 新的清理逻辑：确保文件名是干净的 SEO 格式
if not name or "Error" in name or len(name) > 100:
    # 备用方案：如果 AI 抽风，才使用随机数
    name = f"{cur_info['name'].lower()}-{uuid.uuid4().hex[:5]}"
else:
    # 清理 AI 可能返回的标点、空格或换行
    name = name.lower().replace(" ", "-").replace("_", "-").replace(".webp", "").strip()
    # 过滤掉非法字符
    import re
    name = re.sub(r'[^a-z0-9\-]', '', name)

# 最终拼接后缀
res_name = f"{name}.webp"
            name = f"{cur_info['name'].lower()}-{uuid.uuid4().hex[:5]}.webp" if not name or len(name)>50 else name+".webp"
            text = run_text_engine(eng_type, img, get_prompt(cur_info, platform, draft, link, "content"), cur_key, sel_mod) if btn_all else ""
            st.session_state.results.append({"img": img, "name": name, "data": convert_image(img), "text": text})
    
    if st.session_state.results:
        for res in st.session_state.results:
            l, r = st.columns([1, 2])
            l.image(res['img'], use_container_width=True)
            r.code(res['name'])
            if res['text']: r.text_area("文案", res['text'], height=150)
            r.download_button(f"下载 {res['name']}", res['data'], res['name'])

# --- Tab 2: 回归 3 标题控制 + AI 画图 ---
with tab2:
    st.subheader("🛠️ YouTube 封面工厂")
    mode = st.radio("来源", ("📤 上传背景", "🎨 AI 画图"), horizontal=True)
    t_img = None
    if "上传" in mode:
        u_c = st.file_uploader("上传图", type=['jpg','png'])
        if u_c: t_img = Image.open(u_c)
    else:
        c_p, c_r = st.columns([3, 1])
        prompt = c_p.text_input("画面描述")
        ratio = c_r.selectbox("比例", ["16:9", "9:16"])
        if st.button("✨ 开始画图"):
            with st.spinner("AI 绘画中..."):
                t_img = generate_ai_cover(prompt, ratio, ALI_API_KEY)
                st.session_state.edited_cover = t_img
    
    if st.session_state.edited_cover or t_img:
        work_img = st.session_state.edited_cover if st.session_state.edited_cover else t_img
        ctrl, prev = st.columns([1, 2])
        with ctrl:
            st.markdown("##### 标题 1")
            v1 = st.text_input("内容 1", "TOP SELLING")
            c1, s1 = st.columns(2)
            col1 = c1.color_picker("颜色 1", "#FFFFFF")
            siz1 = s1.slider("大小 1", 0.05, 0.4, 0.1)
            st.markdown("##### 标题 2")
            v2 = st.text_input("内容 2", "CONTAINER HOUSE")
            c2, s2 = st.columns(2)
            col2 = c2.color_picker("颜色 2", "#FFDD00")
            siz2 = s2.slider("大小 2", 0.05, 0.4, 0.15)
            st.markdown("##### 标题 3")
            v3 = st.text_input("内容 3", "FACTORY PRICE")
            c3, s3 = st.columns(2)
            col3 = c3.color_picker("颜色 3", "#FF0000")
            siz3 = s3.slider("大小 3", 0.05, 0.4, 0.1)
        
        configs = [{'text':v1,'color':col1,'size':siz1},{'text':v2,'color':col2,'size':siz2},{'text':v3,'color':col3,'size':siz3}]
        out_img = apply_youtube_style(work_img.copy(), configs)
        prev.image(out_img, use_container_width=True)
        b = io.BytesIO()
        out_img.save(b, format="PNG")
        prev.download_button("⬇️ 下载封面", b.getvalue(), "cover.png", type="primary")

# --- Tab 3: SEO/GEO 深度优化 ---
with tab3:
    st.subheader("🌍 内容深度加工 (中译英 + EEAT + HTML 排版)")
    
    col_input, col_img_upload = st.columns([2, 1])
    with col_input:
        raw_text = st.text_area("粘贴你的中文发货实录或英文草稿", height=250, key="geo_input")
    with col_img_upload:
        geo_image = st.file_uploader("上传对应图片 (AI会根据图片细节优化文案)", type=['jpg','png','webp'], key="geo_img")

    if st.button("✨ 执行深度优化并生成 HTML", type="primary"):
        if raw_text:
            geo_prompt = get_prompt(cur_info, "", raw_text, "", "geo")
            with st.spinner("专家正在排版中..."):
                # 执行 AI 生成
                refined_output = run_text_engine(eng_type, geo_image, geo_prompt, cur_key, sel_mod)
                
                st.markdown("---")
                col_preview, col_source = st.columns(2)
                
                with col_preview:
                    st.markdown("### 👁️ 效果预览 (Preview)")
                    # 在网页中直接渲染 HTML
                    st.components.v1.html(refined_output, height=600, scrolling=True)
                
                with col_source:
                    st.markdown("### 💻 HTML 源代码")
                    st.code(refined_output, language="html")
                    st.caption("提示：点击右上角复制按钮，粘贴到网站后台的 HTML/源码模式下。")
        else:
            st.warning("请先输入内容")





