import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import ImageSynthesis, MultiModalConversation, Generation
from zhipuai import ZhipuAI
from PIL import Image, ImageDraw, ImageFont
import io, base64, re, os, requests, uuid, json

# ==========================================
# 0. 全局配置 & 初始化
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V29.3", layout="wide", page_icon="🦁")

# 初始化 Session State
if 'results_tab1' not in st.session_state: st.session_state.results_tab1 = []
if 'generated_bg' not in st.session_state: st.session_state.generated_bg = None
if 'seo_metadata' not in st.session_state: st.session_state.seo_metadata = {}

# 安全读取 Secrets
def get_secret_safe(key_name, default=""):
    try:
        return st.secrets.get(key_name, default)
    except:
        return default

GOOGLE_API_KEY = get_secret_safe("GOOGLE_API_KEY")
ALI_API_KEY = get_secret_safe("ALI_API_KEY")
ZHIPU_API_KEY = get_secret_safe("ZHIPU_API_KEY")

# 业务配置
BIZ_CONFIG = {
    "logistics": {
        "name": "VastLog", "website": "www.vastlog.com", "color": "#FF9900", "type": "LogisticsService",
        "keywords": ["logistics", "shipping", "freight", "cargo", "DDP", "express"]
    },
    "house": {
        "name": "Wellucky", "website": "www.wellucky.com", "color": "#0066CC", "type": "Product",
        "keywords": ["container house", "modular home", "prefab", "steel structure", "tiny house"]
    }
}

# ==========================================
# 1. 核心工具函数 (已修复图片处理)
# ==========================================
def get_font(size):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except: return ImageFont.load_default()

def convert_to_webp(image):
    buf = io.BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(buf, format='WEBP', quality=85)
    return buf.getvalue()

def pil_to_base64_safe(img):
    """【关键修复】转JPEG并压缩，防止智谱/Gemini报错"""
    buf = io.BytesIO()
    # 强制转RGB
    if img.mode != 'RGB': img = img.convert('RGB')
    # 限制尺寸 (2048px足够)
    max_side = 2048
    if img.width > max_side or img.height > max_side:
        img.thumbnail((max_side, max_side))
    # 存为JPEG
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ==========================================
# 2. AI 调用核心逻辑 (已修复 Gemini/智谱)
# ==========================================
def run_ai_vision(engine, img, prompt, key, model_name):
    """底层 AI 识图函数"""
    if not key: return "Error: 缺少 API Key"
    
    try:
        # --- Google Gemini ---
        if engine == "Google Gemini":
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            # Gemini 最佳实践: [prompt, image]
            response = model.generate_content([prompt, img])
            return response.text
        
        # --- 智谱清言 ---
        elif engine == "智谱清言":
            client = ZhipuAI(api_key=key)
            # 自动回退模型 (glm-4 不支持识图，强制切 glm-4v)
            vision_model = model_name
            if "glm-4" in model_name and "v" not in model_name and "plus" not in model_name:
                 vision_model = "glm-4v"
            
            # 使用修复后的 Base64 JPEG
            b64_img = pil_to_base64_safe(img)
            
            response = client.chat.completions.create(
                model=vision_model,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }]
            )
            return response.choices[0].message.content
        
        # --- 阿里通义 ---
        elif engine == "阿里通义":
            dashscope.api_key = key
            # 使用临时文件上传
            tmp_path = f"temp_{uuid.uuid4()}.jpg"
            if img.mode != "RGB": img = img.convert("RGB")
            img.save(tmp_path, format="JPEG")
            try:
                msgs = [{"role": "user", "content": [{"image": f"file://{os.path.abspath(tmp_path)}"}, {"text": prompt}]}]
                res = MultiModalConversation.call(model=model_name, messages=msgs)
                if isinstance(res.output.choices[0].message.content, list):
                    return res.output.choices[0].message.content[0]['text']
                return res.output.choices[0].message.content
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
        
        return "Error: 未知引擎"
    except Exception as e: return f"Error: {str(e)}"

def run_ai_text(engine, prompt, key, model_name):
    """底层 AI 纯文本函数"""
    if not key: return "Error: 缺少 API Key"
    try:
        if engine == "Google Gemini":
            genai.configure(api_key=key)
            return genai.GenerativeModel(model_name).generate_content(prompt).text
        elif engine == "智谱清言":
            # 纯文本通常用 glm-4
            txt_model = "glm-4" if "v" in model_name else model_name
            client = ZhipuAI(api_key=key)
            resp = client.chat.completions.create(model=txt_model, messages=[{"role":"user","content":prompt}])
            return resp.choices[0].message.content
        elif engine == "阿里通义":
            dashscope.api_key = key
            # 文本通常用 qwen-max
            resp = Generation.call(model='qwen-max', messages=[{"role":"user","content":prompt}])
            return resp.output.text
    except Exception as e: return f"Error: {str(e)}"

# 带重试的识图 (用于 Tab 1)
def run_ai_vision_with_retry(engine, img, prompt, key, model_name, max_retries=2):
    for attempt in range(max_retries):
        res = run_ai_vision(engine, img, prompt, key, model_name)
        if res and not res.startswith("Error"):
            return res
    return res

# ==========================================
# 3. 侧边栏配置 (UI)
# ==========================================
with st.sidebar:
    st.title("⚙️ 配置 V29.3")
    
    # 业务选择
    st.subheader("1. 业务模式")
    biz_choice = st.radio("Business", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"), label_visibility="collapsed")
    cbiz = "logistics" if "VastLog" in biz_choice else "house"
    cinfo = BIZ_CONFIG[cbiz]
    
    st.divider()
    
    # 引擎选择 (修复了模型列表)
    st.subheader("2. AI 引擎")
    engine_choice = st.radio("Vendor", ("Google Gemini", "智谱清言", "阿里通义"))
    
    if engine_choice == "Google Gemini":
        # 【修复】使用真实存在的模型名称
        model_options = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = GOOGLE_API_KEY
    elif engine_choice == "智谱清言":
        # 【修复】识图推荐用 glm-4v
        model_options = ["glm-4v", "glm-4v-flash", "glm-4-plus"]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = ZHIPU_API_KEY
    else:
        model_options = ["qwen-vl-max", "qwen-vl-plus"]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = ALI_API_KEY

# ==========================================
# 4. 主界面 Tabs
# ==========================================
st.title(f"🦁 {cinfo['name']} 数字化运营台")
tab1, tab2, tab3 = st.tabs(["✍️ 智能文案", "🎨 封面工厂", "🌍 GEO/SEO 专家"])

# --- Tab 1: 智能文案 ---
with tab1:
    c1, c2 = st.columns([1, 1])
    files_t1 = c1.file_uploader("📂 上传图片", accept_multiple_files=True, key="t1")
    with c2:
        draft = st.text_area("补充信息 (全套模式)", height=100)
        b1, b2 = st.columns(2)
        btn_name = b1.button("🖼️ 仅识图起名", use_container_width=True)
        btn_full = b2.button("🚀 全套处理", type="primary", use_container_width=True)

    if (btn_name or btn_full) and files_t1:
        st.session_state.results_tab1 = []
        kw_str = ", ".join(cinfo['keywords'][:4])
        
        # 简化版 Prompt，防止模型发疯
        prompt_seo = f"""
        Task: SEO Filename.
        Brand: {cinfo['name']}
        Keywords Context: {kw_str}
        Format: {cinfo['name'].lower()}-keyword1-keyword2.
        Rules: Lowercase, hyphens only, no spaces, max 6 words.
        Output ONLY the filename string.
        """
        
        prompt_copy = f"Write a Facebook post for {cinfo['name']}. Context: {draft}."
        
        bar = st.progress(0)
        for i, f in enumerate(files_t1):
            img = Image.open(f)
            # 1. 起名
            raw_name = run_ai_vision_with_retry(engine_choice, img, prompt_seo, api_key, sel_model)
            # 清洗结果
            clean_name = raw_name.strip().lower().replace(" ", "-").replace("_", "-")
            # 简单的正则提取，防止AI废话
            clean_name = re.sub(r'[^a-z0-9-]', '', clean_name)
            if not clean_name.startswith(cinfo['name'].lower()):
                clean_name = f"{cinfo['name'].lower()}-{clean_name}"
            
            # 2. 文案
            copy_text = ""
            if btn_full:
                copy_text = run_ai_vision(engine_choice, img, prompt_copy, api_key, sel_model)
            
            st.session_state.results_tab1.append({
                "img": img, "name": f"{clean_name[:50]}.webp", "text": copy_text, "data": convert_to_webp(img)
            })
            bar.progress((i+1)/len(files_t1))

    if st.session_state.results_tab1:
        st.divider()
        for res in st.session_state.results_tab1:
            l, r = st.columns([1, 3])
            l.image(res['img'], width=150)
            with r:
                st.text_input("SEO文件名", res['name'], key=f"n_{uuid.uuid4()}")
                if res['text']: st.text_area("文案", res['text'], height=80)
                st.download_button("下载WebP", res['data'], res['name'])

# --- Tab 2: 封面工厂 ---
with tab2:
    bg_col, txt_col = st.columns([1, 1])
    with bg_col:
        st.markdown("#### A. 背景")
        mode = st.radio("来源", ["本地上传", "AI生图 (阿里)"], horizontal=True)
        bg_img = None
        if mode == "本地上传":
            f = st.file_uploader("上传背景", key="t2_up")
            if f: bg_img = Image.open(f).convert("RGBA")
        else:
            p = st.text_input("画面描述", "container ship at sunset")
            if st.button("生成背景"):
                if not ALI_API_KEY: st.error("需配置 ALI_API_KEY")
                else:
                    dashscope.api_key = ALI_API_KEY
                    rsp = ImageSynthesis.call(model=ImageSynthesis.Models.wanx_v1, prompt=p, n=1, size='1024*1024')
                    if rsp.status_code==200:
                        st.session_state.generated_bg = Image.open(io.BytesIO(requests.get(rsp.output.results[0].url).content)).convert("RGBA")
            if st.session_state.generated_bg: bg_img = st.session_state.generated_bg

    with txt_col:
        st.markdown("#### B. 文字")
        with st.expander("标题 1", expanded=True):
            t1 = st.text_input("Txt1", "Global Logistics"); s1 = st.number_input("Size1", 20,300,80); c1 = st.color_picker("Col1", "#FFF"); y1 = st.slider("Y1",0,1000,100)
        with st.expander("标题 2"):
            t2 = st.text_input("Txt2", "DDP Service"); s2 = st.number_input("Size2", 20,300,50); c2 = st.color_picker("Col2", cinfo['color']); y2 = st.slider("Y2",0,1000,250)
        with st.expander("标题 3"):
            t3 = st.text_input("Txt3", "Fast & Safe"); s3 = st.number_input("Size3", 20,300,30); c3 = st.color_picker("Col3", "#FF0"); y3 = st.slider("Y3",0,1000,350)

    if bg_img:
        st.divider()
        final = bg_img.copy(); draw = ImageDraw.Draw(final); W,H = final.size
        def dr(t,s,c,y):
            if not t: return
            f = get_font(int(s))
            try: w = draw.textlength(t, font=f)
            except: w = draw.textbbox((0,0),t,font=f)[2]
            x = (W-w)/2
            draw.text((x+4,y+4),t,font=f,fill="black"); draw.text((x,y),t,font=f,fill=c)
        dr(t1,s1,c1,y1); dr(t2,s2,c2,y2); dr(t3,s3,c3,y3)
        st.image(final, use_container_width=True)
        buf=io.BytesIO(); final.convert("RGB").save(buf,"JPEG"); st.download_button("下载封面", buf.getvalue(), "cover.jpg")

# --- Tab 3: GEO/SEO 专家 ---
with tab3:
    st.caption(f"当前引擎: {engine_choice} | 任务: 中译英 + SEO + 自动插图")
    cc1, cc2 = st.columns([1, 1])
    with cc1: cn_txt = st.text_area("中文原文", height=200)
    with cc2: imgs = st.file_uploader("配图 (AI会自动插入)", accept_multiple_files=True, key="t3_imgs")

    if st.button("✨ 生成 GEO 代码", type="primary"):
        if not cn_txt: st.warning("请输入中文")
        else:
            sys_p = f"""
            Role: SEO Expert for {cinfo['name']}.
            Task: Translate CHINESE to ENGLISH. Keep meaning.
            Format: HTML Article. Use <h2> tags with style="border-left:5px solid {cinfo['color']}; padding-left:10px;".
            Schema: Add <script type="application/ld+json"> for {cinfo['type']}.
            Images: Insert <img src="filename" alt="SEO alt"> tags where appropriate.
            """
            
            with st.spinner("Running AI..."):
                try:
                    final_html = ""
                    # 1. Google 模式 (最强多模态)
                    if engine_choice == "Google Gemini":
                        cnt = [sys_p, f"Input:\n{cn_txt}"]
                        if imgs:
                            cnt.append("\nAvailable Images:")
                            for f in imgs: cnt.extend([f"\nFile: {f.name}", Image.open(f)])
                        genai.configure(api_key=api_key)
                        final_html = genai.GenerativeModel(sel_model).generate_content(cnt).text

                    # 2. 智谱/阿里 模式 (文本+文件名)
                    else:
                        img_note = ""
                        if imgs: img_note = f"\nImage files available: {', '.join([f.name for f in imgs])}"
                        full_p = sys_p + img_note + f"\n\nText:\n{cn_txt}"
                        
                        if engine_choice == "智谱清言": # 用纯文本模型处理翻译和排版
                            client = ZhipuAI(api_key=api_key)
                            # 强制切回文本模型 glm-4 或 glm-4-plus
                            t_model = "glm-4-plus" if "plus" in sel_model else "glm-4"
                            resp = client.chat.completions.create(model=t_model, messages=[{"role":"user","content":full_p}])
                            final_html = resp.choices[0].message.content
                        else: # 阿里 qwen-max
                            resp = Generation.call(model='qwen-max', messages=[{"role":"user","content":full_p}])
                            final_html = resp.output.text

                    # 展示
                    v, c = st.columns([1, 1])
                    v.markdown(final_html, unsafe_allow_html=True)
                    c.code(final_html, language="html")
                except Exception as e: st.error(f"Error: {str(e)}")
