import streamlit as st
import streamlit.components.v1 as components # 新增：用于完美渲染HTML
import google.generativeai as genai
import dashscope 
from dashscope import ImageSynthesis, MultiModalConversation, Generation
from zhipuai import ZhipuAI
from PIL import Image, ImageDraw, ImageFont
import io, base64, re, os, requests, uuid, zipfile

# ==========================================
# 0. 全局配置 & 初始化
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V35.1", layout="wide", page_icon="🦁")

if 'results_tab1' not in st.session_state: st.session_state.results_tab1 = []
if 'generated_bg' not in st.session_state: st.session_state.generated_bg = None

def get_secret_safe(key_name, default=""):
    try: return st.secrets.get(key_name, default)
    except: return default

GOOGLE_API_KEY = get_secret_safe("GOOGLE_API_KEY")
ALI_API_KEY = get_secret_safe("ALI_API_KEY")
ZHIPU_API_KEY = get_secret_safe("ZHIPU_API_KEY")

BIZ_CONFIG = {
    "logistics": {
        "name": "VastLog", 
        "website": "www.vastlog.com", 
        "color": "#FF9900", 
        "type": "LogisticsService", 
        "keywords": ["logistics", "shipping", "freight", "cargo", "DDP"],
        "action": "Get a Free Shipping Quote"
    },
    "house": {
        "name": "Wellucky", 
        "website": "www.welluckyhouse.com", 
        "color": "#0066CC", 
        "type": "Product", 
        "keywords": ["container house", "modular home", "prefab", "steel structure"],
        "action": "Customize Your Container Home"
    }
}

# ==========================================
# 1. 核心工具函数 (修复：文件名清洗)
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
    buf = io.BytesIO()
    if img.mode != 'RGB': img = img.convert('RGB')
    max_side = 2048
    if img.width > max_side or img.height > max_side:
        img.thumbnail((max_side, max_side))
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# 【关键修复】文件名强力清洗函数
def clean_filename_logic(raw_text, brand):
    # 1. 转小写，去空格
    name = raw_text.strip().lower().replace(" ", "-").replace("_", "-")
    # 2. 移除扩展名 (jpg, png, webp)
    name = re.sub(r'\.(jpg|jpeg|png|webp)$', '', name)
    # 3. 只保留字母数字和连字符
    name = re.sub(r'[^a-z0-9-]', '', name)
    # 4. 修复多余连字符
    name = re.sub(r'-+', '-', name).strip('-')
    # 5. 确保品牌开头
    if not name.startswith(brand.lower()):
        name = f"{brand.lower()}-{name}"
    # 6. 【防重名】强制添加 4位 随机哈希
    unique_suffix = uuid.uuid4().hex[:4]
    return f"{name}-{unique_suffix}"

# ==========================================
# 2. AI 调用逻辑
# ==========================================
def run_ai_vision(engine, img, prompt, key, model_name):
    if not key: return "Error: 缺少 API Key"
    try:
        if engine == "Google Gemini":
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, img])
            return response.text
        elif engine == "智谱清言":
            client = ZhipuAI(api_key=key)
            vision_model = "glm-4v"
            b64_img = pil_to_base64_safe(img)
            response = client.chat.completions.create(
                model=vision_model,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}]
            )
            return response.choices[0].message.content
        elif engine == "阿里通义":
            dashscope.api_key = key
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

def run_ai_vision_with_retry(engine, img, prompt, key, model_name, max_retries=2):
    for attempt in range(max_retries):
        res = run_ai_vision(engine, img, prompt, key, model_name)
        if res and not res.startswith("Error"): return res
    return res

# ==========================================
# 3. 侧边栏配置
# ==========================================
with st.sidebar:
    st.title("⚙️ 配置 V35.1")
    st.subheader("1. 业务模式")
    biz_choice = st.radio("Business", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"), label_visibility="collapsed")
    cbiz = "logistics" if "VastLog" in biz_choice else "house"
    cinfo = BIZ_CONFIG[cbiz]
    
    st.divider()
    st.subheader("2. AI 引擎")
    engine_choice = st.radio("Vendor", ("Google Gemini", "智谱清言", "阿里通义"))
    
    if engine_choice == "Google Gemini":
        model_options = ["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro"]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = GOOGLE_API_KEY
    elif engine_choice == "智谱清言":
        model_options = ["glm-4v", "glm-4v-flash"]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = ZHIPU_API_KEY
    else:
        model_options = ["qwen-vl-max", "qwen-vl-plus"]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = ALI_API_KEY

# ==========================================
# 4. 主界面
# ==========================================
st.title(f"🦁 {cinfo['name']} 数字化运营台")
st.caption(f"Current Model: {sel_model}")
tab1, tab2, tab3 = st.tabs(["✍️ 智能文案", "🎨 封面工厂", "🌍 GEO/AIO 专家"])

# --- Tab 1: 智能文案 (已修复命名重复 & 后缀问题) ---
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
        
        # =========================================================
        # V35.3 核心：防偷懒 Prompt
        # 强制 AI 必须描述独特的视觉特征，而不是只给个通用名
        # =========================================================
        prompt_seo = f"""
        Role: Senior SEO Specialist for {cinfo['name']}.
        Task: Generate a UNIQUE, descriptively rich filename for this specific image.
        Target Keywords: {kw_str}.
        
        [MANDATORY NAMING FORMULA]
        {cinfo['name'].lower()} - [Core Product] - [VISUAL DIFFERENTIATOR]
        
        [RULES FOR "VISUAL DIFFERENTIATOR"]
        You MUST describe what makes THIS image different from others. Look for:
        1. Viewpoint (e.g., front-view, aerial-view, interior-bedroom, close-up-steel-frame)
        2. Action/Context (e.g., installation-site, loading-truck, solar-roof-detail)
        3. Features (e.g., glass-sliding-door, foldable-wall, white-panel)
        
        [PROHIBITED]
        - DO NOT just say "{cinfo['name'].lower()}-container-house". That is LAZY.
        - DO NOT include file extensions (.jpg).
        - Use lowercase and hyphens only.
        """
        
        prompt_copy = f"Write a Facebook post for {cinfo['name']}. Context: {draft}."
        
        bar = st.progress(0)
        name_counter = {} # 计数器（仅作为最后的防撞车保险）
        
        for i, f in enumerate(files_t1):
            img = Image.open(f)
            
            # 1. AI 尽力起名 (包含视觉细节)
            raw_name = run_ai_vision_with_retry(engine_choice, img, prompt_seo, api_key, sel_model)
            
            # 2. 清洗
            base_name = raw_name.strip().lower().replace(" ", "-").replace("_", "-")
            base_name = re.sub(r'[^a-z0-9-]', '', base_name)
            base_name = re.sub(r'-+', '-', base_name).strip('-')
            base_name = re.sub(r'\.(jpg|jpeg|png|webp)$', '', base_name)

            if not base_name.startswith(cinfo['name'].lower()):
                base_name = f"{cinfo['name'].lower()}-{base_name}"
            
            # 3. 智能序列号 (仅在 AI 起名真的撞车时才触发)
            if base_name in name_counter:
                name_counter[base_name] += 1
                seq_num = name_counter[base_name]
                final_name = f"{base_name}-{seq_num:02d}" # 撞车了，没办法，加序号
            else:
                name_counter[base_name] = 1
                final_name = base_name # 没撞车，保留 AI 的精准描述 (High Value SEO)

            # 4. 文案
            copy_text = ""
            if btn_full:
                copy_text = run_ai_vision(engine_choice, img, prompt_copy, api_key, sel_model)
            
            st.session_state.results_tab1.append({"img": img, "name": f"{final_name}.webp", "text": copy_text, "data": convert_to_webp(img)})
            bar.progress((i+1)/len(files_t1))

# --- Tab 2: 封面工厂 (保持) ---
with tab2:
    bg_col, txt_col = st.columns([1, 1])
    with bg_col:
        st.markdown("#### A. 背景")
        mode = st.radio("来源", ["本地上传", "AI生图"], horizontal=True)
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

# --- Tab 3: GEO/AIO 专家 (修复：完美预览) ---
with tab3:
    st.caption(f"当前引擎: {engine_choice} | 模型: {sel_model}")
    st.markdown(f"##### 🛡️ 完美排版 & 安全 SEO 套件 (当前对象: **{cinfo['name']}**)")
    
    cc1, cc2 = st.columns([1, 1])
    with cc1: 
        cn_txt = st.text_area("中文原文 / 核心卖点", height=300, placeholder="粘贴内容...")
        target_kw = st.text_input("🎯 目标关键词", placeholder="例如: Luxury Prefab House")
    with cc2: 
        imgs = st.file_uploader("配图 (AI自动插入)", accept_multiple_files=True, key="t3_imgs")

    if st.button("✨ 生成完美排版", type="primary", use_container_width=True):
        if not cn_txt: st.warning("请输入中文")
        else:
            # Wellucky CTA
            wellucky_cta_html = """
<div style="max-width: 700px; margin: 60px auto; padding: 40px 30px; background: #1a1a1a; color: #fff; border-radius: 16px; text-align: center; box-shadow: 0 15px 40px rgba(0,0,0,0.2);">
    <h3 style="font-size: 24px; margin-bottom: 15px; color: #fff; letter-spacing: 0.5px;">Why Choose Wellucky?</h3>
    <p style="color: #ccc; font-size: 15px; margin-bottom: 25px; line-height: 1.6;">We are a <strong>professional manufacturer since 2005</strong>. We offer comprehensive <strong>OEM/ODM services</strong>.</p>
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 15px;">
        <a href="https://www.welluckyhouse.com/contact" target="_blank" style="background: #1e7e34; color: #fff; text-decoration: none; padding: 12px 30px; border-radius: 50px; font-weight: bold; font-size: 16px;">GET A QUOTE</a>
        <a href="mailto:info@welluckyhouse.com" style="border: 1px solid #fff; color: #fff; text-decoration: none; padding: 11px 30px; border-radius: 50px; font-weight: bold; font-size: 16px;">EMAIL US</a>
    </div>
</div>
            """

            # 提示词
            sys_p = f"""
            Role: SEO & Web Designer for {cinfo['name']}. Task: Translate & Format.
            Target Keyword: "{target_kw if target_kw else 'Auto-detect'}"
            [RULE 1: NO SCRIPTS] USE MICRODATA in HTML tags. No <script>.
            [RULE 2: FIDELITY] Translate accurately.
            [RULE 3: STYLE] Use <h2> styled (border-left brand color). HTML Tables for specs. Images with alt text.
            
            OUTPUT FORMAT:
            |||TITLE|||...
            |||SLUG|||...
            |||KEYWORDS|||...
            |||DESCRIPTION|||...
            |||CONTENT|||... (HTML Body)
            """
            
            with st.spinner("正在排版..."):
                try:
                    final_res = ""
                    if engine_choice == "Google Gemini":
                        cnt = [sys_p, f"Input Text:\n{cn_txt}"]
                        if imgs:
                            cnt.append("\nImages:")
                            for f in imgs: cnt.extend([f"\nFile: {f.name}", Image.open(f)])
                        genai.configure(api_key=api_key)
                        final_res = genai.GenerativeModel(sel_model).generate_content(cnt).text
                    else:
                        img_note = f"\nImages: {', '.join([f.name for f in imgs])}" if imgs else ""
                        full_p = sys_p + img_note + f"\n\nText:\n{cn_txt}"
                        if engine_choice == "智谱清言":
                            client = ZhipuAI(api_key=api_key)
                            resp = client.chat.completions.create(model="glm-4-plus", messages=[{"role":"user","content":full_p}])
                            final_res = resp.choices[0].message.content
                        else:
                            resp = Generation.call(model='qwen-max', messages=[{"role":"user","content":full_p}])
                            final_res = resp.output.text

                    # 解析
                    try:
                        p_title = final_res.split("|||TITLE|||")[1].split("|||")[0].strip()
                        p_slug = final_res.split("|||SLUG|||")[1].split("|||")[0].strip()
                        p_kws = final_res.split("|||KEYWORDS|||")[1].split("|||")[0].strip()
                        p_desc = final_res.split("|||DESCRIPTION|||")[1].split("|||")[0].strip()
                        p_content_raw = final_res.split("|||CONTENT|||")[1].strip()
                        
                        # 拼接 + 容器
                        if cinfo['name'] == "Wellucky": p_content_raw += wellucky_cta_html
                        final_html_output = f"""<div style="max-width: 900px; margin: 0 auto; font-family: sans-serif; line-height: 1.8; color: #333; padding: 20px;">{p_content_raw}</div>"""

                        st.success("✅ 生成成功！")
                        
                        st.markdown("### 1. 基础字段")
                        c_t, c_s = st.columns([2, 1])
                        c_t.text_input("📋 1. 主题 (Title)", value=p_title)
                        c_s.text_input("🔗 2. 自定义URL", value=p_slug)
                        
                        st.markdown("### 2. SEO 字段")
                        st.text_input("🔑 3. 关键字", value=p_kws)
                        st.text_area("📝 4 & 5. 描述 / 摘要", value=p_desc, height=100)
                        
                        st.markdown("### 3. 内容编辑器")
                        st.info("💡 请点击编辑器左上角的 [HTML] 按钮粘贴。预览如下：")
                        
                        # 【核心修复】使用 components.html 完美渲染，不再显示代码
                        with st.expander("📄 预览 & 代码", expanded=True):
                            # 1. 清洗掉可能存在的markdown符号
                            clean_code = final_html_output.replace("```html", "").replace("```", "")
                            # 2. 真实渲染
                            components.html(clean_code, height=600, scrolling=True)
                            st.divider()
                            st.caption("复制代码：")
                            st.code(clean_code, language="html")

                    except Exception as parse_e:
                        st.error("解析格式略有偏差，请手动复制：")
                        st.code(final_res)

                except Exception as e: st.error(f"Error: {str(e)}")
