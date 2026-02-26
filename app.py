import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import dashscope 
from dashscope import ImageSynthesis, MultiModalConversation, Generation
from zhipuai import ZhipuAI
from PIL import Image, ImageDraw, ImageFont
import io, base64, re, os, requests, uuid, zipfile, time

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V36.0", layout="wide", page_icon="🦁")

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
        "name": "VastLog", "website": "www.vastlog.com", "color": "#FF9900", 
        "type": "LogisticsService", "keywords": ["logistics", "shipping", "freight", "cargo"],
        "action": "Get a Free Shipping Quote"
    },
    "house": {
        "name": "Wellucky", "website": "www.welluckyhouse.com", "color": "#0066CC", 
        "type": "Product", "keywords": ["container house", "modular home", "prefab"],
        "action": "Customize Your Container Home"
    }
}

# ==========================================
# 1. 核心工具 (加入防卡死图片压缩)
# ==========================================
def get_font(size):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except: return ImageFont.load_default()

def resize_image_for_api(img, max_size=1500):
    """预处理：压缩图片尺寸，防止API超时卡死"""
    if img.mode != 'RGB': img = img.convert('RGB')
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size))
    return img

def convert_to_webp(image):
    buf = io.BytesIO()
    img = resize_image_for_api(image, 1500) # 转换前也压缩一下
    img.save(buf, format='WEBP', quality=85)
    return buf.getvalue()

def pil_to_base64_safe(img):
    buf = io.BytesIO()
    img = resize_image_for_api(img, 1500)
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ==========================================
# 2. AI 调用逻辑 (增强稳定性)
# ==========================================
def run_ai_vision(engine, img, prompt, key, model_name):
    if not key: return "Error: 缺少 API Key"
    try:
        # 统一预处理图片，防止卡顿
        processed_img = resize_image_for_api(img)
        
        if engine == "Google Gemini":
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, processed_img])
            return response.text
        elif engine == "智谱清言":
            client = ZhipuAI(api_key=key)
            vision_model = "glm-4v"
            b64_img = pil_to_base64_safe(processed_img)
            response = client.chat.completions.create(
                model=vision_model,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}]
            )
            return response.choices[0].message.content
        elif engine == "阿里通义":
            dashscope.api_key = key
            tmp_path = f"temp_{uuid.uuid4()}.jpg"
            processed_img.save(tmp_path, format="JPEG")
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
        try:
            res = run_ai_vision(engine, img, prompt, key, model_name)
            if res and not res.startswith("Error"): return res
        except:
            time.sleep(1) # 失败稍微等一下
    return f"{uuid.uuid4().hex[:8]}" # 如果全失败，返回随机码保底，防止程序崩

# ==========================================
# 3. 侧边栏配置
# ==========================================
with st.sidebar:
    st.title("⚙️ 配置 V36.0")
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
st.caption(f"Engine: {engine_choice} | Model: {sel_model}")
tab1, tab2, tab3 = st.tabs(["✍️ 智能文案", "🎨 封面工厂", "🌍 GEO/AIO 专家"])

# --- Tab 1: 智能文案 (修复卡顿问题) ---
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
        
        # 强制视觉差异 Prompt
        prompt_seo = f"""
        Role: SEO Expert for {cinfo['name']}.
        Task: Create a UNIQUE filename based on VISUAL DIFFERENCES.
        Keywords: {kw_str}.
        Format: {cinfo['name'].lower()}-keyword-[VisualFeature].
        Rules: Lowercase, hyphens only. No .jpg extension.
        Focus on: Angle, Color, Context, Interior/Exterior.
        """
        
        prompt_copy = f"Write a Facebook post for {cinfo['name']}. Context: {draft}."
        
        # 进度条 + 状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        name_counter = {}
        
        for i, f in enumerate(files_t1):
            # 【修复】显示实时进度，不再让用户以为卡死
            status_text.info(f"⏳ 正在分析第 {i+1} / {len(files_t1)} 张图片: {f.name} ...")
            
            img = Image.open(f)
            # 1. AI 起名 (带重试)
            raw_name = run_ai_vision_with_retry(engine_choice, img, prompt_seo, api_key, sel_model)
            
            # 2. 清洗
            base = raw_name.strip().lower().replace(" ", "-").replace("_", "-")
            base = re.sub(r'[^a-z0-9-]', '', base)
            base = re.sub(r'-+', '-', base).strip('-')
            base = re.sub(r'\.(jpg|jpeg|png|webp)$', '', base) # 再次确保无后缀

            if not base.startswith(cinfo['name'].lower()):
                base = f"{cinfo['name'].lower()}-{base}"
            
            # 3. 序列号防重
            if base in name_counter:
                name_counter[base] += 1
                fname = f"{base}-{name_counter[base]:02d}"
            else:
                name_counter[base] = 1
                fname = base

            # 4. 文案
            copy_txt = ""
            if btn_full:
                copy_txt = run_ai_vision(engine_choice, img, prompt_copy, api_key, sel_model)
            
            st.session_state.results_tab1.append({"img": img, "name": f"{fname}.webp", "text": copy_txt, "data": convert_to_webp(img)})
            progress_bar.progress((i+1)/len(files_t1))
        
        status_text.success("✅ 所有图片处理完成！")

    # 结果展示
    if st.session_state.results_tab1:
        st.divider()
        c_down, c_clear = st.columns([1, 1])
        with c_down:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for res in st.session_state.results_tab1:
                    zf.writestr(res['name'], res['data'])
            st.download_button(f"📦 批量下载 ZIP ({len(st.session_state.results_tab1)}张)", zip_buf.getvalue(), "images.zip", "application/zip", use_container_width=True, type="primary")
        with c_clear:
            if st.button("🗑️ 清空列表", use_container_width=True):
                st.session_state.results_tab1 = []
                st.rerun()

        st.divider()
        for i, res in enumerate(st.session_state.results_tab1):
            l, r = st.columns([1, 3])
            l.image(res['img'], width=120)
            with r:
                ukey = f"{i}_{uuid.uuid4()}"
                st.text_input("文件名", res['name'], key=f"n_{ukey}")
                if res['text']: st.text_area("文案", res['text'], height=60, key=f"t_{ukey}")

# --- Tab 2: 封面工厂 (保持不变) ---
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

# --- Tab 3: GEO/AIO 专家 (修复：复制按钮消失) ---
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

            sys_p = f"""
            Role: SEO & Web Designer for {cinfo['name']}. Task: Translate & Format.
            Target Keyword: "{target_kw if target_kw else 'Auto-detect'}"
            [RULE 1: NO SCRIPTS] USE MICRODATA in HTML. No <script>.
            [RULE 2: FIDELITY] Translate accurately.
            [RULE 3: STYLE] Use <h2> styled (border-left brand color). HTML Tables. Images with alt text.
            
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
                        
                        # 【核心修复】分离 Tab：一个看效果，一个复制由 Streamlit 原生提供的带复制按钮的代码框
                        tab_view, tab_code = st.tabs(["👁️ 效果预览 (不可复制)", "💻 获取 HTML 代码 (一键复制)"])
                        
                        with tab_view:
                            # 清洗markdown符号用于预览
                            clean_view = final_html_output.replace("```html", "").replace("```", "")
                            components.html(clean_view, height=600, scrolling=True)
                        
                        with tab_code:
                            st.info("👇 点击代码框右上角的 📄 图标即可一键复制全部代码")
                            # 这里放原始代码，Streamlit 会自动加上复制按钮
                            st.code(final_html_output, language="html")

                    except Exception as parse_e:
                        st.error("解析格式略有偏差，请手动复制：")
                        st.code(final_res)

                except Exception as e: st.error(f"Error: {str(e)}")
