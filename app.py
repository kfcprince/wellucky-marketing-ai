import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import ImageSynthesis, MultiModalConversation, Generation
from zhipuai import ZhipuAI
from PIL import Image, ImageDraw, ImageFont
import io, base64, re, os, requests, uuid, zipfile

# ==========================================
# 0. 全局配置 & 初始化
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V32.1", layout="wide", page_icon="🦁")

if 'results_tab1' not in st.session_state: st.session_state.results_tab1 = []
if 'generated_bg' not in st.session_state: st.session_state.generated_bg = None

def get_secret_safe(key_name, default=""):
    try: return st.secrets.get(key_name, default)
    except: return default

GOOGLE_API_KEY = get_secret_safe("GOOGLE_API_KEY")
ALI_API_KEY = get_secret_safe("ALI_API_KEY")
ZHIPU_API_KEY = get_secret_safe("ZHIPU_API_KEY")

# 业务配置 (已包含 Action CTA)
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
# 1. 核心工具函数
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
    st.title("⚙️ 配置 V32.1")
    
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
        
        prompt_seo = f"""
        Role: SEO Expert for {cinfo['name']}.
        Task: Create a UNIQUE filename for this image.
        Keywords: {kw_str}.
        CRITICAL RULES:
        1. Analyze specific visual details: Color? Angle? Details?
        2. Format: {cinfo['name'].lower()}-feature-detail-keyword.
        3. No generic names. Be specific.
        4. Lowercase, hyphens only.
        5. Output ONLY the filename string.
        """
        
        prompt_copy = f"Write a Facebook post for {cinfo['name']}. Context: {draft}."
        
        bar = st.progress(0)
        for i, f in enumerate(files_t1):
            img = Image.open(f)
            # 1. 起名
            raw_name = run_ai_vision_with_retry(engine_choice, img, prompt_seo, api_key, sel_model)
            clean_name = re.sub(r'[^a-z0-9-]', '', raw_name.strip().lower().replace(" ", "-").replace("_", "-"))
            clean_name = re.sub(r'-+', '-', clean_name).strip('-')
            
            if not clean_name.startswith(cinfo['name'].lower()):
                clean_name = f"{cinfo['name'].lower()}-{clean_name}"
            
            if len(clean_name.split('-')) < 3:
                 clean_name = f"{clean_name}-{uuid.uuid4().hex[:4]}"

            # 2. 文案
            copy_text = ""
            if btn_full:
                copy_text = run_ai_vision(engine_choice, img, prompt_copy, api_key, sel_model)
            
            st.session_state.results_tab1.append({"img": img, "name": f"{clean_name}.webp", "text": copy_text, "data": convert_to_webp(img)})
            bar.progress((i+1)/len(files_t1))

    # 批量操作区
    if st.session_state.results_tab1:
        st.divider()
        st.markdown("### 🛠️ 批量操作")
        col_down, col_clear = st.columns([1, 1])
        with col_down:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for res in st.session_state.results_tab1:
                    zf.writestr(res['name'], res['data'])
            st.download_button(f"📦 批量下载 {len(st.session_state.results_tab1)} 张图片 (ZIP)", zip_buffer.getvalue(), f"{cinfo['name'].lower()}-batch.zip", "application/zip", use_container_width=True, type="primary")
        with col_clear:
            if st.button("🗑️ 清空列表", use_container_width=True):
                st.session_state.results_tab1 = []
                st.rerun()

        st.divider()
        for i, res in enumerate(st.session_state.results_tab1):
            l, r = st.columns([1, 3])
            l.image(res['img'], width=150)
            with r:
                ukey = f"{i}_{uuid.uuid4()}"
                st.text_input("SEO文件名", res['name'], key=f"n_{ukey}")
                if res['text']: st.text_area("文案", res['text'], height=80, key=f"t_{ukey}")
                st.download_button("⬇️ 单图下载", res['data'], res['name'], key=f"d_{ukey}")

# --- Tab 2: 封面工厂 ---
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

# --- Tab 3: GEO/AIO 专家 (高保真 + Wellucky 专属) ---
with tab3:
    st.caption(f"当前引擎: {engine_choice} | 模型: {sel_model}")
    st.markdown(f"##### 🛡️ CMS 镜像发布套件 (当前对象: **{cinfo['name']}**)")
    
    # 输入区保持不变
    cc1, cc2 = st.columns([1, 1])
    with cc1: 
        cn_txt = st.text_area("中文原文 / 核心卖点", height=300, placeholder="粘贴内容...")
        target_kw = st.text_input("🎯 目标关键词", placeholder="例如: Luxury Prefab House")
    with cc2: 
        imgs = st.file_uploader("配图 (AI自动插入)", accept_multiple_files=True, key="t3_imgs")

    if st.button("✨ 生成内容 (按后台顺序排列)", type="primary", use_container_width=True):
        if not cn_txt: st.warning("请输入中文")
        else:
            # Wellucky 专属 CTA (硬编码)
            wellucky_cta_html = """
<div style="margin: 40px 0; padding: 50px 30px; background: #1a1a1a; color: #fff; border-radius: 20px; text-align: center;">
    <h3 style="font-size: 28px; margin-bottom: 15px; color: #fff;">Why Choose Wellucky?</h3>
    <p style="color: #aaa; margin-bottom: 30px; max-width: 800px; margin-left: auto; margin-right: auto;">
        We are a <strong>professional manufacturer since 2005</strong> with a proven track record in engineering and exporting high-quality prefab modular structures. We offer comprehensive <strong>OEM/ODM services</strong>—from design consultation to final delivery—ensuring your specific project needs are met.
    </p>
    <p style="color: #fff; font-weight: bold; margin-bottom: 30px;">Invest in Efficiency, Quality, and Innovation. Let’s Build Your Vision Together.</p>
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;">
        <a href="https://www.welluckyhouse.com/contact" target="_blank" rel="noopener noreferrer" style="background: #1e7e34; color: #fff; text-decoration: none; padding: 15px 40px; border-radius: 50px; font-weight: bold; font-size: 18px;">INQUIRY FOR QUOTE</a>
        <a href="mailto:info@welluckyhouse.com" style="border: 2px solid #fff; color: #fff; text-decoration: none; padding: 13px 40px; border-radius: 50px; font-weight: bold; font-size: 18px;">EMAIL US DIRECTLY</a>
    </div>
    <p style="margin-top: 25px; color: #4cd137; font-weight: bold;">Contact us for your tailored prefab solution</p>
</div>
            """

            # 提示词：要求 AI 按特定标记输出，方便切割
            sys_p = f"""
            Role: SEO Specialist for {cinfo['name']}. 
            Task: Generate content for CMS backend.
            Target Keyword: "{target_kw if target_kw else 'Auto-detect'}"
            
            CRITICAL INSTRUCTIONS:
            1. Translate & Optimize Content (EEAT).
            2. Strictly follow the output format below using separators.
            3. **Alt Text Rule**: Must include target keyword.
            4. **HTML Body**: Include H2/H3, Table, Images. 
               - H2 Style: style="border-left:5px solid {cinfo['color']}; padding-left:10px;"
            5. **FAQ Data**: Extract 3 pairs of Q&A separately.
            6. **Schema**: JSON-LD for {cinfo['type']}.
            
            OUTPUT FORMAT (Do not change separators):
            
            |||TITLE|||
            (Insert SEO Title here, Max 60 chars)
            
            |||SLUG|||
            (Insert URL slug here, lowercase hyphens)
            
            |||KEYWORDS|||
            (Insert 5-8 comma separated keywords)
            
            |||DESCRIPTION|||
            (Insert SEO Description, Max 160 chars)
            
            |||CONTENT|||
            (Insert full HTML body code here. Do NOT include <html> or <body> tags, just the inner content.)
            
            |||FAQ_LIST|||
            Q1: ...
            A1: ...
            Q2: ...
            A2: ...
            Q3: ...
            A3: ...
            
            |||SCHEMA|||
            (Insert JSON-LD code)
            """
            
            with st.spinner(f"正在构建 {cinfo['name']} 专属数据包..."):
                try:
                    # AI 调用
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

                    # ==========================================
                    # 解析 AI 返回的数据，并按 CMS 顺序展示
                    # ==========================================
                    try:
                        # 简单的文本切割提取
                        p_title = final_res.split("|||TITLE|||")[1].split("|||")[0].strip()
                        p_slug = final_res.split("|||SLUG|||")[1].split("|||")[0].strip()
                        p_kws = final_res.split("|||KEYWORDS|||")[1].split("|||")[0].strip()
                        p_desc = final_res.split("|||DESCRIPTION|||")[1].split("|||")[0].strip()
                        p_content = final_res.split("|||CONTENT|||")[1].split("|||")[0].strip()
                        p_faq = final_res.split("|||FAQ_LIST|||")[1].split("|||")[0].strip()
                        p_schema = final_res.split("|||SCHEMA|||")[1].strip()
                        
                        # 处理 HTML (Wellucky 拼接)
                        if cinfo['name'] == "Wellucky":
                            p_content += wellucky_cta_html
                            # 把 Schema 也拼接到 HTML 末尾，防止用户忘贴
                            p_content += f"\n\n<!-- SEO Schema -->\n<script type=\"application/ld+json\">\n{p_schema}\n</script>"
                        else:
                            p_content += f"\n\n<!-- SEO Schema -->\n<script type=\"application/ld+json\">\n{p_schema}\n</script>"

                        # ==========================================
                        # 界面渲染：严格按照截图顺序
                        # ==========================================
                        st.success("✅ 生成成功！请按照下方顺序依次复制到后台：")
                        
                        st.markdown("### 1. 基础字段")
                        c_t, c_s = st.columns([2, 1])
                        c_t.text_input("📋 1. 主题 (Title)", value=p_title)
                        c_s.text_input("🔗 2. 自定义URL", value=p_slug)
                        
                        st.markdown("### 2. SEO 字段")
                        st.text_input("🔑 3. 关键字", value=p_kws)
                        st.text_area("📝 4 & 5. 描述 / 摘要", value=p_desc, height=100)
                        
                        st.markdown("### 3. 内容编辑器")
                        st.info("💡 请点击编辑器左上角的 [HTML] 按钮，粘贴下方代码：")
                        with st.expander("📄 6. 内容 (HTML + Schema + CTA)", expanded=True):
                            st.code(p_content, language="html")
                        
                        st.markdown("### 4. 底部 FAQ 模块 (如有)")
                        with st.expander("❓ 7. FAQ 问答对 (如有独立输入框)", expanded=False):
                            st.text(p_faq)

                    except Exception as parse_e:
                        st.error("解析格式略有偏差，显示原始内容，请手动复制：")
                        st.code(final_res)

                except Exception as e: st.error(f"Error: {str(e)}")
