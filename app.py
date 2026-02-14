import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import ImageSynthesis, MultiModalConversation
from zhipuai import ZhipuAI
from PIL import Image, ImageDraw, ImageFont
import io, base64, re, os, requests, uuid, zipfile # 新增 zipfile

# ==========================================
# 0. 全局配置 & 初始化
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V29.6", layout="wide", page_icon="🦁")

if 'results_tab1' not in st.session_state: st.session_state.results_tab1 = []
if 'generated_bg' not in st.session_state: st.session_state.generated_bg = None

def get_secret_safe(key_name, default=""):
    try: return st.secrets.get(key_name, default)
    except: return default

GOOGLE_API_KEY = get_secret_safe("GOOGLE_API_KEY")
ALI_API_KEY = get_secret_safe("ALI_API_KEY")
ZHIPU_API_KEY = get_secret_safe("ZHIPU_API_KEY")

BIZ_CONFIG = {
    "logistics": {"name": "VastLog", "website": "www.vastlog.com", "color": "#FF9900", "type": "LogisticsService", "keywords": ["logistics", "shipping", "freight", "cargo", "DDP"]},
    "house": {"name": "Wellucky", "website": "www.wellucky.com", "color": "#0066CC", "type": "Product", "keywords": ["container house", "modular home", "prefab", "steel structure"]}
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
    st.title("⚙️ 配置 V29.6")
    
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
        
        prompt_seo = f"""
        Role: SEO Expert for {cinfo['name']}.
        Task: Create a UNIQUE filename for this image.
        Keywords to use: {kw_str}.
        CRITICAL RULES:
        1. Analyze specific visual details: Color? Angle? Interior/Exterior? Roof style?
        2. Format: {cinfo['name'].lower()}-feature-detail-keyword.
        3. DO NOT just output '{cinfo['name'].lower()}-container-house'. be specific!
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

    # --- 批量操作区 (核心升级) ---
    if st.session_state.results_tab1:
        st.divider()
        st.markdown("### 🛠️ 批量操作区")
        
        # 定义两列：批量下载 和 清空
        col_down, col_clear = st.columns([1, 1])
        
        with col_down:
            # 创建 ZIP 压缩包
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for res in st.session_state.results_tab1:
                    # 将每张图片写入 ZIP
                    zf.writestr(res['name'], res['data'])
            
            st.download_button(
                label=f"📦 批量下载 {len(st.session_state.results_tab1)} 张图片 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{cinfo['name'].lower()}-batch-images.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )
            st.caption("⚠️ 浏览器安全限制：批量下载必须打包为 ZIP 格式，否则会被拦截。")

        with col_clear:
            if st.button("🗑️ 清空当前列表 (开始下一批)", use_container_width=True):
                st.session_state.results_tab1 = []
                st.rerun()

        st.divider()
        st.markdown("### 🖼️ 结果预览")
        
        # 单图展示
        for i, res in enumerate(st.session_state.results_tab1):
            l, r = st.columns([1, 3])
            l.image(res['img'], width=150)
            with r:
                unique_key = f"{i}_{uuid.uuid4()}"
                st.text_input("SEO文件名", res['name'], key=f"name_{unique_key}")
                if res['text']: st.text_area("文案", res['text'], height=80, key=f"txt_{unique_key}")
                
                # 依然保留单图下载功能
                st.download_button("⬇️ 单图下载", res['data'], res['name'], key=f"dl_{unique_key}")

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
    # 更新标题，明确告知包含的功能
    st.caption(f"当前引擎: {engine_choice} | 模型: {sel_model}")
    st.markdown("##### 🛡️ 功能：中译英 + EEAT 润色 + Schema 结构化数据 + 自动插图")
    
    cc1, cc2 = st.columns([1, 1])
    with cc1: 
        cn_txt = st.text_area("中文原文 / 产品参数", height=300, placeholder="在此输入中文内容...")
    with cc2: 
        imgs = st.file_uploader("文章配图 (AI会自动插入HTML)", accept_multiple_files=True, key="t3_imgs")
        if imgs: st.info(f"已加载 {len(imgs)} 张图片，将根据上下文自动插入文章。")

    # 按钮文案也改得更直观
    if st.button("✨ 生成全套 SEO 代码 (HTML + Schema)", type="primary", use_container_width=True):
        if not cn_txt: 
            st.warning("⚠️ 请先输入中文原文！")
        else:
            # ====================================================
            # 核心提示词 (Prompt) - 这里就是您关心的 EEAT 和 Schema 指令
            # ====================================================
            sys_p = f"""
            Role: Senior Google SEO Expert for {cinfo['name']} ({cinfo['website']}).
            
            Mission:
            1. TRANSLATE the user's Chinese text to English.
            2. EEAT OPTIMIZATION: Ensure the tone is Professional, Authoritative, and Trustworthy. No 'Chinglish'.
            3. FORMATTING: Output a complete HTML Article. 
               - Use <h2> tags with this specific style: style="border-left:5px solid {cinfo['color']}; padding-left:10px; color:#333;"
               - Use <p> tags for paragraphs.
            4. SCHEMA.ORG (Crucial): 
               - Append a valid <script type="application/ld+json"> block at the end.
               - Schema Type: "{cinfo['type']}" (Product or Service).
               - Brand Name: "{cinfo['name']}".
            5. IMAGES: 
               - I will provide filenames. Insert <img src="filename" alt="SEO Optimized Description" style="width:100%; border-radius:8px; margin:20px 0;"> tags naturally into the content where they fit best.
            
            Output ONLY the HTML code.
            """
            
            with st.spinner("SEO 专家正在进行 EEAT 优化和 Schema 编写..."):
                try:
                    final_html = ""
                    
                    # 1. Google Gemini (多模态处理)
                    if engine_choice == "Google Gemini":
                        cnt = [sys_p, f"Input Text:\n{cn_txt}"]
                        if imgs:
                            cnt.append("\nAvailable Image Files to Insert:")
                            for f in imgs: cnt.extend([f"\nFilename: {f.name}", Image.open(f)])
                        genai.configure(api_key=api_key)
                        final_html = genai.GenerativeModel(sel_model).generate_content(cnt).text

                    # 2. 智谱/阿里 (文本处理)
                    else:
                        img_note = f"\nAvailable Image filenames: {', '.join([f.name for f in imgs])}" if imgs else ""
                        full_p = sys_p + img_note + f"\n\nSource Text to Translate:\n{cn_txt}"
                        
                        if engine_choice == "智谱清言":
                            client = ZhipuAI(api_key=api_key)
                            # 智谱翻译建议用 glm-4-plus
                            t_model = "glm-4-plus" 
                            resp = client.chat.completions.create(model=t_model, messages=[{"role":"user","content":full_p}])
                            final_html = resp.choices[0].message.content
                        else:
                            # 阿里翻译建议用 qwen-max
                            resp = Generation.call(model='qwen-max', messages=[{"role":"user","content":full_p}])
                            final_html = resp.output.text

                    # 展示结果
                    st.success("✅ SEO 代码生成完毕！包含 EEAT 优化与 JSON-LD Schema。")
                    v, c = st.columns([1, 1])
                    with v:
                        st.markdown("### 👁️ 网页预览")
                        st.caption("注：图片在网站后台上传后才会显示，此处仅预览排版")
                        st.markdown(final_html, unsafe_allow_html=True)
                    with c:
                        st.markdown("### 💻 HTML 源代码 (直接复制)")
                        st.code(final_html, language="html")
                        
                except Exception as e: 
                    st.error(f"生成出错: {str(e)}")
