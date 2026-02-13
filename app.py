import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import ImageSynthesis, MultiModalConversation
from zhipuai import ZhipuAI
from PIL import Image, ImageDraw, ImageFont
import io, base64, re, os, requests, uuid

# ==========================================
# 0. 全局配置与初始化
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V28.4", layout="wide", page_icon="🦁")

# 初始化 Session State
if 'results_tab1' not in st.session_state: st.session_state.results_tab1 = []
if 'generated_bg' not in st.session_state: st.session_state.generated_bg = None

# API Key 获取 (优先读 Secrets，否则留空)
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    ALI_API_KEY = st.secrets.get("ALI_API_KEY", "")
    ZHIPU_API_KEY = st.secrets.get("ZHIPU_API_KEY", "")
except:
    GOOGLE_API_KEY = ALI_API_KEY = ZHIPU_API_KEY = ""

# 业务配置
BIZ_CONFIG = {
    "logistics": {"name": "VastLog", "website": "www.vastlog.com", "color": "#FF9900", "type": "LogisticsService"},
    "house": {"name": "Wellucky", "website": "www.wellucky.com", "color": "#0066CC", "type": "Product"}
}

# ==========================================
# 1. 核心工具函数
# ==========================================
def get_font(size):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except: return ImageFont.load_default()

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def convert_to_webp(image):
    buf = io.BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(buf, format='WEBP', quality=80)
    return buf.getvalue()

def get_clean_seo_name(ai_res, brand):
    if not ai_res or "Error" in ai_res: return f"{brand.lower()}-item-{uuid.uuid4().hex[:4]}"
    name = ai_res.lower()
    name = re.sub(r'[^a-z0-9]', ' ', name)
    words = [w for w in name.split() if len(w) > 2 and w not in {'this','image','photo','view'}]
    brand_low = brand.lower()
    if brand_low in words: words.remove(brand_low)
    words.insert(0, brand_low)
    return "-".join(words[:6])

# --- 统一 AI 调用接口 (核心修复) ---
def run_ai_vision(engine, img, prompt, key, model_name):
    """统一处理 Google/Ali/Zhipu 的识图请求"""
    if not key: return "Error: 缺少 API Key"
    try:
        if engine == "Google Gemini":
            genai.configure(api_key=key)
            m = genai.GenerativeModel(model_name)
            return m.generate_content([prompt, img]).text
            
        elif engine == "智谱清言":
            client = ZhipuAI(api_key=key)
            # 智谱 4V 调用格式
            response = client.chat.completions.create(
                model=model_name, # e.g., glm-4v
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": pil_to_base64(img)}}
                    ]
                }]
            )
            return response.choices[0].message.content
            
        elif engine == "阿里通义":
            dashscope.api_key = key
            # 阿里需要本地路径或URL，这里用临时文件
            tmp = f"temp_{uuid.uuid4()}.png"
            img.save(tmp)
            local_uri = f"file://{os.path.abspath(tmp)}"
            try:
                msgs = [{"role": "user", "content": [{"image": local_uri}, {"text": prompt}]}]
                res = MultiModalConversation.call(model=model_name, messages=msgs)
                content = res.output.choices[0].message.content
                # 阿里返回结构有时不同，提取 text
                if isinstance(content, list): return content[0]['text']
                return content
            finally:
                if os.path.exists(tmp): os.remove(tmp)
        return "Error: Unknown Engine"
    except Exception as e:
        return f"API Error: {str(e)}"

# ==========================================
# 2. 侧边栏 (功能完全恢复)
# ==========================================
with st.sidebar:
    st.title("⚙️ 系统配置 V28.4")
    
    # 2.1 业务切换
    st.markdown("### 🏢 业务线")
    biz_choice = st.radio("选择业务", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"), label_visibility="collapsed")
    cbiz = "logistics" if "VastLog" in biz_choice else "house"
    cinfo = BIZ_CONFIG[cbiz]

    st.divider()

    # 2.2 引擎与模型切换 (恢复智谱)
    st.markdown("### 🧠 AI 引擎")
    engine_choice = st.radio("厂商", ("Google Gemini", "智谱清言", "阿里通义"))
    
    # 动态显示默认模型，支持手动修改
    if engine_choice == "Google Gemini":
        st.caption("推荐: gemini-2.0-flash, gemini-1.5-pro")
        sel_model = st.text_input("模型名称", value="gemini-2.0-flash")
        api_key = GOOGLE_API_KEY
    elif engine_choice == "智谱清言":
        st.caption("推荐: glm-4v (识图), glm-4 (文本)")
        sel_model = st.text_input("模型名称", value="glm-4v")
        api_key = ZHIPU_API_KEY
    else:
        st.caption("推荐: qwen-vl-max, qwen-vl-plus")
        sel_model = st.text_input("模型名称", value="qwen-vl-max")
        api_key = ALI_API_KEY

# ==========================================
# 3. 主界面
# ==========================================
st.markdown(f"### 🦁 {cinfo['name']} 数字化运营台 <span style='font-size:0.8rem; color:grey'>{engine_choice} / {sel_model}</span>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["✍️ Tab 1: 智能文案", "🎨 Tab 2: 封面工厂", "🌍 Tab 3: GEO/EEAT 专家"])

# ----------------------------------------------------------------
# Tab 1: 智能文案 (全模型兼容)
# ----------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1, 1])
    files_t1 = c1.file_uploader("📂 批量上传产品图", accept_multiple_files=True, key="t1_up")
    with c2:
        draft = st.text_area("文案补充信息 (仅全套模式)", height=100, placeholder="例：美国专线，时效15天...")
        b1, b2 = st.columns(2)
        btn_rename = b1.button("🖼️ 仅识图起名", use_container_width=True)
        btn_full = b2.button("🚀 全套处理", type="primary", use_container_width=True)

    if (btn_rename or btn_full) and files_t1:
        st.session_state.results_tab1 = []
        
        prompt_seo = f"Identify this product. Output format: {cinfo['name']}-keyword-keyword. Rules: Lowercase, use hyphens, max 3 keywords. No sentences."
        prompt_copy = f"Write a Facebook post for {cinfo['name']}. Context: {draft}. Tone: Professional."

        bar = st.progress(0)
        for i, f in enumerate(files_t1):
            img = Image.open(f)
            
            # 1. 识图起名 (调用统一接口)
            raw_name = run_ai_vision(engine_choice, img, prompt_seo, api_key, sel_model)
            clean_name = get_clean_seo_name(raw_name, cinfo['name'])
            
            # 2. 文案 (仅全套)
            copy_text = ""
            if btn_full:
                copy_text = run_ai_vision(engine_choice, img, prompt_copy, api_key, sel_model)
            
            st.session_state.results_tab1.append({
                "img": img, "name": f"{clean_name}.webp", "text": copy_text, "data": convert_to_webp(img)
            })
            bar.progress((i+1)/len(files_t1))

    # 展示结果
    if st.session_state.results_tab1:
        st.divider()
        for res in st.session_state.results_tab1:
            col_l, col_r = st.columns([1, 3])
            col_l.image(res['img'], width=150)
            with col_r:
                st.code(res['name'], language="bash")
                if res['text']: st.text_area("Copywriting", res['text'], height=80)
                st.download_button("下载 WebP", res['data'], file_name=res['name'])

# ----------------------------------------------------------------
# Tab 2: 封面工厂 (生图+文字控制)
# ----------------------------------------------------------------
with tab2:
    # A. 背景
    col_bg, col_txt = st.columns([1, 1])
    with col_bg:
        st.markdown("#### A. 背景层")
        bg_mode = st.radio("来源", ["本地上传", "AI 生图 (阿里Wanx)"], horizontal=True)
        bg_image = None
        
        if bg_mode == "本地上传":
            f = st.file_uploader("上传背景", key="t2_up")
            if f: bg_image = Image.open(f).convert("RGBA")
        else:
            p_draw = st.text_input("画面描述", "container ship at sunset, 4k")
            if st.button("生成背景"):
                if not ALI_API_KEY: st.error("请配置 ALI_API_KEY")
                else:
                    try:
                        dashscope.api_key = ALI_API_KEY
                        rsp = ImageSynthesis.call(model=ImageSynthesis.Models.wanx_v1, prompt=p_draw, n=1, size='1024*1024')
                        if rsp.status_code == 200:
                            img_url = rsp.output.results[0].url
                            st.session_state.generated_bg = Image.open(io.BytesIO(requests.get(img_url).content)).convert("RGBA")
                        else: st.error(f"失败: {rsp.message}")
                    except Exception as e: st.error(str(e))
            if st.session_state.generated_bg: bg_image = st.session_state.generated_bg

    # B. 文字
    with col_txt:
        st.markdown("#### B. 文字层 (3组)")
        # 标题1
        with st.expander("标题 1 (主)", expanded=True):
            t1_t = st.text_input("内容1", "Global Service")
            c1, c2, c3 = st.columns(3)
            t1_s = c1.number_input("大小1", 20, 300, 80)
            t1_c = c2.color_picker("颜色1", "#FFFFFF")
            t1_y = c3.slider("Y轴1", 0, 1000, 100)
        # 标题2
        with st.expander("标题 2 (副)"):
            t2_t = st.text_input("内容2", "DDP Shipping")
            c1, c2, c3 = st.columns(3)
            t2_s = c1.number_input("大小2", 20, 300, 50)
            t2_c = c2.color_picker("颜色2", cinfo['color'])
            t2_y = c3.slider("Y轴2", 0, 1000, 250)
        # 标题3
        with st.expander("标题 3 (饰)"):
            t3_t = st.text_input("内容3", "Fast & Safe")
            c1, c2, c3 = st.columns(3)
            t3_s = c1.number_input("大小3", 20, 300, 30)
            t3_c = c2.color_picker("颜色3", "#FFFF00")
            t3_y = c3.slider("Y轴3", 0, 1000, 350)

    # C. 合成
    if bg_image:
        st.divider()
        final_img = bg_image.copy()
        draw = ImageDraw.Draw(final_img)
        W, H = final_img.size
        
        def draw_layer(txt, size, color, y):
            if not txt: return
            f = get_font(int(size))
            try: w = draw.textlength(txt, font=f)
            except: w = draw.textbbox((0,0), txt, font=f)[2]
            x = (W - w) / 2
            draw.text((x+4, y+4), txt, font=f, fill="black") # 阴影
            draw.text((x, y), txt, font=f, fill=color)
        
        draw_layer(t1_t, t1_s, t1_c, t1_y)
        draw_layer(t2_t, t2_s, t2_c, t2_y)
        draw_layer(t3_t, t3_s, t3_c, t3_y)
        
        st.image(final_img, use_container_width=True)
        buf = io.BytesIO()
        final_img.convert("RGB").save(buf, format="JPEG", quality=95)
        st.download_button("📥 下载封面图", buf.getvalue(), "cover.jpg")

# ----------------------------------------------------------------
# Tab 3: GEO 专家 (支持 中译英 + EEAT + 插图)
# ----------------------------------------------------------------
with tab3:
    st.caption(f"当前引擎: {engine_choice} | 任务：中文 -> 英文 SEO 文章 + 自动插图")
    
    col_in, col_up = st.columns([1, 1])
    with col_in:
        cn_text = st.text_area("粘贴中文原文", height=250, placeholder="此处输入中文内容...")
    with col_up:
        t3_imgs = st.file_uploader("上传配图 (自动插入文章)", accept_multiple_files=True, key="t3_imgs")
        if engine_choice != "Google Gemini" and t3_imgs and len(t3_imgs) > 1:
            st.warning("⚠️ 注意：智谱/阿里对单次多图插入 HTML 的支持可能不如 Google Gemini 稳定。建议 Tab 3 优先使用 Google。")

    if st.button("✨ 生成 GEO/EEAT 代码", type="primary"):
        if not cn_text:
            st.warning("请输入中文！")
        else:
            # 构建 Prompt
            sys_prompt = f"""
            Role: SEO Expert for {cinfo['name']} ({cinfo['type']}).
            Task: Translate CHINESE input to ENGLISH. Keep original meaning intact.
            Format: HTML Article with EEAT standards.
            Styles: <h2 style="border-left:5px solid {cinfo['color']}; padding-left:10px;">Title</h2>
            Schema: Add <script type="application/ld+json"> for {cinfo['name']}.
            Image Rules: Insert <img src="filename" alt="SEO description" style="width:100%; margin:20px 0;"> tags where appropriate.
            """
            
            final_res = ""
            
            with st.spinner(f"正在使用 {engine_choice} 处理..."):
                try:
                    # 分逻辑处理
                    if engine_choice == "Google Gemini":
                        # Google 支持 Text + List[Images]
                        content = [sys_prompt, f"Input Text:\n{cn_text}"]
                        if t3_imgs:
                            content.append("\nAvailable Images:")
                            for img_f in t3_imgs:
                                content.append(f"\nFilename: {img_f.name}")
                                content.append(Image.open(img_f)) # 传入 PIL 对象
                        
                        genai.configure(api_key=GOOGLE_API_KEY)
                        m = genai.GenerativeModel(sel_model)
                        final_res = m.generate_content(content).text

                    elif engine_choice == "智谱清言":
                        # 智谱 V4 目前主要针对单图，这里做纯文本+Schema处理，图片可能需简化
                        # 策略：只发文本，让它生成 img 占位符
                        client = ZhipuAI(api_key=ZHIPU_API_KEY)
                        img_note = ""
                        if t3_imgs:
                            img_names = [f.name for f in t3_imgs]
                            img_note = f"\nAvailable Image Filenames to insert: {', '.join(img_names)}"
                        
                        msgs = [{"role":"user", "content": sys_prompt + img_note + f"\n\nText to Translate:\n{cn_text}"}]
                        resp = client.chat.completions.create(model=sel_model, messages=msgs)
                        final_res = resp.choices[0].message.content

                    else: # 阿里
                        # 同智谱逻辑，优先处理文本
                        dashscope.api_key = ALI_API_KEY
                        img_note = ""
                        if t3_imgs: img_note = f"\nImages: {', '.join([f.name for f in t3_imgs])}"
                        msgs = [{"role":"user", "content": sys_prompt + img_note + f"\n\nContent:\n{cn_text}"}]
                        # 注意：阿里文本模型通常是 qwen-turbo/max，这里如果用户选了 vl 模型传纯文本也可以兼容
                        res = MultiModalConversation.call(model=sel_model, messages=msgs)
                        final_res = res.output.choices[0].message.content[0]['text']

                    # 结果展示
                    st.success("✅ 生成完成")
                    c_view, c_code = st.columns([1, 1])
                    with c_view:
                        st.markdown("### 👁️ 预览")
                        st.markdown(final_res, unsafe_allow_html=True)
                    with c_code:
                        st.markdown("### 💻 代码")
                        st.code(final_res, language="html")

                except Exception as e:
                    st.error(f"处理失败: {str(e)}")
