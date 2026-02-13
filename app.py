import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import ImageSynthesis
from PIL import Image, ImageDraw, ImageFont
import io, re, os, requests, uuid

# ==========================================
# 0. 初始化与页面配置
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V28.3", layout="wide", page_icon="🦁")

if 'results_tab1' not in st.session_state: st.session_state.results_tab1 = []
if 'generated_bg' not in st.session_state: st.session_state.generated_bg = None

# ==========================================
# 1. 核心配置与工具
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    ALI_API_KEY = st.secrets.get("ALI_API_KEY", "")
except:
    GOOGLE_API_KEY = ALI_API_KEY = ""

BIZ_CONFIG = {
    "logistics": {"name": "VastLog", "website": "www.vastlog.com", "color": "#FF9900", "type": "LogisticsService"},
    "house": {"name": "Wellucky", "website": "www.wellucky.com", "color": "#0066CC", "type": "Product"}
}

def get_font(size):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except: return ImageFont.load_default()

def convert_to_webp(image):
    buf = io.BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(buf, format='WEBP', quality=80)
    return buf.getvalue()

def get_clean_seo_name(ai_res, brand):
    if not ai_res or "Error" in ai_res: return f"{brand.lower()}-item-{uuid.uuid4().hex[:4]}"
    name = ai_res.lower()
    name = re.sub(r'[^a-z0-9]', ' ', name)
    words = [w for w in name.split() if len(w) > 2 and w not in {'this','image','photo'}]
    brand_low = brand.lower()
    if brand_low in words: words.remove(brand_low)
    words.insert(0, brand_low)
    return "-".join(words[:6])

# ==========================================
# 2. 侧边栏配置 (核心修改：模型名称可配置)
# ==========================================
with st.sidebar:
    st.title("⚙️ 核心设置 V28.3")
    
    # 业务选择
    biz_choice = st.radio("🏢 业务模式", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"))
    cbiz = "logistics" if "VastLog" in biz_choice else "house"
    cinfo = BIZ_CONFIG[cbiz]
    
    st.divider()
    
    # --- 模型配置 (修复 NotFound 问题的关键) ---
    st.markdown("### 🧠 AI 模型配置")
    st.info("如果遇到 NotFound 报错，请确认此处填写的模型名称与您 Google API 权限一致。")
    # 默认给 2.0-flash，您可以随时改为 2.5-flash 或其他
    gemini_model_name = st.text_input("Google 模型名称", value="gemini-2.0-flash") 

# ==========================================
# 3. 主功能区
# ==========================================
st.markdown(f"### 🦁 {cinfo['name']} 数字化运营 V28.3")
tab1, tab2, tab3 = st.tabs(["✍️ Tab 1: 智能文案", "🎨 Tab 2: 封面工厂", "🌍 Tab 3: GEO/EEAT 专家"])

# ----------------------------------------------------------------
# Tab 1: 智能文案
# ----------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1, 1])
    files_t1 = c1.file_uploader("📂 上传图片", accept_multiple_files=True, key="t1_up")
    with c2:
        draft = st.text_area("文案重点 (仅全套模式生效)", height=100)
        b1, b2 = st.columns(2)
        btn_rename = b1.button("🖼️ 仅识图起名", use_container_width=True)
        btn_full = b2.button("🚀 全套处理", type="primary", use_container_width=True)

    if (btn_rename or btn_full) and files_t1:
        st.session_state.results_tab1 = []
        genai.configure(api_key=GOOGLE_API_KEY)
        # 使用侧边栏配置的模型名
        try:
            model = genai.GenerativeModel(gemini_model_name)
            
            prompt_seo = f"Identify product. Output format: {cinfo['name']}-keyword-keyword. No sentences."
            bar = st.progress(0)
            
            for i, f in enumerate(files_t1):
                img = Image.open(f)
                
                # 1. 起名
                try:
                    raw_name = model.generate_content([prompt_seo, img]).text
                    clean_name = get_clean_seo_name(raw_name, cinfo['name'])
                except Exception as e:
                    clean_name = f"{cinfo['name']}-err-{uuid.uuid4().hex[:4]}"
                    st.error(f"起名失败 (图片 {i+1}): {str(e)}")

                # 2. 文案
                copy_text = ""
                if btn_full:
                    p_copy = f"Write professional FB post for {cinfo['name']}. Context: {draft}."
                    try: copy_text = model.generate_content([p_copy, img]).text
                    except: pass
                
                st.session_state.results_tab1.append({
                    "img": img, "name": f"{clean_name}.webp", "text": copy_text, "data": convert_to_webp(img)
                })
                bar.progress((i+1)/len(files_t1))
        except Exception as e:
            st.error(f"模型初始化失败，请检查模型名称: {str(e)}")

    if st.session_state.results_tab1:
        st.divider()
        for res in st.session_state.results_tab1:
            lc, rc = st.columns([1, 3])
            lc.image(res['img'], width=150)
            with rc:
                st.code(res['name'], language="bash")
                if res['text']: st.text_area("Copy", res['text'], height=80)
                st.download_button("下载 WebP", res['data'], file_name=res['name'])

# ----------------------------------------------------------------
# Tab 2: 封面工厂 (保持)
# ----------------------------------------------------------------
with tab2:
    bg_col1, bg_col2 = st.columns([1, 1])
    with bg_col1:
        st.markdown("#### A. 背景来源")
        bg_mode = st.radio("模式", ["上传图片", "AI 生图 (Wanx)"], horizontal=True)
        bg_image = None
        if bg_mode == "上传图片":
            f = st.file_uploader("背景图", key="t2_up")
            if f: bg_image = Image.open(f).convert("RGBA")
        else:
            p = st.text_input("画面描述", "container ship at sea")
            if st.button("生成背景"):
                if not ALI_API_KEY: st.error("缺阿里 Key")
                else:
                    dashscope.api_key = ALI_API_KEY
                    rsp = ImageSynthesis.call(model=ImageSynthesis.Models.wanx_v1, prompt=p, n=1, size='1024*1024')
                    if rsp.status_code==200:
                        st.session_state.generated_bg = Image.open(io.BytesIO(requests.get(rsp.output.results[0].url).content)).convert("RGBA")
            if st.session_state.generated_bg: bg_image = st.session_state.generated_bg

    with bg_col2:
        st.markdown("#### B. 标题控制")
        with st.expander("标题 1", expanded=True):
            t1_t = st.text_input("Txt1", "Global Logistics"); t1_s = st.number_input("Size1", 20,200,80); t1_c = st.color_picker("Clr1", "#FFF"); t1_y = st.slider("Y1", 0,1000,100)
        with st.expander("标题 2"):
            t2_t = st.text_input("Txt2", "DDP Service"); t2_s = st.number_input("Size2", 20,200,50); t2_c = st.color_picker("Clr2", cinfo['color']); t2_y = st.slider("Y2", 0,1000,250)
        with st.expander("标题 3"):
            t3_t = st.text_input("Txt3", "Fast & Safe"); t3_s = st.number_input("Size3", 20,200,30); t3_c = st.color_picker("Clr3", "#FF0"); t3_y = st.slider("Y3", 0,1000,350)

    if bg_image:
        st.divider()
        final_img = bg_image.copy(); draw = ImageDraw.Draw(final_img); W,H = final_img.size
        def dr(t,s,c,y):
            if not t: return
            f = get_font(int(s))
            try: w = draw.textlength(t, font=f)
            except: w = draw.textbbox((0,0),t,font=f)[2]
            x = (W-w)/2
            draw.text((x+3,y+3),t,font=f,fill="black"); draw.text((x,y),t,font=f,fill=c)
        dr(t1_t,t1_s,t1_c,t1_y); dr(t2_t,t2_s,t2_c,t2_y); dr(t3_t,t3_s,t3_c,t3_y)
        st.image(final_img, use_container_width=True)
        buf=io.BytesIO(); final_img.convert("RGB").save(buf,"JPEG"); st.download_button("下载封面", buf.getvalue(), "cover.jpg")

# ----------------------------------------------------------------
# Tab 3: GEO 专家 (核心修复：动态模型 + 中译英 + 插图)
# ----------------------------------------------------------------
with tab3:
    st.caption(f"当前使用模型: {gemini_model_name} | 功能：中文转英文 + EEAT + 自动插图")
    
    c3_in1, c3_in2 = st.columns([1, 1])
    with c3_in1:
        cn_text = st.text_area("📝 中文原文", height=300, placeholder="例如：集装箱房屋安装步骤说明...")
    with c3_in2:
        uploaded_imgs = st.file_uploader("🖼️ 文章配图", accept_multiple_files=True, key="t3_imgs")
        st.info("💡 提示：AI 将阅读这些图片，并将其插入到英文文章的逻辑位置中。")

    if st.button("✨ 生成 GEO 英文代码", type="primary"):
        if not cn_text:
            st.warning("⚠️ 请输入中文内容")
        else:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                # 使用侧边栏自定义的模型名
                model = genai.GenerativeModel(gemini_model_name)
                
                # 构建多模态 Prompt
                # 1. 系统指令
                sys_prompt = f"""
                You are a Senior Content Expert for {cinfo['name']} ({cinfo['type']}).
                Task: Translate the user's CHINESE text to English, then format it as a high-quality SEO Article.
                
                Guidelines:
                1. **Translation**: Accurate meaning, but professional/native tone. NO Chinglish.
                2. **Formatting**:
                   - Use <h2> tags: <h2 style="border-left: 5px solid {cinfo['color']}; padding-left: 10px;">Title</h2>
                   - Use <p> for text.
                3. **Images**:
                   - I have provided images. Insert them into the HTML where they make sense contextually.
                   - Format: <img src="[filename]" alt="[AI Generated Descriptive Alt Text]" style="width:100%; border-radius:8px; margin:20px 0;">
                   - Use the exact filenames of the uploaded images.
                4. **Schema**:
                   - Add <script type="application/ld+json"> at the end.
                   - Type: {cinfo['type']}. Brand: {cinfo['name']}.
                """
                
                content_parts = [sys_prompt, "\n\nInput Chinese Text:\n" + cn_text]
                
                # 2. 附加图片
                if uploaded_imgs:
                    content_parts.append("\n\nAvailable Images for Insertion:")
                    for img_f in uploaded_imgs:
                        p_img = Image.open(img_f)
                        content_parts.append(f"\nFilename: {img_f.name}")
                        content_parts.append(p_img) # 直接传入 PIL 对象
                
                with st.spinner(f"正在调用 {gemini_model_name} 进行深度处理..."):
                    response = model.generate_content(content_parts)
                    res_html = response.text
                    
                    st.success("✅ 处理完成")
                    
                    # 结果分栏
                    vc, cc = st.columns([1, 1])
                    with vc:
                        st.markdown("### 👁️ 效果预览")
                        st.markdown(res_html, unsafe_allow_html=True)
                        st.caption("*注：图片需上传到网站后台后才能正常显示")
                    with cc:
                        st.markdown("### 💻 HTML 代码")
                        st.code(res_html, language="html")

            except Exception as e:
                st.error(f"❌ 调用失败: {str(e)}")
                if "NotFound" in str(e):
                    st.warning(f"请检查左侧边栏输入的模型名称 '{gemini_model_name}' 是否正确。")
