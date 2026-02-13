import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import MultiModalConversation, ImageSynthesis
from zhipuai import ZhipuAI
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io, base64, uuid, re, os, requests

# ==========================================
# 0. 初始化与页面配置
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V28.1", layout="wide", page_icon="🦁")

# 状态初始化
if 'results_tab1' not in st.session_state: st.session_state.results_tab1 = []
if 'generated_bg' not in st.session_state: st.session_state.generated_bg = None

# ==========================================
# 1. 核心配置与工具函数
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    ALI_API_KEY = st.secrets.get("ALI_API_KEY", "")
    ZHIPU_API_KEY = st.secrets.get("ZHIPU_API_KEY", "")
except:
    GOOGLE_API_KEY = ALI_API_KEY = ZHIPU_API_KEY = ""

BIZ_CONFIG = {
    "logistics": {"name": "VastLog", "website": "www.vastlog.com", "color": "#FF9900"},
    "house": {"name": "Wellucky", "website": "www.wellucky.com", "color": "#0066CC"}
}

# --- 字体加载辅助函数 (解决无法调整大小问题) ---
def get_font(size):
    # 尝试加载常见字体，Streamlit Cloud 通常有 DejaVuSans
    possible_fonts = ["DejaVuSans-Bold.ttf", "arial.ttf", "Roboto-Bold.ttf"]
    for f in possible_fonts:
        try:
            return ImageFont.truetype(f, size)
        except:
            continue
    return ImageFont.load_default() # 如果都失败，回退到默认（不可调大小）

# --- 核心：文件名清洗 (保留 V27 逻辑) ---
def get_clean_seo_name(ai_res, brand):
    if not ai_res or "Error" in ai_res: return f"{brand.lower()}-item-{uuid.uuid4().hex[:4]}"
    name = ai_res.lower()
    name = re.sub(r'[^a-z0-9]', ' ', name)
    stop_words = {'this', 'appears', 'to', 'be', 'an', 'a', 'the', 'is', 'of', 'view', 'image', 'photo', 'picture'}
    words = [w for w in name.split() if len(w) > 2 and w not in stop_words]
    brand_low = brand.lower()
    if brand_low in words: words.remove(brand_low)
    words.insert(0, brand_low)
    return "-".join(words[:6])

def convert_to_webp(image):
    buf = io.BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(buf, format='WEBP', quality=80)
    return buf.getvalue()

def run_ai_vision(engine, img, prompt, key, model):
    if not key: return "Error: No Key"
    try:
        if engine == "google":
            genai.configure(api_key=key)
            m = genai.GenerativeModel(model)
            res = m.generate_content([prompt, img])
            return res.text
        elif engine == "ali":
            dashscope.api_key = key
            tmp_p = f"v_{uuid.uuid4().hex}.png"; img.save(tmp_p)
            url = f"file://{os.path.abspath(tmp_p).replace('\\', '/')}"
            res = MultiModalConversation.call(model=model, messages=[{"role":"user","content":[{"image":url},{"text":prompt}]}])
            if os.path.exists(tmp_p): os.remove(tmp_p)
            return res.output.choices[0].message.content[0]['text']
        else: # 智谱
            client = ZhipuAI(api_key=key)
            # 略...为了代码简洁，逻辑同上
            return "Zhipu Logic Placeholder"
    except Exception as e: return f"Error: {str(e)}"

# ==========================================
# 2. 侧边栏配置
# ==========================================
with st.sidebar:
    st.title("⚙️ 核心设置")
    biz_choice = st.radio("🏢 业务模式", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"))
    cbiz = "logistics" if "VastLog" in biz_choice else "house"
    cinfo = BIZ_CONFIG[cbiz]
    
    st.divider()
    engine_choice = st.radio("🧠 AI 引擎", ("Google Gemini", "阿里通义"))
    if "Google" in engine_choice:
        etype, mlist, ekey = "google", ["gemini-1.5-flash"], GOOGLE_API_KEY
    else:
        etype, mlist, ekey = "ali", ["qwen-vl-max"], ALI_API_KEY
    sel_mod = st.selectbox("模型版本", mlist)

# ==========================================
# 3. 主功能区
# ==========================================
st.markdown(f"### 🦁 {cinfo['name']} 数字化运营 V28.1")
tab1, tab2, tab3 = st.tabs(["✍️ Tab 1: 智能文案", "🎨 Tab 2: 封面工厂", "🌍 Tab 3: GEO 专家"])

# ----------------------------------------------------------------
# Tab 1: 智能文案 (功能已恢复：仅重命名 vs 全套)
# ----------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        files = st.file_uploader("📂 上传图片 (批量)", accept_multiple_files=True, key="t1_up")
    with c2:
        draft = st.text_area("📝 文案重点 (仅全套模式生效)", placeholder="例：美国DDP专线，时效15天...")
        
        # --- 恢复两个独立按钮 ---
        b1, b2 = st.columns(2)
        btn_rename = b1.button("🖼️ 仅识图起名 (SEO)", use_container_width=True)
        btn_full = b2.button("🚀 全套处理 (含贴文)", type="primary", use_container_width=True)

    if (btn_rename or btn_full) and files:
        st.session_state.results_tab1 = [] # 清空旧数据
        
        prompt_seo = f"Identify product in image. Output format: {cinfo['name']}-keyword-keyword. No sentences."
        prompt_copy = f"Write a professional Facebook post for {cinfo['name']}. Context: {draft}."

        progress = st.progress(0)
        for i, f in enumerate(files):
            img = Image.open(f)
            
            # 1. 必做：识图起名
            raw_name = run_ai_vision(etype, img, prompt_seo, ekey, sel_mod)
            clean_name = get_clean_seo_name(raw_name, cinfo['name'])
            
            # 2. 选做：文案生成
            copy_text = ""
            if btn_full:
                copy_text = run_ai_vision(etype, img, prompt_copy, ekey, sel_mod)
            
            st.session_state.results_tab1.append({
                "img": img, "name": f"{clean_name}.webp", "text": copy_text, "data": convert_to_webp(img)
            })
            progress.progress((i+1)/len(files))

    # 结果展示
    if st.session_state.results_tab1:
        st.divider()
        for res in st.session_state.results_tab1:
            lc, rc = st.columns([1, 3])
            lc.image(res['img'], use_container_width=True)
            with rc:
                st.code(res['name'], language="bash")
                if res['text']:
                    st.text_area("FB Copy", res['text'], height=100)
                st.download_button("下载 WebP", res['data'], file_name=res['name'])

# ----------------------------------------------------------------
# Tab 2: 封面工厂 (功能已恢复：AI生图 + 3标题独立控制)
# ----------------------------------------------------------------
with tab2:
    st.caption("功能：AI 生成背景 或 上传背景 + 3个独立标题控制")
    
    # --- A. 背景来源 ---
    bg_col1, bg_col2 = st.columns([1, 1])
    with bg_col1:
        st.markdown("#### A. 背景来源")
        bg_mode = st.radio("选择模式", ["上传本地图片", "AI 文生图 (Wanx)"], horizontal=True)
        
        bg_image = None
        
        if bg_mode == "上传本地图片":
            bg_file = st.file_uploader("上传背景图", type=['jpg', 'png', 'webp'])
            if bg_file: bg_image = Image.open(bg_file).convert("RGBA")
            
        else: # AI 生图
            ai_prompt = st.text_input("输入画面描述 (例如: container ship at sunset)", value="futuristic container ship on ocean")
            if st.button("🎨 生成背景图"):
                if not ALI_API_KEY:
                    st.error("需要配置阿里 API Key")
                else:
                    try:
                        with st.spinner("AI 正在绘图..."):
                            dashscope.api_key = ALI_API_KEY
                            rsp = ImageSynthesis.call(model=ImageSynthesis.Models.wanx_v1, prompt=ai_prompt, n=1, size='1024*1024')
                            if rsp.status_code == 200:
                                img_url = rsp.output.results[0].url
                                # 下载图片
                                bg_content = requests.get(img_url).content
                                st.session_state.generated_bg = Image.open(io.BytesIO(bg_content)).convert("RGBA")
                            else:
                                st.error(f"生图失败: {rsp.message}")
                    except Exception as e: st.error(str(e))
            
            if st.session_state.generated_bg:
                bg_image = st.session_state.generated_bg
                st.success("AI 背景图已就绪")

    # --- B. 3个独立标题控制 (恢复需求) ---
    with bg_col2:
        st.markdown("#### B. 文字图层控制")
        
        # 标题 1
        with st.expander("标题 1 (主标题)", expanded=True):
            t1_text = st.text_input("内容", "VastLog Global")
            c1_a, c1_b, c1_c = st.columns(3)
            t1_size = c1_a.number_input("大小", 20, 200, 80, key="s1")
            t1_color = c1_b.color_picker("颜色", "#FFFFFF", key="c1")
            t1_y = c1_c.slider("垂直位置 Y", 0, 1000, 100, key="y1")

        # 标题 2
        with st.expander("标题 2 (副标题)"):
            t2_text = st.text_input("内容", "DDP Shipping", key="txt2")
            c2_a, c2_b, c2_c = st.columns(3)
            t2_size = c2_a.number_input("大小", 20, 200, 50, key="s2")
            t2_color = c2_b.color_picker("颜色", cinfo['color'], key="c2") # 默认品牌色
            t2_y = c2_c.slider("垂直位置 Y", 0, 1000, 250, key="y2")

        # 标题 3
        with st.expander("标题 3 (装饰/角标)"):
            t3_text = st.text_input("内容", "FAST & SAFE", key="txt3")
            c3_a, c3_b, c3_c = st.columns(3)
            t3_size = c3_a.number_input("大小", 20, 200, 30, key="s3")
            t3_color = c3_b.color_picker("颜色", "#FFFF00", key="c3")
            t3_y = c3_c.slider("垂直位置 Y", 0, 1000, 350, key="y3")

    # --- C. 合成逻辑 ---
    if bg_image:
        st.divider()
        st.markdown("#### C. 最终合成预览")
        
        # 创建画布
        final_img = bg_image.copy()
        draw = ImageDraw.Draw(final_img)
        W, H = final_img.size
        
        # 简单的阴影效果偏移量
        shadow_offset = 3
        
        # 绘制函数
        def draw_text(text, size, color, y_pos):
            if not text: return
            font = get_font(int(size))
            # 计算居中 X
            try:
                # Pillow >= 10.0
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
            except:
                # 旧版 Pillow
                text_w = draw.textlength(text, font=font)
            
            x_pos = (W - text_w) / 2
            
            # 绘制阴影 (黑色)
            draw.text((x_pos + shadow_offset, y_pos + shadow_offset), text, font=font, fill="#000000")
            # 绘制正文
            draw.text((x_pos, y_pos), text, font=font, fill=color)

        draw_text(t1_text, t1_size, t1_color, t1_y)
        draw_text(t2_text, t2_size, t2_color, t2_y)
        draw_text(t3_text, t3_size, t3_color, t3_y)

        # 展示
        st.image(final_img, use_container_width=True)
        
        # 下载
        buf = io.BytesIO()
        final_img.convert("RGB").save(buf, format="JPEG", quality=95)
        st.download_button("📥 下载最终封面", buf.getvalue(), file_name=f"cover-{cinfo['name']}.jpg")

# ----------------------------------------------------------------
# Tab 3: GEO 专家 (独立输入，互不影响)
# ----------------------------------------------------------------
with tab3:
    st.caption("功能：生成符合 EEAT 标准的 HTML/JSON-LD 代码")
    
    t3_txt = st.text_area("输入产品/服务详情", height=150, placeholder="支持直接粘贴 Tab 1 的结果，或手动输入...")
    
    if st.button("生成 GEO 代码"):
        if not t3_txt:
            st.warning("请先输入内容")
        else:
            prompt_geo = f"""
            You are an SEO Expert for {cinfo['name']}.
            Input: {t3_txt}
            Requirements:
            1. Generate HTML Article with <h2 style="border-left: 5px solid {cinfo['color']}; padding-left: 10px;">Title</h2>.
            2. Generate JSON-LD schema for {cbiz} business.
            """
            
            # 这里简单用 Gemini 演示
            if etype == "google":
                genai.configure(api_key=ekey)
                model = genai.GenerativeModel("gemini-1.5-flash")
                res = model.generate_content(prompt_geo).text
                
                c_code, c_view = st.columns(2)
                with c_code:
                    st.code(res, language="html")
                with c_view:
                    st.markdown(res, unsafe_allow_html=True)
            else:
                st.info("Demo模式：请在 Tab 3 使用 Google 引擎以获得最佳 SEO 效果")
