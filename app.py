import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import MultiModalConversation 
from zhipuai import ZhipuAI
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io, base64, uuid, re, os, json

# ==========================================
# 0. 初始化与页面设置
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V28.0", layout="wide", page_icon="🦁")

# 初始化 session_state 防止报错
if 'results' not in st.session_state: st.session_state.results = []
if 'cover_img' not in st.session_state: st.session_state.cover_img = None

# ==========================================
# 1. 核心配置与工具函数
# ==========================================
# 请确保 secrets.toml 已配置，或者在此处直接填入 Key 用于测试
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    ALI_API_KEY = st.secrets.get("ALI_API_KEY", "")
    ZHIPU_API_KEY = st.secrets.get("ZHIPU_API_KEY", "")
except:
    GOOGLE_API_KEY = ALI_API_KEY = ZHIPU_API_KEY = ""

BIZ_CONFIG = {
    "logistics": {
        "name": "VastLog", "website": "www.vastlog.com", 
        "color": "#FF9900", "type": "LogisticsService",
        "keywords": "Sea Freight, Air Freight, DDP Shipping"
    },
    "house": {
        "name": "Wellucky", "website": "www.wellucky.com", 
        "color": "#0066CC", "type": "Product",
        "keywords": "Prefab House, Steel Structure, Modular Home"
    }
}

# --- 图片转 Base64 ---
def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- 图片转 WebP ---
def convert_to_webp(image):
    buf = io.BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(buf, format='WEBP', quality=80)
    return buf.getvalue()

# --- 清洗文件名 (保留 V27 逻辑) ---
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

# --- AI 调用通用接口 ---
def run_ai_vision(engine, img, prompt, key, model):
    if not key: return "Error: API Key Missing"
    try:
        if engine == "google":
            genai.configure(api_key=key)
            m = genai.GenerativeModel(model)
            res = m.generate_content([prompt, img])
            return res.text
        elif engine == "ali":
            dashscope.api_key = key
            # 阿里需要临时文件路径
            tmp_p = f"v_{uuid.uuid4().hex}.png"; img.save(tmp_p)
            url = f"file://{os.path.abspath(tmp_p).replace('\\', '/')}"
            res = MultiModalConversation.call(model=model, messages=[{"role":"user","content":[{"image":url},{"text":prompt}]}])
            if os.path.exists(tmp_p): os.remove(tmp_p)
            return res.output.choices[0].message.content[0]['text']
        else: # 智谱
            client = ZhipuAI(api_key=key)
            res = client.chat.completions.create(
                model="glm-4v",
                messages=[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":pil_to_base64(img)}}]}]
            )
            return res.choices[0].message.content
    except Exception as e: return f"Error: {str(e)}"

# --- AI 纯文本调用 (用于 Tab 3) ---
def run_ai_text(engine, prompt, key, model):
    if not key: return "Error: API Key Missing"
    try:
        # 简化处理，统一用 Vision 模型处理纯文本也可以，或者根据引擎分流
        if engine == "google":
            genai.configure(api_key=key)
            m = genai.GenerativeModel(model)
            return m.generate_content(prompt).text
        # ... 其他引擎省略，默认用 Google 做 SEO 文字处理最强 ...
        return "Error: Currently only Google supported for Text Mode in this demo."
    except Exception as e: return f"Error: {str(e)}"

# ==========================================
# 2. 侧边栏配置
# ==========================================
with st.sidebar:
    st.title("⚙️ V28.0 控制台")
    
    # 业务切换
    biz_choice = st.radio("🏢 业务模式", ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"))
    cbiz = "logistics" if "VastLog" in biz_choice else "house"
    cinfo = BIZ_CONFIG[cbiz]
    
    st.divider()
    
    # 引擎选择
    engine_choice = st.radio("🧠 AI 引擎", ("Google Gemini", "阿里通义", "智谱清言"))
    if "Google" in engine_choice:
        etype, mlist, ekey = "google", ["gemini-1.5-flash", "gemini-1.5-pro"], GOOGLE_API_KEY
    elif "阿里" in engine_choice:
        etype, mlist, ekey = "ali", ["qwen-vl-max", "qwen-vl-plus"], ALI_API_KEY
    else:
        etype, mlist, ekey = "zhipu", ["glm-4v"], ZHIPU_API_KEY
    
    sel_mod = st.selectbox("模型版本", mlist)

# ==========================================
# 3. 主界面逻辑
# ==========================================
st.markdown(f"### {cinfo['name']} 数字化运营工作台 <span style='font-size:0.6em;color:gray'>V28.0 Independent Edition</span>", unsafe_allow_html=True)

# 定义三个独立的 Tab
tab1, tab2, tab3 = st.tabs(["📸 Tab 1: 智能文案 & SEO", "🎨 Tab 2: 封面工厂", "🌏 Tab 3: GEO/EEAT 专家"])

# ----------------------------------------------------------------
# Tab 1: 智能文案 (保留核心功能)
# ----------------------------------------------------------------
with tab1:
    st.caption("功能：图片 SEO 命名清洗 + 社交媒体贴文生成")
    col1, col2 = st.columns([1, 1])
    with col1:
        files = st.file_uploader("上传原始素材", accept_multiple_files=True, key="tab1_uploader")
    with col2:
        draft = st.text_area("补充信息 (如：尺寸、材质、航线)", placeholder="例如：20ft 预制舱 / 上海到洛杉矶DDP")
        run_btn = st.button("🚀 开始分析 (Tab 1)", type="primary")

    if run_btn and files:
        st.session_state.results = [] # 清空旧结果
        prompt_seo = f"Context: {cinfo['name']} ({cinfo['keywords']}). Task: Identify the product in the image. Output: Just 3-4 english keywords connected by hyphens. No sentences."
        
        prompt_social = f"""
        Role: Senior Social Media Manager for {cinfo['name']}.
        Task: Write a Facebook post about this image.
        Context: {draft}.
        Tone: Professional, Trustworthy.
        Requirement: Include bullet points and emojis.
        """

        progress_bar = st.progress(0)
        for i, f in enumerate(files):
            img = Image.open(f)
            # 1. 识别命名
            raw_name = run_ai_vision(etype, img, prompt_seo, ekey, sel_mod)
            clean_name = get_clean_seo_name(raw_name, cinfo['name'])
            # 2. 生成文案
            copywriting = run_ai_vision(etype, img, prompt_social, ekey, sel_mod)
            
            st.session_state.results.append({
                "img": img, 
                "name": f"{clean_name}.webp", 
                "copy": copywriting,
                "data": convert_to_webp(img)
            })
            progress_bar.progress((i + 1) / len(files))

    # 结果展示
    if st.session_state.results:
        st.divider()
        for res in st.session_state.results:
            c1, c2 = st.columns([1, 2])
            c1.image(res['img'], caption=res['name'], use_container_width=True)
            with c2:
                st.code(res['name'], language="bash")
                st.text_area("FB Copy", res['copy'], height=150)
                st.download_button("下载 WebP", res['data'], file_name=res['name'])

# ----------------------------------------------------------------
# Tab 2: 封面工厂 (独立模块)
# ----------------------------------------------------------------
with tab2:
    st.caption(f"功能：为 {cinfo['name']} 生成带品牌规范的封面图 (独立运行)")
    
    # 2.1 独立上传入口
    t2_col1, t2_col2 = st.columns([1, 1])
    img_file_t2 = t2_col1.file_uploader("上传背景图", type=["png", "jpg", "jpeg"], key="tab2_uploader")
    
    # 2.2 编辑控件
    with t2_col2:
        title_text = st.text_input("主标题 (H1)", value="New Arrival" if cbiz == "house" else "CN ✈ US")
        sub_text = st.text_input("副标题 (H2)", value="Ready to Ship" if cbiz == "house" else "5-7 Days DDP")
        overlay_color = st.color_picker("蒙版/文字颜色", cinfo['color'])
        
    # 2.3 绘图逻辑
    if img_file_t2:
        image_t2 = Image.open(img_file_t2).convert("RGBA")
        
        # 简单的绘图处理 (PIL)
        # 创建一个覆盖层
        txt_layer = Image.new("RGBA", image_t2.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)
        W, H = image_t2.size
        
        # 模拟不同业务的风格
        if cbiz == "logistics":
            # VastLog 风格：左上角彩色角标
            draw.polygon([(0,0), (W/3, 0), (0, H/3)], fill=overlay_color)
            # 这里的文字位置需要精细计算，此处仅演示逻辑
            draw.text((20, 50), title_text, fill="white", font_size=int(H/15))
        else:
            # Wellucky 风格：底部半透明黑条 + 居中文字
            draw.rectangle([(0, H*0.8), (W, H)], fill=(0, 0, 0, 150))
            draw.text((W/10, H*0.85), f"{title_text} | {sub_text}", fill="white", font_size=int(H/20))
            
        # 合并
        out_img = Image.alpha_composite(image_t2, txt_layer)
        st.image(out_img, caption="预览效果", use_container_width=True)
        
        # 下载
        buf = io.BytesIO()
        out_img.convert("RGB").save(buf, format="JPEG")
        st.download_button("下载封面图", buf.getvalue(), file_name=f"cover-{cinfo['name']}.jpg")
    else:
        st.info("👈 请先在左侧上传一张图片")

# ----------------------------------------------------------------
# Tab 3: GEO/SEO 专家 (独立模块)
# ----------------------------------------------------------------
with tab3:
    st.caption(f"功能：符合 EEAT 标准的 HTML 排版与 Schema 代码生成 (业务: {cinfo['name']})")
    
    # 3.1 独立输入
    source_text = st.text_area("输入原始文案 / 产品参数 / 物流线路详情", height=150, 
                              placeholder="粘贴刚才生成的文案，或者直接输入参数...")
    
    # 3.2 专家生成按钮
    if st.button("✨ 生成 EEAT 代码", type="primary"):
        if not source_text:
            st.warning("请先输入内容！")
        else:
            with st.spinner("SEO 专家正在排版..."):
                # 构建 Prompt
                sys_prompt = f"""
                You are an SEO Expert specializing in {cinfo['name']} ({cbiz}).
                Target: Google SEO (EEAT standards).
                
                Input Text: {source_text}
                
                Output Requirement 1 (HTML):
                - Create a structured Article.
                - Use <h2> tags with this specific style: <h2 style="border-left: 5px solid {cinfo['color']}; padding-left: 10px; color: #333;">Title</h2>
                - Content must be authoritative.
                
                Output Requirement 2 (JSON-LD):
                - Generate a valid <script type="application/ld+json"> block.
                - Schema Type: {cinfo['type']}.
                - Brand: {cinfo['name']}.
                - URL: {cinfo['website']}.
                """
                
                # 调用 AI (此处简单复用 vision 接口处理文本，或者用 run_ai_text)
                # 注意：实际生产建议用专门的 text model，这里为了演示方便用了通用的
                if etype == "google":
                    final_code = run_ai_text(etype, sys_prompt, ekey, sel_mod)
                else:
                    final_code = "Current Demo Mode supports Google Engine for Text Gen better."

                c_view, c_code = st.columns([1, 1])
                
                with c_view:
                    st.markdown("### 👁️ 预览效果")
                    # 提取 HTML 部分展示 (简单模拟)
                    st.markdown(final_code, unsafe_allow_html=True)
                
                with c_code:
                    st.markdown("### 💻 源代码 (HTML + JSON-LD)")
                    st.code(final_code, language="html")
