import streamlit as st
import base64
import json
import re
import google.generativeai as genai
from openai import OpenAI
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="内容优化流水线",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Clean light CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #f8f9fb;
    color: #1a1a2e;
}

.block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 4rem; }

.section-label {
    font-size: 12px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
    margin-top: 28px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e5e7eb;
}

.info-tip {
    background: #f0f4ff;
    border-left: 3px solid #6366f1;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    font-size: 12px;
    color: #4b5563;
    margin-top: 8px;
    line-height: 1.6;
}

.stButton > button {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.stButton > button:hover { background: #4f46e5 !important; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    background: #ffffff !important;
    font-size: 14px !important;
}

.stSelectbox > div > div {
    border-radius: 8px !important;
    background: #ffffff !important;
}

.stProgress > div > div { background: #6366f1 !important; }

.stTabs [data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 500;
    color: #6b7280;
}

.stTabs [aria-selected="true"] { color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Password gate ─────────────────────────────────────────────────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return True
    st.markdown("### 🔐 请输入访问密码")
    pwd = st.text_input("密码", type="password")
    if st.button("进入"):
        if pwd == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密码错误")
    return False

if not check_password():
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ 内容优化流水线")
st.markdown("<p style='color:#6b7280;font-size:14px;margin-top:-8px;margin-bottom:8px;'>图片识别（Gemini）· 翻译优化（通义千问）· SEO/GEO · 生成HTML</p>", unsafe_allow_html=True)
st.divider()

# ── API helpers ───────────────────────────────────────────────────────────────
def get_gemini_model():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-1.5-flash")

def get_qianwen_client():
    return OpenAI(
        api_key=st.secrets["QIANWEN_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def clean_json(text):
    return re.sub(r"```json|```", "", text).strip()

def identify_image_gemini(img_file, page_name, brand_name, lang_key):
    model = get_gemini_model()
    img_file.seek(0)
    pil_img = Image.open(img_file)
    img_file.seek(0)
    resp = model.generate_content([
        f"""SEO expert. Analyze image. Respond ONLY with JSON (no markdown):
{{
  "filename": "seo-filename-using-{page_name or 'image'}-prefix-max-5-words-lowercase-hyphens-no-extension",
  "alt": "descriptive alt text in {'English' if lang_key == 'en' else lang_key}, under 125 chars, factual",
  "description": "one sentence factual description of image content"
}}
Brand: {brand_name or 'N/A'}. Do NOT exaggerate.""",
        pil_img
    ])
    return clean_json(resp.text)

def call_qianwen(prompt, max_tokens=4000):
    client = get_qianwen_client()
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

# ── STEP 1: Basic settings ────────────────────────────────────────────────────
st.markdown('<div class="section-label">① 基本设置</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    page_type = st.selectbox("页面类型", ["产品页 (Product Page)", "文章页 (Article Page)"])
    page_type_key = "product" if "产品" in page_type else "article"
with col2:
    target_lang = st.selectbox("目标语言", ["英文 English", "西班牙语 Spanish", "德语 German"])
    lang_key = {"英文 English": "en", "西班牙语 Spanish": "es", "德语 German": "de"}[target_lang]

col3, col4 = st.columns(2)
with col3:
    page_name = st.text_input("页面/产品名称（用于图片重命名和SEO）", placeholder="e.g. 20ft-shipping-container-home")
with col4:
    brand_name = st.text_input("品牌名称", placeholder="e.g. ContainerLife")

keywords_raw = st.text_input("目标关键词（逗号分隔）", placeholder="e.g. container home, modular home")
keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

# ── STEP 2: Text content ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">② 文字内容</div>', unsafe_allow_html=True)

text_content = st.text_area("粘贴中文原文", height=180,
    placeholder="将业务员提供的文字粘贴到这里...\n支持标题、段落、产品参数等各种格式")

doc_file = st.file_uploader("或上传文件（Word / Excel / TXT）",
    type=["docx", "xlsx", "xls", "doc", "txt"])

if doc_file and doc_file.name.endswith(".txt"):
    text_content = doc_file.read().decode("utf-8", errors="ignore")
    st.success(f"✓ 已读取: {doc_file.name}")
elif doc_file:
    st.info(f"已上传: {doc_file.name}（Word/Excel内容请同时在上方粘贴文字）")

st.markdown("**优化选项**")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: opt_seo = st.checkbox("SEO优化", value=True)
with c2: opt_geo = st.checkbox("GEO/AIO优化", value=True)
with c3: opt_structure = st.checkbox("语义化结构", value=True)
with c4: opt_faq = st.checkbox("生成FAQ", value=True)
with c5: opt_schema = st.checkbox("Schema Microdata", value=False)

# ── STEP 3: Images ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">③ 图片素材</div>', unsafe_allow_html=True)

image_files = st.file_uploader("上传图片（可多选）",
    type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

if image_files:
    cols = st.columns(min(len(image_files), 5))
    for i, f in enumerate(image_files):
        with cols[i % 5]:
            st.image(f, use_container_width=True)
            st.caption(f.name)

st.markdown('<div class="info-tip">📷 图片由 <b>Gemini</b> 识别内容，自动生成SEO文件名和alt文字。文字翻译和优化由 <b>通义千问</b> 处理。</div>', unsafe_allow_html=True)

# ── RUN ───────────────────────────────────────────────────────────────────────
st.markdown("")
run_btn = st.button("⚡ 开始处理", use_container_width=True, type="primary")

if run_btn:
    if not text_content:
        st.error("请先输入或粘贴文字内容")
        st.stop()

    progress = st.progress(0)
    log_area = st.empty()
    logs = []

    def log(msg):
        logs.append(msg)
        log_area.markdown("  \n".join(f"<small style='color:#6b7280'>• {l}</small>" for l in logs[-6:]),
                          unsafe_allow_html=True)

    # Image identification
    image_results = []
    if image_files:
        log(f"Gemini 识别图片中（共 {len(image_files)} 张）...")
        for i, img_file in enumerate(image_files):
            try:
                raw = identify_image_gemini(img_file, page_name, brand_name, lang_key)
                parsed = json.loads(raw)
                image_results.append({
                    "original_name": img_file.name,
                    "new_name": parsed.get("filename", f"{page_name or 'image'}-{i+1}") + ".webp",
                    "alt": parsed.get("alt", f"{page_name or 'image'} {i+1}"),
                    "description": parsed.get("description", ""),
                    "placeholder": f"IMAGE_PLACEHOLDER_{i+1}",
                    "file": img_file
                })
                log(f"✓ 图片 {i+1}/{len(image_files)} 识别完成")
            except Exception as e:
                image_results.append({
                    "original_name": img_file.name,
                    "new_name": f"{page_name or 'image'}-{i+1}.webp",
                    "alt": f"{page_name or 'image'} {i+1}",
                    "description": "",
                    "placeholder": f"IMAGE_PLACEHOLDER_{i+1}",
                    "file": img_file
                })
                log(f"⚠ 图片 {i+1} 识别失败，使用默认命名")
            progress.progress(int((i+1) / len(image_files) * 35))

    # Text optimization
    progress.progress(40)
    log("通义千问 翻译 + SEO/GEO优化中...")

    img_desc = "\n".join([
        f"Image {i+1}: filename=\"{r['new_name']}\", alt=\"{r['alt']}\", description=\"{r['description']}\""
        for i, r in enumerate(image_results)
    ]) or "No images"

    schema_instr = (
        f"Add HTML Microdata schema attributes inline for {'Product' if page_type_key == 'product' else 'Article'} schema (no JSON-LD)."
        if opt_schema else "Do NOT add any Schema markup."
    )
    faq_instr = (
        'Add a FAQ section: <div class="faq-section"><h2>Frequently Asked Questions</h2>... with 3-5 Q&As targeting long-tail keywords.'
        if opt_faq else ""
    )
    geo_instr = (
        "GEO/AIO: Write authoritatively with direct answers. Key facts in first 1-2 sentences of each paragraph."
        if opt_geo else ""
    )

    prompt = f"""You are an expert SEO/GEO content optimizer and HTML developer.

TASK: Transform the Chinese content below into optimized HTML for a {page_type_key} page.

CRITICAL RULES:
- Preserve ALL original information — do NOT remove, fabricate, or exaggerate any facts
- Only improve wording and clarity; meaning must be 100% faithful to the original

REQUIREMENTS:
1. Translate to {'English' if lang_key == 'en' else lang_key}
2. {"Semantic HTML: one H1, H2, H3, p, ul, ol, strong tags" if opt_structure else "Basic paragraph structure"}
3. {"SEO: naturally include keywords: " + ', '.join(keywords) if opt_seo and keywords else ""}
4. {geo_instr}
5. {schema_instr}
6. {faq_instr}
7. Insert image placeholders [[IMAGE_N_URL]] at logical positions:
   <figure>
     <img src="[[IMAGE_N_URL]]" alt="ALT_TEXT" loading="lazy" width="800" height="600">
     <figcaption>CAPTION</figcaption>
   </figure>
8. Output ONLY inner HTML (no <html><head><body> tags)
9. Brand: {brand_name or 'N/A'}

IMAGES ({len(image_results)} total):
{img_desc}

ORIGINAL CONTENT:
{text_content}"""

    try:
        html_output = call_qianwen(prompt)
        html_output = re.sub(r"```html|```", "", html_output).strip()

        for r in image_results:
            n = r["placeholder"].replace("IMAGE_PLACEHOLDER_", "")
            html_output = html_output.replace(f"[[IMAGE_{n}_URL]]", r["placeholder"])

        progress.progress(80)
        log("✓ 内容优化完成，生成SEO摘要...")

        seo_raw = call_qianwen(f"""Generate SEO metadata. Respond ONLY with JSON (no markdown):
{{
  "title": "SEO title under 60 chars",
  "meta_description": "meta description under 155 chars",
  "url_slug": "seo-url-slug",
  "og_title": "Open Graph title",
  "og_description": "OG description",
  "primary_keyword": "main keyword",
  "secondary_keywords": ["kw1", "kw2", "kw3"],
  "geo_summary": "1-2 sentence direct answer for AI search engines"
}}
HTML: {html_output[:2000]}""", max_tokens=800)

        seo = json.loads(clean_json(seo_raw))

        progress.progress(100)
        log("✅ 全部完成！")

        st.session_state["html_output"] = html_output
        st.session_state["image_results"] = image_results
        st.session_state["seo"] = seo
        st.rerun()

    except Exception as e:
        st.error(f"处理出错: {e}")

# ── OUTPUT ────────────────────────────────────────────────────────────────────
if "html_output" in st.session_state:
    html_output = st.session_state["html_output"]
    image_results = st.session_state["image_results"]
    seo = st.session_state["seo"]

    st.divider()
    st.success("✅ 处理完成！")

    tab1, tab2, tab3, tab4 = st.tabs(["📄 HTML代码", "🖼️ 图片对照表", "🔍 SEO Meta", "🔗 URL & 摘要"])

    with tab1:
        st.caption("复制以下代码，粘贴到CMS源代码编辑框")
        st.code(html_output, language="html")
        st.download_button("⬇ 下载HTML", html_output,
            file_name=f"{seo.get('url_slug', 'output')}.html", mime="text/html")

    with tab2:
        if image_results:
            st.caption("上传图片到网站后台后，填入链接，点击「更新HTML」")
            updated_html = html_output
            for i, img in enumerate(image_results):
                c1, c2, c3 = st.columns([1, 2, 3])
                with c1:
                    img["file"].seek(0)
                    st.image(img["file"], width=70)
                with c2:
                    st.markdown(f"`{img['placeholder']}`")
                    st.caption(f"文件名: {img['new_name']}")
                    st.caption(f"Alt: {img['alt']}")
                with c3:
                    url = st.text_input("图片URL", key=f"url_{i}",
                        placeholder="https://yoursite.com/images/...")
                    if url:
                        updated_html = updated_html.replace(img["placeholder"], url)
            if st.button("🔄 更新图片链接"):
                st.session_state["html_output"] = updated_html
                st.success("✓ 已更新，切换到「HTML代码」查看")
                st.rerun()
        else:
            st.info("本次未上传图片")

    with tab3:
        st.code(f"""<!-- 复制到页面 <head> 中 -->
<title>{seo.get('title', '')}</title>
<meta name="description" content="{seo.get('meta_description', '')}">
<meta property="og:title" content="{seo.get('og_title', '')}">
<meta property="og:description" content="{seo.get('og_description', '')}">
<meta property="og:type" content="{'product' if 'page_type_key' in dir() and page_type_key == 'product' else 'article'}">""",
        language="html")

    with tab4:
        st.markdown("**URL Slug**")
        st.code(seo.get("url_slug", ""), language="text")
        st.markdown("**Meta Description**")
        st.info(seo.get("meta_description", ""))
        st.markdown("**GEO/AIO摘要**（适合放页面第一段，AI搜索引擎容易引用）")
        st.success(seo.get("geo_summary", ""))
        st.markdown("**主要关键词：** `" + seo.get("primary_keyword", "") + "`")
        if seo.get("secondary_keywords"):
            st.markdown("**次要关键词：** " + "  ·  ".join(seo.get("secondary_keywords", [])))
