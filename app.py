import streamlit as st
import anthropic
import base64
import json
import re
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="内容优化流水线",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Password gate ─────────────────────────────────────────────────────────────
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("## 🔐 请输入访问密码")
    pwd = st.text_input("密码", type="password")
    if st.button("进入"):
        correct = st.secrets.get("APP_PASSWORD", "admin123")
        if pwd == correct:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密码错误")
    return False

if not check_password():
    st.stop()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.main { background: #0c0c0f; }
.block-container { max-width: 960px; padding-top: 2rem; }

h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; }
h3 { font-size: 1rem !important; }

.badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #4fc8a0;
    border: 1px solid #4fc8a0;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

.step-card {
    background: #14141a;
    border: 1px solid #2a2a38;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.step-header {
    font-size: 15px;
    font-weight: 600;
    color: #e8e8f0;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.step-num {
    background: #7c6af7;
    color: white;
    width: 24px; height: 24px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
}

.info-box {
    background: rgba(124,106,247,0.08);
    border: 1px solid rgba(124,106,247,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #8888a8;
    margin-top: 8px;
}

.output-code {
    background: #14141a;
    border: 1px solid #2a2a38;
    border-radius: 8px;
    padding: 16px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    line-height: 1.7;
    color: #e8e8f0;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 500px;
    overflow-y: auto;
}

.stButton > button {
    background: #7c6af7 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #9080ff !important;
    transform: translateY(-1px) !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background: #1c1c26 !important;
    border: 1px solid #2a2a38 !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
}

.stProgress > div > div { background: linear-gradient(90deg, #7c6af7, #4fc8a0) !important; }

.img-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
.img-card {
    background: #1c1c26;
    border: 1px solid #2a2a38;
    border-radius: 8px;
    padding: 8px;
    width: 150px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    color: #8888a8;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="badge">CONTENT PIPELINE v1.0</div>', unsafe_allow_html=True)
st.markdown("# 内容优化 <span style='color:#7c6af7'>流水线</span>", unsafe_allow_html=True)
st.markdown("<p style='color:#8888a8;font-size:14px;margin-top:-8px;'>图片处理 · 翻译 · SEO/GEO优化 · 生成可直接复制的HTML代码</p>", unsafe_allow_html=True)

st.markdown("---")

# ── Claude client ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ── Helpers ───────────────────────────────────────────────────────────────────
def image_to_base64(uploaded_file):
    bytes_data = uploaded_file.read()
    uploaded_file.seek(0)
    return base64.standard_b64encode(bytes_data).decode("utf-8")

def get_media_type(filename):
    ext = filename.lower().split(".")[-1]
    return {"png": "image/png", "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")

def clean_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return text

# ── STEP 1: Basic settings ────────────────────────────────────────────────────
st.markdown('<div class="step-header"><span class="step-num">1</span> 基本设置</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    page_type = st.selectbox("页面类型", ["产品页 (Product Page)", "文章页 (Article Page)"])
    page_type_key = "product" if "产品" in page_type else "article"
with col2:
    target_lang = st.selectbox("目标语言", ["英文 English", "西班牙语 Spanish", "德语 German"])
    lang_map = {"英文 English": "en", "西班牙语 Spanish": "es", "德语 German": "de"}
    lang_key = lang_map[target_lang]

col3, col4 = st.columns(2)
with col3:
    page_name = st.text_input("页面/产品名称（用于图片重命名和SEO）", placeholder="e.g. 20ft-shipping-container-home")
with col4:
    brand_name = st.text_input("品牌名称", placeholder="e.g. ContainerLife")

keywords_raw = st.text_input("目标关键词（用逗号分隔）", placeholder="e.g. container home, shipping container house, modular home")
keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

st.markdown("---")

# ── STEP 2: Text content ──────────────────────────────────────────────────────
st.markdown('<div class="step-header"><span class="step-num">2</span> 文字内容</div>', unsafe_allow_html=True)

text_content = st.text_area(
    "直接粘贴内容（中文原文）",
    height=200,
    placeholder="将业务员提供的文字内容粘贴到这里...\n\n支持标题、段落、产品参数等各种格式"
)

doc_file = st.file_uploader(
    "或上传文件（Word / Excel）",
    type=["docx", "xlsx", "xls", "doc", "txt"],
    help="上传后文字内容将从文件中提取"
)

if doc_file:
    st.success(f"✓ 已上传: {doc_file.name}")
    if doc_file.name.endswith(".txt"):
        text_content = doc_file.read().decode("utf-8", errors="ignore")
        st.info("已读取文本文件内容")

st.markdown("**优化选项**")
col_a, col_b, col_c, col_d, col_e = st.columns(5)
with col_a: opt_seo = st.checkbox("SEO优化", value=True)
with col_b: opt_geo = st.checkbox("GEO/AIO优化", value=True)
with col_c: opt_structure = st.checkbox("语义化结构", value=True)
with col_d: opt_faq = st.checkbox("生成FAQ", value=True)
with col_e: opt_schema = st.checkbox("Schema Microdata", value=False)

st.markdown("---")

# ── STEP 3: Images ────────────────────────────────────────────────────────────
st.markdown('<div class="step-header"><span class="step-num">3</span> 图片素材（AI自动识别内容 + 生成alt文字）</div>', unsafe_allow_html=True)

image_files = st.file_uploader(
    "上传图片（可多选）",
    type=["jpg", "jpeg", "png", "webp", "gif"],
    accept_multiple_files=True
)

if image_files:
    cols = st.columns(min(len(image_files), 5))
    for i, img_file in enumerate(image_files):
        with cols[i % 5]:
            st.image(img_file, use_container_width=True)
            st.caption(img_file.name)

st.markdown('<div class="info-box">图片处理：①AI识别图片内容 ②生成SEO友好文件名 ③生成alt描述。处理完后图片占位符会显示在输出HTML中，你填入真实URL后点"更新链接"即可。</div>', unsafe_allow_html=True)

st.markdown("---")

# ── RUN ───────────────────────────────────────────────────────────────────────
run_btn = st.button("⚡ 开始处理", use_container_width=True)

if run_btn:
    if not text_content and not doc_file:
        st.error("请先输入文字内容或上传文件")
        st.stop()

    if not page_name:
        st.warning("建议填写页面名称，用于图片重命名")

    client = get_client()
    progress = st.progress(0)
    status = st.empty()
    log_area = st.empty()
    logs = []

    def log(msg):
        logs.append(msg)
        log_area.markdown("\n\n".join(f"`{l}`" for l in logs[-5:]))

    # ── Process images ────────────────────────────────────────────────────────
    image_results = []

    if image_files:
        log(f"⏳ 正在识别 {len(image_files)} 张图片...")
        progress.progress(10)

        for i, img_file in enumerate(image_files):
            log(f"⏳ 识别图片 {i+1}/{len(image_files)}: {img_file.name}")
            try:
                b64 = image_to_base64(img_file)
                media_type = get_media_type(img_file.name)

                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                            {"type": "text", "text": f"""You are an SEO expert. Analyze this image and respond ONLY with JSON (no markdown):
{{
  "filename": "seo-friendly-filename-no-extension-using-{page_name or 'image'}-as-prefix-max-5-words-lowercase-hyphens",
  "alt": "descriptive alt text in {'English' if lang_key == 'en' else lang_key} for SEO, under 125 chars",
  "description": "one sentence about what this image shows"
}}
Brand: {brand_name or 'N/A'}. Page topic: {page_name or 'product'}.
IMPORTANT: Keep original facts only, do not exaggerate."""}
                        ]
                    }]
                )

                raw = resp.content[0].text
                raw = clean_json(raw)
                parsed = json.loads(raw)

                image_results.append({
                    "original_name": img_file.name,
                    "new_name": parsed.get("filename", f"{page_name}-image-{i+1}") + ".webp",
                    "alt": parsed.get("alt", f"{page_name} image {i+1}"),
                    "description": parsed.get("description", ""),
                    "placeholder": f"IMAGE_PLACEHOLDER_{i+1}",
                    "file": img_file
                })
                log(f"✓ 图片 {i+1} 识别完成: {parsed.get('filename', '')}")

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

            progress.progress(10 + int((i+1) / len(image_files) * 30))

        log(f"✓ 全部图片识别完成")

    # ── Text optimization ─────────────────────────────────────────────────────
    progress.progress(45)
    log("⏳ 正在翻译并进行SEO/GEO优化...")

    img_descriptions = "\n".join([
        f"Image {i+1}: filename=\"{r['new_name']}\", alt=\"{r['alt']}\", description=\"{r['description']}\""
        for i, r in enumerate(image_results)
    ]) or "No images"

    schema_instruction = (
        f"Add HTML Microdata schema attributes (itemscope, itemtype, itemprop) inline for {'Product' if page_type_key == 'product' else 'Article'} schema. Do NOT use JSON-LD."
        if opt_schema else "Do NOT add any Schema markup."
    )

    faq_instruction = (
        "Add a FAQ section at the end with 3-5 relevant Q&As. Use <div class=\"faq-section\"><h2>Frequently Asked Questions</h2> structure. Target long-tail keywords."
        if opt_faq else ""
    )

    geo_instruction = (
        "GEO/AIO optimization: Write in clear, authoritative, direct-answer style. Include concise definitions and facts. Structure so key facts appear in first 1-2 sentences of paragraphs."
        if opt_geo else ""
    )

    prompt = f"""You are an expert SEO/GEO content optimizer and HTML developer.

TASK: Transform the Chinese content below into optimized HTML for a {page_type_key} page.

CRITICAL RULES:
- Preserve ALL original information — do NOT remove, fabricate, or exaggerate facts
- Only improve wording, flow, and clarity
- Keep the meaning 100% faithful to the original

REQUIREMENTS:
1. Translate to {'English' if lang_key == 'en' else lang_key}
2. {f"Use semantic HTML: H1 (one only), H2, H3, p, ul, ol, strong tags" if opt_structure else "Use basic paragraph structure"}
3. {f"SEO: Naturally include keywords: {', '.join(keywords) or page_name}. Use in headings and naturally throughout." if opt_seo else ""}
4. {geo_instruction}
5. {schema_instruction}
6. {faq_instruction}
7. Insert image placeholders as [[IMAGE_N_URL]] at logical positions. Format:
   <figure>
     <img src="[[IMAGE_N_URL]]" alt="ALT_TEXT" loading="lazy" width="800" height="600">
     <figcaption>CAPTION</figcaption>
   </figure>
8. Output ONLY inner HTML content (no <html>, <head>, <body> tags)
9. Brand: {brand_name or 'N/A'}

IMAGES ({len(image_results)} total):
{img_descriptions}

ORIGINAL CONTENT:
{text_content}"""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        html_output = resp.content[0].text
        html_output = re.sub(r"```html|```", "", html_output).strip()

        # Replace [[IMAGE_N_URL]] with placeholders
        for r in image_results:
            n = r["placeholder"].replace("IMAGE_PLACEHOLDER_", "")
            html_output = html_output.replace(f"[[IMAGE_{n}_URL]]", r["placeholder"])

        progress.progress(80)
        log("✓ 内容优化完成")

        # ── SEO metadata ──────────────────────────────────────────────────────
        log("⏳ 正在生成SEO摘要...")

        seo_resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Generate SEO metadata. Respond ONLY with JSON (no markdown):
{{
  "title": "SEO title under 60 chars",
  "meta_description": "meta description under 155 chars",
  "url_slug": "seo-friendly-url-slug",
  "og_title": "Open Graph title",
  "og_description": "OG description under 200 chars",
  "primary_keyword": "main keyword",
  "secondary_keywords": ["kw1", "kw2", "kw3"],
  "geo_summary": "1-2 sentence direct answer for AI search engines"
}}

HTML: {html_output[:2000]}"""
            }]
        )

        seo_raw = clean_json(seo_resp.content[0].text)
        seo = json.loads(seo_raw)

        progress.progress(100)
        log("✓ 全部完成！")
        status.success("✅ 处理完成！")

        # ── Store results ──────────────────────────────────────────────────────
        st.session_state["html_output"] = html_output
        st.session_state["image_results"] = image_results
        st.session_state["seo"] = seo

    except Exception as e:
        st.error(f"处理出错: {e}")
        st.stop()

# ── OUTPUT ────────────────────────────────────────────────────────────────────
if "html_output" in st.session_state:
    html_output = st.session_state["html_output"]
    image_results = st.session_state["image_results"]
    seo = st.session_state["seo"]

    st.markdown("---")
    st.markdown("## ✅ 处理完成 — 输出结果")

    tab1, tab2, tab3, tab4 = st.tabs(["📄 HTML代码", "🖼️ 图片对照表", "🔍 SEO摘要", "🔗 URL & Meta"])

    with tab1:
        st.markdown("**复制以下代码，粘贴到CMS源代码编辑框**")
        st.code(html_output, language="html")
        st.download_button("⬇ 下载HTML文件", html_output, file_name=f"{seo.get('url_slug', 'output')}.html", mime="text/html")

    with tab2:
        if image_results:
            st.markdown("**上传图片到网站后台后，将链接填入对应栏，然后点击「更新HTML」**")

            updated_html = html_output
            for i, img in enumerate(image_results):
                col1, col2, col3 = st.columns([1, 2, 3])
                with col1:
                    img.get("file").seek(0)
                    st.image(img["file"], width=80)
                with col2:
                    st.markdown(f"**{img['placeholder']}**")
                    st.caption(f"建议文件名: `{img['new_name']}`")
                    st.caption(f"Alt: {img['alt']}")
                with col3:
                    url = st.text_input(f"图片URL", key=f"url_{i}", placeholder="https://yoursite.com/images/...")
                    if url:
                        updated_html = updated_html.replace(img["placeholder"], url)

            if st.button("🔄 更新HTML中的图片链接"):
                st.session_state["html_output"] = updated_html
                st.success("✓ 链接已更新，切换到「HTML代码」标签查看最新版本")
                st.rerun()
        else:
            st.info("本次未上传图片")

    with tab3:
        seo_meta = f"""<!-- SEO Meta Tags — 复制到页面 <head> 中 -->

<title>{seo.get('title', '')}</title>
<meta name="description" content="{seo.get('meta_description', '')}">

<!-- Open Graph -->
<meta property="og:title" content="{seo.get('og_title', '')}">
<meta property="og:description" content="{seo.get('og_description', '')}">
<meta property="og:type" content="{'product' if page_type_key == 'product' else 'article'}">

<!-- GEO/AIO摘要（可放在页面最顶部段落） -->
<!-- {seo.get('geo_summary', '')} -->

<!-- Keywords -->
<!-- Primary: {seo.get('primary_keyword', '')} -->
<!-- Secondary: {', '.join(seo.get('secondary_keywords', []))} -->"""

        st.code(seo_meta, language="html")

    with tab4:
        st.markdown("**页面URL Slug（建议）**")
        st.code(seo.get("url_slug", ""), language="text")

        st.markdown("**Meta Description**")
        st.info(seo.get("meta_description", ""))

        st.markdown("**GEO/AIO摘要**（适合放在页面第一段，AI搜索引擎容易引用）")
        st.success(seo.get("geo_summary", ""))

        st.markdown("**主要关键词**")
        st.code(seo.get("primary_keyword", ""), language="text")

        if seo.get("secondary_keywords"):
            st.markdown("**次要关键词**")
            st.write(" · ".join(seo.get("secondary_keywords", [])))
