import streamlit as st
import base64
import json
import re
import io
import zipfile
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

# ── API helpers ───────────────────────────────────────────────────────────────
def get_qianwen_client():
    return OpenAI(
        api_key=st.secrets["QIANWEN_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def get_qianwen_vl_client():
    return OpenAI(
        api_key=st.secrets["QIANWEN_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def clean_json(text):
    return re.sub(r"```json|```", "", text).strip()

def identify_image_qianwen(img_file, page_name, brand_name, lang_key):
    client = get_qianwen_vl_client()
    img_file.seek(0)
    pil_img = Image.open(img_file).convert("RGB")
    pil_img.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    img_file.seek(0)
    resp = get_qianwen_vl_client().chat.completions.create(
        model="qwen-vl-plus-latest",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": f"""You are an SEO expert. Analyze this image and respond ONLY with JSON (no markdown, no explanation):
{{
  "filename": "seo-filename-using-{page_name or 'image'}-prefix-max-5-words-lowercase-hyphens-no-extension",
  "alt": "descriptive alt text in {'English' if lang_key == 'en' else lang_key}, under 125 chars, factual description",
  "description": "one sentence factual description of what this image shows"
}}
Brand: {brand_name or 'N/A'}. Keep it factual, do NOT exaggerate."""}
            ]
        }],
        max_tokens=300
    )
    return clean_json(resp.choices[0].message.content)

def call_qianwen(prompt, max_tokens=4000):
    client = get_qianwen_client()
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ 内容优化流水线")
st.markdown("<p style='color:#6b7280;font-size:14px;margin-top:-8px;margin-bottom:8px;'>图片识别 · 翻译优化 · SEO/GEO · 生成HTML &nbsp;—&nbsp; 全程由<b>通义千问</b>处理</p>", unsafe_allow_html=True)

# ── Mode switcher ─────────────────────────────────────────────────────────────
mode = st.radio("选择模式", ["⚡ 完整流程（图片+文字→HTML）", "🖼️ 仅图片处理（重命名+转WebP）"],
    horizontal=True, label_visibility="collapsed")
st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# IMAGE-ONLY MODE
# ════════════════════════════════════════════════════════════════════════════════
if mode == "🖼️ 仅图片处理（重命名+转WebP）":

    st.markdown("### 🖼️ 图片处理")
    st.markdown("<p style='color:#6b7280;font-size:13px;margin-top:-8px;'>上传图片 → AI识别内容 → 自动重命名 → 转WebP → 下载</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        img_page_name = st.text_input("页面/产品名称（用于命名前缀）", placeholder="e.g. container-home")
    with col2:
        img_brand = st.text_input("品牌名称", placeholder="e.g. Wellucky", key="img_brand")
    img_lang = st.selectbox("Alt文字语言", ["英文 English", "西班牙语 Spanish", "德语 German"], key="img_lang")
    img_lang_key = {"英文 English": "en", "西班牙语 Spanish": "es", "德语 German": "de"}[img_lang]
    img_quality = st.slider("WebP压缩质量", 60, 100, 85, help="85为推荐值，肉眼基本无差别，文件更小")

    img_files = st.file_uploader("上传图片（可多选）",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        accept_multiple_files=True, key="img_only_files")

    if img_files:
        cols = st.columns(min(len(img_files), 5))
        for i, f in enumerate(img_files):
            with cols[i % 5]:
                st.image(f, use_container_width=True)
                st.caption(f.name)

    use_ai = st.checkbox("使用AI识别内容并命名（消耗API）", value=True,
        help="关闭后仅转换格式，文件名使用「前缀-序号」格式")

    process_btn = st.button("🚀 开始处理图片", use_container_width=True, type="primary")

    if process_btn and img_files:
        progress = st.progress(0)
        log_area = st.empty()
        logs = []

        def log_img(msg):
            logs.append(msg)
            log_area.markdown("  \n".join(f"<small style='color:#6b7280'>• {l}</small>" for l in logs[-5:]),
                              unsafe_allow_html=True)

        processed = []  # {name, webp_bytes, alt}

        for i, img_file in enumerate(img_files):
            log_img(f"处理第 {i+1}/{len(img_files)} 张: {img_file.name}")

            # Convert to WebP
            img_file.seek(0)
            pil_img = Image.open(img_file).convert("RGB")
            webp_buf = io.BytesIO()
            pil_img.save(webp_buf, "WEBP", quality=img_quality)
            webp_bytes = webp_buf.getvalue()

            # Naming
            if use_ai:
                try:
                    img_file.seek(0)
                    raw = identify_image_qianwen(img_file, img_page_name, img_brand, img_lang_key)
                    parsed = json.loads(raw)
                    new_name = parsed.get("filename", f"{img_page_name or 'image'}-{i+1}") + ".webp"
                    alt = parsed.get("alt", "")
                    log_img(f"✓ AI命名: {new_name}")
                except Exception as e:
                    new_name = f"{img_page_name or 'image'}-{i+1}.webp"
                    alt = ""
                    log_img(f"⚠ AI命名失败: {str(e)[:120]}")
            else:
                new_name = f"{img_page_name or 'image'}-{i+1}.webp"
                alt = ""
                log_img(f"✓ 命名: {new_name}")

            processed.append({"name": new_name, "bytes": webp_bytes, "alt": alt, "original": img_file.name})
            progress.progress(int((i+1) / len(img_files) * 100))

        log_img("✅ 全部处理完成！")
        st.session_state["img_processed"] = processed

    # ── Image results ──────────────────────────────────────────────────────────
    if "img_processed" in st.session_state:
        processed = st.session_state["img_processed"]
        st.success(f"✅ 已处理 {len(processed)} 张图片")
        st.divider()

        # Build ZIP for bulk download
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in processed:
                zf.writestr(p["name"], p["bytes"])
        zip_buf.seek(0)

        st.download_button(
            "📦 打包下载全部（ZIP）",
            data=zip_buf.getvalue(),
            file_name=f"{img_page_name or 'images'}-webp.zip",
            mime="application/zip",
            use_container_width=True
        )

        st.markdown("**或单张下载：**")
        st.markdown("")

        for p in processed:
            c1, c2, c3, c4 = st.columns([1, 2, 3, 1.5])
            with c1:
                st.image(p["bytes"], width=70)
            with c2:
                st.markdown(f"`{p['name']}`")
                st.caption(f"原文件: {p['original']}")
            with c3:
                if p["alt"]:
                    st.caption(f"Alt: {p['alt']}")
            with c4:
                st.download_button(
                    "⬇ 下载",
                    data=p["bytes"],
                    file_name=p["name"],
                    mime="image/webp",
                    key=f"dl_{p['name']}"
                )

    st.stop()  # Don't render the full pipeline below

st.divider()

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

st.markdown('<div class="info-tip">📷 图片由 <b>通义千问视觉模型（qwen-vl-plus）</b> 识别内容，自动生成SEO文件名和alt文字。文字翻译和优化由 <b>通义千问（qwen-plus）</b> 处理。</div>', unsafe_allow_html=True)

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
        log(f"通义千问 识别图片中（共 {len(image_files)} 张）...")
        for i, img_file in enumerate(image_files):
            try:
                raw = identify_image_qianwen(img_file, page_name, brand_name, lang_key)
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
                log(f"⚠ 图片 {i+1} 识别失败: {str(e)[:120]}")
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

    # Contact CTA block (appended to every HTML output)
    contact_block = """
<!-- Contact CTA -->
<div style="background:#1a1a1a;border-radius:12px;padding:40px 24px;text-align:center;margin-top:48px;">
  <h2 style="color:#ffffff;font-size:1.4rem;margin-bottom:12px;">Request a Quote for Your Project</h2>
  <p style="color:#cccccc;font-size:14px;max-width:560px;margin:0 auto 24px;">Wellucky provides OEM/ODM services for folding container house camp solutions. Share your headcount, site location, and functional requirements to receive a configuration recommendation and shipping plan.</p>
  <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
    <a href="https://www.welluckyhouse.com/contact" style="background:#22a06b;color:#fff;padding:12px 28px;border-radius:50px;text-decoration:none;font-weight:700;font-size:14px;letter-spacing:0.05em;">GET A QUOTE</a>
    <a href="mailto:Info@welluckyhouse.com" style="background:transparent;color:#fff;padding:12px 28px;border-radius:50px;text-decoration:none;font-weight:600;font-size:14px;border:1.5px solid #ffffff;">EMAIL US</a>
    <a href="https://wa.me/8618615329580" style="background:transparent;color:#fff;padding:12px 28px;border-radius:50px;text-decoration:none;font-weight:600;font-size:14px;border:1.5px solid #ffffff;">WHATSAPP</a>
  </div>
</div>"""

    # Full preview HTML wrapper
    preview_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 860px; margin: 0 auto; padding: 32px 24px; color: #1a1a2e; line-height: 1.7; }}
  h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 16px; }}
  h2 {{ font-size: 1.4rem; font-weight: 600; margin-top: 32px; margin-bottom: 12px; }}
  h3 {{ font-size: 1.1rem; font-weight: 600; margin-top: 24px; }}
  p {{ margin-bottom: 16px; }}
  ul, ol {{ margin-bottom: 16px; padding-left: 24px; }}
  li {{ margin-bottom: 6px; }}
  figure {{ margin: 24px 0; }}
  figure img {{ max-width: 100%; border-radius: 8px; background: #e5e7eb; min-height: 200px; display: block; }}
  figcaption {{ font-size: 13px; color: #6b7280; margin-top: 6px; text-align: center; }}
  .faq-section {{ background: #f8f9fb; border-radius: 10px; padding: 24px; margin-top: 32px; }}
  .faq-section h2 {{ margin-top: 0; }}
  strong {{ color: #1a1a2e; }}
</style>
</head>
<body>
{html_output}
{contact_block}
</body>
</html>"""

    html_with_contact = html_output + "\n" + contact_block

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👁️ 预览", "📄 HTML代码", "🖼️ 图片对照表", "🔍 SEO Meta", "🔗 URL & 摘要"])

    with tab1:
        st.caption("以下为页面渲染预览（图片占位符显示为空白，填入真实链接后可完整预览）")
        st.components.v1.html(preview_html, height=800, scrolling=True)

    with tab2:
        st.caption("复制以下代码，粘贴到CMS源代码编辑框（已包含底部联系栏）")
        st.code(html_with_contact, language="html")
        st.download_button("⬇ 下载HTML", html_with_contact,
            file_name=f"{seo.get('url_slug', 'output')}.html", mime="text/html")

    with tab3:
        if image_results:
            # ── Build ZIP in memory ───────────────────────────────────────────
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for img in image_results:
                    img["file"].seek(0)
                    try:
                        pil_img = Image.open(img["file"])
                        webp_buf = io.BytesIO()
                        pil_img.convert("RGB").save(webp_buf, format="WEBP", quality=85)
                        zf.writestr(img["new_name"], webp_buf.getvalue())
                    except Exception:
                        img["file"].seek(0)
                        zf.writestr(img["new_name"], img["file"].read())
                    img["file"].seek(0)
            zip_buffer.seek(0)

            # ── Top actions row ───────────────────────────────────────────────
            st.caption("上传图片到网站后台后，填入链接，点击「更新HTML」")
            ca, cb = st.columns([1, 4])
            with ca:
                st.download_button(
                    "📦 一键下载全部（ZIP）",
                    data=zip_buffer,
                    file_name="images.zip",
                    mime="application/zip",
                    use_container_width=True
                )

            st.markdown("---")

            # ── Per-image rows ────────────────────────────────────────────────
            updated_html = html_output
            for i, img in enumerate(image_results):
                c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
                with c1:
                    img["file"].seek(0)
                    st.image(img["file"], width=70)
                with c2:
                    st.markdown(f"`{img['placeholder']}`")
                    st.caption(f"文件名: {img['new_name']}")
                    st.caption(f"Alt: {img['alt']}")
                with c3:
                    # Single image download as WebP
                    img["file"].seek(0)
                    try:
                        pil_img = Image.open(img["file"])
                        single_buf = io.BytesIO()
                        pil_img.convert("RGB").save(single_buf, format="WEBP", quality=85)
                        single_buf.seek(0)
                        st.download_button(
                            f"⬇ 下载",
                            data=single_buf,
                            file_name=img["new_name"],
                            mime="image/webp",
                            key=f"dl_{i}",
                            use_container_width=True
                        )
                    except Exception:
                        img["file"].seek(0)
                        st.download_button(
                            f"⬇ 下载",
                            data=img["file"].read(),
                            file_name=img["new_name"],
                            mime="image/webp",
                            key=f"dl_{i}",
                            use_container_width=True
                        )
                    img["file"].seek(0)
                with c4:
                    url = st.text_input("图片URL", key=f"url_{i}",
                        placeholder="https://yoursite.com/images/...")
                    if url:
                        updated_html = updated_html.replace(img["placeholder"], url)

            st.markdown("---")
            if st.button("🔄 更新图片链接"):
                st.session_state["html_output"] = updated_html
                st.success("✓ 已更新，切换到「HTML代码」查看")
                st.rerun()
        else:
            st.info("本次未上传图片")

    with tab4:
        st.code(f"""<!-- 复制到页面 <head> 中 -->
<title>{seo.get('title', '')}</title>
<meta name="description" content="{seo.get('meta_description', '')}">
<meta property="og:title" content="{seo.get('og_title', '')}">
<meta property="og:description" content="{seo.get('og_description', '')}">
<meta property="og:type" content="{'product' if 'page_type_key' in dir() and page_type_key == 'product' else 'article'}">""",
        language="html")

    with tab5:
        st.markdown("**URL Slug**")
        st.code(seo.get("url_slug", ""), language="text")
        st.markdown("**Meta Description**")
        st.info(seo.get("meta_description", ""))
        st.markdown("**GEO/AIO摘要**（适合放页面第一段，AI搜索引擎容易引用）")
        st.success(seo.get("geo_summary", ""))
        st.markdown("**主要关键词：** `" + seo.get("primary_keyword", "") + "`")
        if seo.get("secondary_keywords"):
            st.markdown("**次要关键词：** " + "  ·  ".join(seo.get("secondary_keywords", [])))
