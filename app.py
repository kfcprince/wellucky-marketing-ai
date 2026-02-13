import streamlit as st
import google.generativeai as genai
import dashscope 
from dashscope import ImageSynthesis, MultiModalConversation, Generation
from zhipuai import ZhipuAI
from PIL import Image, ImageDraw, ImageFont
import io, base64, re, os, requests, uuid, json

# ==========================================
# 0. 全局配置
# ==========================================
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V29.2", layout="wide", page_icon="🦁")

# 初始化 session_state
if 'results_tab1' not in st.session_state: st.session_state.results_tab1 = []
if 'generated_bg' not in st.session_state: st.session_state.generated_bg = None
if 'seo_metadata' not in st.session_state: st.session_state.seo_metadata = {}

# 读取 API Keys
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    ALI_API_KEY = st.secrets.get("ALI_API_KEY", "")
    ZHIPU_API_KEY = st.secrets.get("ZHIPU_API_KEY", "")
except:
    GOOGLE_API_KEY = ALI_API_KEY = ZHIPU_API_KEY = ""

# 业务配置
BIZ_CONFIG = {
    "logistics": {
        "name": "VastLog", 
        "website": "www.vastlog.com", 
        "color": "#FF9900", 
        "type": "LogisticsService",
        "description": "Professional international logistics and shipping solutions",
        "keywords": ["logistics", "shipping", "freight", "cargo", "delivery", "transport", "DDP", "express"]
    },
    "house": {
        "name": "Wellucky", 
        "website": "www.wellucky.com", 
        "color": "#0066CC", 
        "type": "Product",
        "description": "Innovative container house and modular building solutions",
        "keywords": ["container", "house", "modular", "prefab", "portable", "cabin", "building", "steel"]
    }
}

# 社媒平台规则
PLATFORM_RULES = {
    "Facebook": {"length": "Keep under 2000 characters", "hashtags": "3-5", "tone": "friendly and engaging"},
    "LinkedIn": {"length": "Keep under 3000 characters", "hashtags": "3-5", "tone": "professional and authoritative"},
    "Twitter/X": {"length": "Keep under 280 characters", "hashtags": "2-3", "tone": "concise and impactful"},
    "Instagram": {"length": "Keep under 2200 characters", "hashtags": "20-30", "tone": "visual and inspiring"}
}

# YouTube封面预设
COVER_PRESETS = {
    "YouTube标准 (1280x720)": (1280, 720, "Safe area: center 1546x423"),
    "Facebook封面 (820x312)": (820, 312, "Mobile safe: center 640x312"),
    "LinkedIn横幅 (1584x396)": (1584, 396, "Logo safe: left 268x268")
}

# ==========================================
# 1. 工具函数
# ==========================================
def get_font(size):
    """获取字体"""
    try: 
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except: 
        return ImageFont.load_default()

def pil_to_base64(img):
    """PIL图片转Base64"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def convert_to_webp(image):
    """转换为WebP格式"""
    buf = io.BytesIO()
    if image.mode == 'RGBA': 
        image = image.convert('RGB')
    image.save(buf, format='WEBP', quality=85)
    return buf.getvalue()

# ==========================================
# 新增：智能文件名生成函数（测试版）
# ==========================================

def run_ai_vision_with_retry(engine, img, prompt, key, model_name, max_retries=2):
    """
    带重试机制的图像识别
    如果第一次返回的格式不对，会自动重试一次
    """
    for attempt in range(max_retries):
        result = run_ai_vision(engine, img, prompt, key, model_name)
        
        # 检查结果是否符合预期格式
        if result and not result.startswith("Error"):
            # 基本格式检查：应该包含至少2个连字符（brand-word1-word2）
            if result.count('-') >= 2:
                # 检查是否包含垃圾词
                garbage_words = ['this', 'image', 'photo', 'picture', 'showing', 'depicts']
                if not any(word in result.lower() for word in garbage_words):
                    return result  # ✅ 格式合格，直接返回
        
        # 如果是最后一次尝试，返回结果（即使不理想）
        if attempt == max_retries - 1:
            return result
    
    return result


def generate_seo_filename_smart(engine, img, brand, business_type, api_key, model_name):
    """
    智能生成SEO文件名 - 使用Few-Shot Learning
    
    参数:
        engine: AI引擎 (Google Gemini / 智谱清言 / 阿里通义)
        img: PIL图片对象
        brand: 品牌名 (Wellucky / VastLog)
        business_type: 业务类型 (house / logistics)
        api_key: API密钥
        model_name: 模型名称
    
    返回:
        SEO友好的文件名（不含扩展名）
    """
    
    # 获取业务关键词
    biz_config = BIZ_CONFIG.get(business_type, {})
    biz_keywords = biz_config.get("keywords", ["product"])
    keyword_examples = ", ".join(biz_keywords[:6])
    
    # 构建Few-Shot Prompt（核心：给AI看3个标准例子）
    prompt = f"""You are a professional SEO filename generator for {brand}.

TASK: Analyze this product image and generate a SEO-optimized filename.

EXACT OUTPUT FORMAT:
{brand.lower()}-keyword1-keyword2-keyword3-keyword4

✅ PERFECT EXAMPLES (learn from these):

Example 1:
Image: White container house with modern design, exterior view
Output: {brand.lower()}-container-house-white-exterior

Example 2:
Image: Blue modular office building with steel frame structure
Output: {brand.lower()}-modular-office-blue-steel-frame

Example 3:
Image: Portable cabin interior showing modern furniture
Output: {brand.lower()}-portable-cabin-modern-interior

KEYWORD VOCABULARY (use words from these categories):

Product Types: {keyword_examples}
Colors: white, blue, gray, black, red, green, beige, brown
Materials: steel, aluminum, wood, composite, metal, glass
Views: exterior, interior, front, side, rear, aerial, detail, close-up
Features: modern, portable, compact, large, small, luxury, custom, prefab, modular

STRICT RULES:
1. MUST start with: {brand.lower()}-
2. Add 3-5 descriptive keywords after brand
3. Use ONLY hyphens (-) to separate words
4. Use ONLY lowercase letters
5. NO spaces, NO underscores, NO special characters
6. NO generic words: image, photo, picture, view, showing, product, item, thing
7. NO file extensions (.webp, .jpg, .png)
8. NO explanations or extra text

ANALYSIS STEPS:
1. Identify the main product type
2. Note prominent colors
3. Identify materials if visible
4. Determine the viewing angle
5. Spot distinctive features

Now analyze the image and output ONLY the filename (one line, nothing else):"""

    # 调用AI（带重试机制）
    raw_response = run_ai_vision_with_retry(
        engine=engine,
        img=img,
        prompt=prompt,
        key=api_key,
        model_name=model_name,
        max_retries=2  # 最多重试2次
    )
    
    # === 清理AI返回的内容 ===
    
    filename = raw_response.strip().lower()
    
    # 移除常见的垃圾前缀/后缀
    garbage_patterns = [
        r'^(filename|output|result|answer):\s*',  # "filename: xxx"
        r'^(the filename is|here is)\s*',         # "the filename is xxx"
        r'```.*?```',                              # markdown代码块
        r'\*\*|\*',                                # markdown粗体/斜体
        r'\.webp$|\.jpg$|\.png$|\.jpeg$'          # 文件扩展名
    ]
    
    for pattern in garbage_patterns:
        filename = re.sub(pattern, '', filename, flags=re.IGNORECASE)
    
    filename = filename.strip()
    
    # 只保留字母、数字和连字符
    filename = re.sub(r'[^a-z0-9-]', '-', filename)
    
    # 清理多余的连字符
    filename = re.sub(r'-+', '-', filename)      # 多个连字符 → 单个
    filename = filename.strip('-')                # 删除首尾连字符
    
    # === 格式验证与修复 ===
    
    brand_lower = brand.lower()
    parts = filename.split('-')
    
    # 检查1：是否以品牌名开头
    if not filename.startswith(brand_lower + '-'):
        # 尝试在parts中找到品牌名
        if brand_lower in parts:
            parts.remove(brand_lower)
        # 把品牌名加到开头
        parts.insert(0, brand_lower)
        filename = '-'.join(parts)
    
    # 重新分割
    parts = filename.split('-')
    
    # 检查2：是否有足够的关键词（至少3部分：brand + 2个描述词）
    if len(parts) < 3:
        # 格式不合格，使用后备方案
        fallback_keyword = biz_keywords[0] if biz_keywords else "product"
        fallback_filename = f"{brand_lower}-{fallback_keyword}-{uuid.uuid4().hex[:6]}"
        return fallback_filename
    
    # 检查3：过滤垃圾词
    stop_words = {
        'this', 'that', 'the', 'and', 'for', 'with', 'from',
        'image', 'photo', 'picture', 'view', 'showing', 'depicts',
        'product', 'item', 'thing', 'object', 'file', 'name'
    }
    
    cleaned_parts = [brand_lower]  # 保留品牌名
    for part in parts[1:]:  # 从第二个词开始检查
        # 跳过太短的词（<3字符）或停用词
        if len(part) >= 3 and part not in stop_words:
            cleaned_parts.append(part)
    
    # 如果清理后词太少，说明都是垃圾词
    if len(cleaned_parts) < 3:
        fallback_keyword = biz_keywords[0] if biz_keywords else "product"
        fallback_filename = f"{brand_lower}-{fallback_keyword}-{uuid.uuid4().hex[:6]}"
        return fallback_filename
    
    # 检查4：限制长度（品牌名 + 最多5个关键词）
    if len(cleaned_parts) > 6:
        cleaned_parts = cleaned_parts[:6]
    
    final_filename = '-'.join(cleaned_parts)
    
    return final_filename

def run_ai_text(engine, prompt, key, model_name):
    """纯文本AI调用"""
    if not key: 
        return "Error: 缺少 API Key"
    
    try:
        if engine == "Google Gemini":
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        
        elif engine == "智谱清言":
            client = ZhipuAI(api_key=key)
            text_model = "glm-4-plus" if "plus" in model_name else "glm-4"
            response = client.chat.completions.create(
                model=text_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        
        elif engine == "阿里通义":
            dashscope.api_key = key
            messages = [{"role": "user", "content": prompt}]
            response = Generation.call(
                model='qwen-max',
                messages=messages
            )
            return response.output.text
        
        return "Error: 未知引擎"
    
    except Exception as e: 
        return f"Error: {str(e)}"

def run_ai_vision(engine, img, prompt, key, model_name):
    """图像识别AI调用"""
    if not key: 
        return "Error: 缺少 API Key"
    
    try:
        if engine == "Google Gemini":
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, img])
            return response.text
        
        elif engine == "智谱清言":
            client = ZhipuAI(api_key=key)
            img_base64 = f"data:image/png;base64,{pil_to_base64(img)}"
            vision_model = model_name if "v" in model_name.lower() else "glm-4v"
            response = client.chat.completions.create(
                model=vision_model,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_base64}}
                    ]
                }]
            )
            return response.choices[0].message.content
        
        elif engine == "阿里通义":
            dashscope.api_key = key
            tmp_path = f"temp_{uuid.uuid4()}.png"
            img.save(tmp_path)
            
            try:
                messages = [{
                    "role": "user", 
                    "content": [
                        {"image": f"file://{os.path.abspath(tmp_path)}"},
                        {"text": prompt}
                    ]
                }]
                response = MultiModalConversation.call(
                    model=model_name,
                    messages=messages
                )
                
                if isinstance(response.output.choices[0].message.content, list):
                    return response.output.choices[0].message.content[0]['text']
                return response.output.choices[0].message.content
            
            finally:
                if os.path.exists(tmp_path): 
                    os.remove(tmp_path)
        
        return "Error: 未知引擎"
    
    except Exception as e: 
        return f"Error: {str(e)}"

def generate_schema_json(biz_info, article_title, article_content, image_urls=[]):
    """生成Schema.org JSON-LD结构"""
    schema = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Organization",
                "name": biz_info['name'],
                "url": f"https://{biz_info['website']}",
                "logo": f"https://{biz_info['website']}/logo.png",
                "description": biz_info['description']
            },
            {
                "@type": "Article",
                "headline": article_title,
                "description": article_content[:200],
                "author": {
                    "@type": "Organization",
                    "name": biz_info['name']
                },
                "publisher": {
                    "@type": "Organization",
                    "name": biz_info['name'],
                    "logo": {
                        "@type": "ImageObject",
                        "url": f"https://{biz_info['website']}/logo.png"
                    }
                },
                "datePublished": "2024-02-13",
                "image": image_urls
            },
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {
                        "@type": "ListItem",
                        "position": 1,
                        "name": "Home",
                        "item": f"https://{biz_info['website']}"
                    },
                    {
                        "@type": "ListItem",
                        "position": 2,
                        "name": article_title,
                        "item": f"https://{biz_info['website']}/blog/{article_title.lower().replace(' ', '-')}"
                    }
                ]
            }
        ]
    }
    return json.dumps(schema, indent=2, ensure_ascii=False)

def analyze_seo_score(html_content):
    """分析SEO得分"""
    checks = {
        "H1标签": bool(re.search(r'<h1>', html_content)),
        "H2标签": len(re.findall(r'<h2>', html_content)) >= 2,
        "Schema标记": bool(re.search(r'application/ld\+json', html_content)),
        "图片Alt": len(re.findall(r'alt="[^"]+"', html_content)) > 0,
        "内链数量": len(re.findall(r'<a href="/', html_content)) >= 1,
        "字数统计": len(re.sub(r'<[^>]+>', '', html_content)) >= 300
    }
    
    score = sum(checks.values()) / len(checks) * 100
    return score, checks

# ==========================================
# 2. 侧边栏配置
# ==========================================
with st.sidebar:
    st.title("⚙️ 系统配置 V29.2")
    
    # 业务选择
    st.markdown("### 🏢 业务模式")
    biz_choice = st.radio(
        "Business", 
        ("🚢 VastLog (物流)", "🏠 Wellucky (房屋)"), 
        label_visibility="collapsed"
    )
    cbiz = "logistics" if "VastLog" in biz_choice else "house"
    cinfo = BIZ_CONFIG[cbiz]
    
    st.info(f"**当前品牌:** {cinfo['name']}\n**网站:** {cinfo['website']}")
    
    st.divider()
    
    # AI引擎选择
    st.markdown("### 🧠 AI 引擎配置")
    engine_choice = st.radio("选择AI厂商", ("Google Gemini", "智谱清言", "阿里通义"))
    
    if engine_choice == "Google Gemini":
        model_options = [
        "gemini-2-5-flash-preview",
        "gemini-2-5-pro",
        "gemini-3-pro-preview",
        ]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = GOOGLE_API_KEY
        api_status = "✅ 已配置" if GOOGLE_API_KEY else "❌ 未配置"
    
    elif engine_choice == "智谱清言":
        model_options = [
            "glm-4-6v",
            "glm-4v",
            "glm-4-plus",
            "glm-4"
        ]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = ZHIPU_API_KEY
        api_status = "✅ 已配置" if ZHIPU_API_KEY else "❌ 未配置"
        
        if "v" not in sel_model.lower():
            st.caption("⚠️ 图片识别需选择带'v'的模型")
    
    else:
        model_options = [
            "qwen-vl-max",
            "qwen-vl-plus",
            "qwen-max"
        ]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = ALI_API_KEY
        api_status = "✅ 已配置" if ALI_API_KEY else "❌ 未配置"
    
    st.caption(f"API状态: {api_status}")
    
    st.divider()
    
    st.markdown("### 📊 系统状态")
    st.caption(f"• 引擎: {engine_choice}")
    st.caption(f"• 模型: {sel_model}")
    st.caption(f"• 品牌: {cinfo['name']}")

# ==========================================
# 3. 主界面
# ==========================================
st.markdown(f"## 🦁 {cinfo['name']} 数字化运营台")

tab1, tab2, tab3 = st.tabs([
    "✍️ 智能文案生成", 
    "🎨 封面工厂", 
    "🌍 GEO/EEAT 优化专家"
])

# ==========================================
# Tab 1: 智能文案生成
# ==========================================
with tab1:
    st.markdown("### 📝 批量图片识别 + 社媒文案生成")
    
    col_upload, col_settings = st.columns([1, 1])
    
    with col_upload:
        files_t1 = st.file_uploader(
            "📂 批量上传产品图片", 
            accept_multiple_files=True, 
            key="t1_upload",
            help="支持 JPG, PNG 格式"
        )
    
    with col_settings:
        platform_choice = st.selectbox(
            "🎯 目标社媒平台",
            list(PLATFORM_RULES.keys()),
            help="不同平台有不同的字数和风格要求"
        )
        
        draft_context = st.text_area(
            "📋 补充背景信息（可选）",
            height=80,
            placeholder="例如：促销活动、产品特点、目标受众等..."
        )
        
        include_hashtags = st.checkbox("生成Hashtags", value=True)
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        btn_rename_only = st.button(
            "🖼️ 仅识图重命名", 
            use_container_width=True,
            help="只识别图片并生成SEO文件名"
        )
    
    with col_btn2:
        btn_full_process = st.button(
            "🚀 完整处理（文件名+文案）", 
            type="primary", 
            use_container_width=True,
            help="识别图片 + 生成社媒文案 + 转WebP"
        )
    
    # 处理逻辑
    if (btn_rename_only or btn_full_process) and files_t1:
        if not api_key:
            st.error("❌ 请先在Streamlit Secrets中配置API Key！")
        else:
            st.session_state.results_tab1 = []
            
            # 社媒文案Prompt
            platform_rule = PLATFORM_RULES[platform_choice]
            prompt_copywriting = f"""
You are a social media expert for {cinfo['name']} ({cinfo['type']}).
Create a {platform_choice} post for this product image.

Requirements:
- {platform_rule['length']}
- Tone: {platform_rule['tone']}
- Include product benefits and call-to-action
{'- Include ' + platform_rule['hashtags'] + ' relevant hashtags at the end' if include_hashtags else ''}

Context: {draft_context if draft_context else 'Professional product promotion'}

Write the post directly, no explanations.
"""
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 🔄 循环处理每张图片
            for idx, uploaded_file in enumerate(files_t1):
                status_text.text(f"处理中: {uploaded_file.name} ({idx+1}/{len(files_t1)})")
                
                img = Image.open(uploaded_file).convert("RGB")
                
                # 1. 图片识别并重命名（使用智能函数）
                clean_filename = generate_seo_filename_smart(
                    engine=engine_choice,
                    img=img,
                    brand=cinfo['name'],
                    business_type=cbiz,
                    api_key=api_key,
                    model_name=sel_model
                ) + ".webp"
                
                # 2. 生成文案（如果选择完整处理）
                copywriting_text = ""
                if btn_full_process:
                    copywriting_text = run_ai_vision(engine_choice, img, prompt_copywriting, api_key, sel_model)
                
                # 3. 转换为WebP
                webp_data = convert_to_webp(img)
                
                # 保存结果
                st.session_state.results_tab1.append({
                    "original_name": uploaded_file.name,
                    "img": img,
                    "new_name": clean_filename,
                    "copy_text": copywriting_text,
                    "webp_data": webp_data
                })
                
                # 更新进度条
                progress_bar.progress((idx + 1) / len(files_t1))
            
            status_text.text("✅ 处理完成！")
    
    # 显示结果
    if st.session_state.results_tab1:
        st.divider()
        st.markdown("### 📊 处理结果")
        
        for idx, result in enumerate(st.session_state.results_tab1):
            with st.expander(f"🖼️ {result['original_name']} → {result['new_name']}", expanded=(idx==0)):
                col_img, col_content = st.columns([1, 2])
                
                with col_img:
                    st.image(result['img'], use_column_width=True)
                    st.download_button(
                        "⬇️ 下载WebP",
                        data=result['webp_data'],
                        file_name=result['new_name'],
                        mime="image/webp",
                        use_container_width=True,
                        key=f"dl_{idx}"
                    )
                
                with col_content:
                    # 调试信息
                    with st.expander("🔍 调试信息", expanded=False):
                        st.caption(f"✨ 使用Few-Shot智能命名")
                        st.caption(f"品牌: {cinfo['name']} | 业务: {cbiz}")
                    
                    st.text_input("SEO文件名", value=result['new_name'], key=f"name_{idx}")
                    
                    if result['copy_text']:
                        st.text_area(
                            f"{platform_choice} 文案",
                            value=result['copy_text'],
                            height=200,
                            key=f"copy_{idx}"
                        )
                        
                        # 文案统计
                        char_count = len(result['copy_text'])
                        hashtag_count = result['copy_text'].count('#')
                        st.caption(f"📊 字符数: {char_count} | Hashtags: {hashtag_count}")
# Tab 2 和 Tab 3 保持不变...
# （由于字数限制，这里省略，你可以直接复制之前的Tab 2和Tab 3代码）

# ==========================================
# Tab 2: 封面工厂
# ==========================================
with tab2:
    st.markdown("### 🎨 YouTube/社媒封面设计工具")
    
    col_bg, col_text = st.columns([1, 1])
    
    with col_bg:
        st.markdown("#### A. 背景图层")
        
        preset_choice = st.selectbox(
            "封面尺寸预设",
            list(COVER_PRESETS.keys()),
            help="选择目标平台的标准尺寸"
        )
        preset_size = COVER_PRESETS[preset_choice]
        st.caption(f"📐 尺寸: {preset_size[0]}x{preset_size[1]} | {preset_size[2]}")
        
        bg_source = st.radio(
            "背景来源",
            ["📁 本地上传", "🤖 AI生成 (阿里Wanx)"],
            horizontal=True
        )
        
        bg_image = None
        
        if bg_source == "📁 本地上传":
            uploaded_bg = st.file_uploader(
                "上传背景图", 
                type=['jpg', 'jpeg', 'png'],
                key="bg_upload"
            )
            if uploaded_bg:
                bg_image = Image.open(uploaded_bg).convert("RGBA")
                bg_image = bg_image.resize((preset_size[0], preset_size[1]))
        
        else:
            ai_prompt = st.text_input(
                "🎨 描述画面内容",
                placeholder="例如: modern container house in sunset, professional photography"
            )
            
            if st.button("🚀 生成AI背景", use_container_width=True):
                if not ALI_API_KEY:
                    st.error("❌ 需要配置阿里云API Key")
                else:
                    with st.spinner("AI绘图中..."):
                        try:
                            dashscope.api_key = ALI_API_KEY
                            response = ImageSynthesis.call(
                                model=ImageSynthesis.Models.wanx_v1,
                                prompt=ai_prompt,
                                n=1,
                                size='1024*1024'
                            )
                            
                            if response.status_code == 200:
                                img_url = response.output.results[0].url
                                img_data = requests.get(img_url).content
                                bg_image = Image.open(io.BytesIO(img_data)).convert("RGBA")
                                bg_image = bg_image.resize((preset_size[0], preset_size[1]))
                                st.session_state.generated_bg = bg_image
                                st.success("✅ 生成成功！")
                            else:
                                st.error(f"生成失败: {response.message}")
                        
                        except Exception as e:
                            st.error(f"错误: {str(e)}")
            
            if st.session_state.generated_bg:
                bg_image = st.session_state.generated_bg
    
    with col_text:
        st.markdown("#### B. 文字叠加层")
        
        with st.expander("🔤 主标题", expanded=True):
            text1 = st.text_input("文字内容", "Global Service", key="t1_text")
            col_t1a, col_t1b = st.columns(2)
            size1 = col_t1a.number_input("字号", 20, 300, 80, key="t1_size")
            color1 = col_t1b.color_picker("颜色", "#FFFFFF", key="t1_color")
            y1 = st.slider("垂直位置", 0, preset_size[1], int(preset_size[1]*0.3), key="t1_y")
        
        with st.expander("🔤 副标题"):
            text2 = st.text_input("文字内容", cinfo['name'], key="t2_text")
            col_t2a, col_t2b = st.columns(2)
            size2 = col_t2a.number_input("字号", 20, 300, 50, key="t2_size")
            color2 = col_t2b.color_picker("颜色", cinfo['color'], key="t2_color")
            y2 = st.slider("垂直位置", 0, preset_size[1], int(preset_size[1]*0.5), key="t2_y")
        
        with st.expander("🔤 装饰文字"):
            text3 = st.text_input("文字内容", "Fast & Reliable", key="t3_text")
            col_t3a, col_t3b = st.columns(2)
            size3 = col_t3a.number_input("字号", 20, 300, 30, key="t3_size")
            color3 = col_t3b.color_picker("颜色", "#FFD700", key="t3_color")
            y3 = st.slider("垂直位置", 0, preset_size[1], int(preset_size[1]*0.7), key="t3_y")
    
    if bg_image:
        st.divider()
        st.markdown("### 🖼️ 封面预览")
        
        final_cover = bg_image.copy()
        draw = ImageDraw.Draw(final_cover)
        W, H = final_cover.size
        
        def draw_text_with_shadow(text, size, color, y_pos):
            if not text:
                return
            
            font = get_font(int(size))
            
            try:
                text_width = draw.textlength(text, font=font)
            except:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
            
            x = (W - text_width) / 2
            
            draw.text((x + 4, y_pos + 4), text, font=font, fill="black")
            draw.text((x, y_pos), text, font=font, fill=color)
        
        draw_text_with_shadow(text1, size1, color1, y1)
        draw_text_with_shadow(text2, size2, color2, y2)
        draw_text_with_shadow(text3, size3, color3, y3)
        
        if "YouTube" in preset_choice:
            safe_x1 = int((W - 1546) / 2)
            safe_y1 = int((H - 423) / 2)
            safe_x2 = safe_x1 + 1546
            safe_y2 = safe_y1 + 423
            draw.rectangle(
                [(safe_x1, safe_y1), (safe_x2, safe_y2)], 
                outline="red", 
                width=3
            )
        
        st.image(final_cover, use_column_width=True)
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            buf_jpg = io.BytesIO()
            final_cover.convert("RGB").save(buf_jpg, format="JPEG", quality=95)
            st.download_button(
                "⬇️ 下载JPG (高质量)",
                data=buf_jpg.getvalue(),
                file_name=f"{cinfo['name'].lower()}-cover.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
        
        with col_exp2:
            buf_png = io.BytesIO()
            final_cover.save(buf_png, format="PNG")
            st.download_button(
                "⬇️ 下载PNG (透明)",
                data=buf_png.getvalue(),
                file_name=f"{cinfo['name'].lower()}-cover.png",
                mime="image/png",
                use_container_width=True
            )

# Tab 3: GEO/EEAT（继续使用之前的完整代码）
with tab3:
    st.markdown("### 🌍 SEO内容生成 + EEAT优化 + Schema标记")
    
    st.info(f"""
    **当前配置:**
    - AI引擎: {engine_choice} ({sel_model})
    - 品牌: {cinfo['name']}
    - 输出: 中文→英文 + HTML + Schema + SEO元数据
    """)
    
    col_input, col_images = st.columns([2, 1])
    
    with col_input:
        chinese_content = st.text_area(
            "📝 输入中文原文",
            height=250,
            placeholder="输入需要翻译和优化的中文内容...",
            help="支持产品介绍、博客文章、新闻稿等"
        )
        
        article_title = st.text_input(
            "📌 文章标题（中文）",
            placeholder="例如：集装箱房屋的5大优势",
            help="用于生成H1标签和SEO元数据"
        )
    
    with col_images:
        uploaded_images = st.file_uploader(
            "🖼️ 上传配图（可选）",
            accept_multiple_files=True,
            type=['jpg', 'jpeg', 'png'],
            key="t3_images",
            help="AI会为每张图生成优化的alt描述"
        )
        
        if uploaded_images:
            st.caption(f"已上传 {len(uploaded_images)} 张图片")
            for img_file in uploaded_images:
                st.image(img_file, width=100)
    
    with st.expander("⚙️ 高级选项"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            include_faq = st.checkbox("生成FAQ Schema", value=False)
            include_howto = st.checkbox("生成HowTo Schema", value=False)
        
        with col_adv2:
            target_word_count = st.number_input(
                "目标字数",
                min_value=300,
                max_value=3000,
                value=800,
                step=100
            )
            internal_links = st.text_area(
                "内链建议（每行一个URL）",
                placeholder="https://www.wellucky.com/products\nhttps://www.wellucky.com/about",
                height=60
            )
    
    if st.button("✨ 生成完整SEO内容", type="primary", use_container_width=True):
        if not chinese_content or not article_title:
            st.warning("⚠️ 请输入中文内容和文章标题")
        elif not api_key:
            st.error("❌ 请先配置API Key")
        else:
            with st.spinner("🤖 AI正在生成SEO优化内容..."):
                try:
                    image_filenames = []
                    
                    if uploaded_images:
                        for img_file in uploaded_images:
                            image_filenames.append(img_file.name)
                    
                    main_prompt = f"""
You are an SEO expert specializing in EEAT (Experience, Expertise, Authoritativeness, Trustworthiness) content optimization for {cinfo['name']}.

**TASK:**
1. Translate the following CHINESE content to PROFESSIONAL ENGLISH
2. Optimize for SEO and EEAT principles
3. Generate complete HTML article with Schema markup
4. Target word count: approximately {target_word_count} words

**BRAND CONTEXT:**
- Company: {cinfo['name']}
- Website: {cinfo['website']}
- Business Type: {cinfo['type']}
- Description: {cinfo['description']}

**CONTENT TO TRANSLATE:**
Title: {article_title}

Body:
{chinese_content}

**FORMATTING REQUIREMENTS:**

1. HTML Structure:
   - Use semantic HTML5 tags
   - H1 for main title (translate article title)
   - H2 for major sections (styled with border-left: 5px solid {cinfo['color']}; padding-left: 15px;)
   - H3 for subsections
   - Proper paragraph tags <p>

2. Image Integration:
   {"- Insert images using: <img src='FILENAME' alt='SEO_DESCRIPTION' style='width:100%; max-width:800px; margin:20px auto; display:block;'>" if uploaded_images else "- No images uploaded"}
   {"- Available images: " + ", ".join(image_filenames) if uploaded_images else ""}
   - Generate descriptive, keyword-rich alt text for each image
   - Position images logically within content flow

3. SEO Elements:
   - Natural keyword integration (avoid keyword stuffing)
   - Include semantic variations of main keywords
   - Add internal links where relevant: {internal_links if internal_links else 'None specified'}
   - Use strong/em tags for emphasis (sparingly)

4. EEAT Optimization:
   - Demonstrate expertise through detailed explanations
   - Show real-world experience and examples
   - Include authoritative references or data points
   - Build trust through transparent, helpful information

5. Schema.org JSON-LD:
   - Include complete Schema markup in <script type="application/ld+json">
   - Use Article schema as primary type
   - Include Organization schema for {cinfo['name']}
   - Add BreadcrumbList for navigation
   {"- Add FAQPage schema with 3-5 relevant Q&A pairs" if include_faq else ""}
   {"- Add HowTo schema with step-by-step instructions" if include_howto else ""}

**OUTPUT FORMAT:**
Return ONLY the complete HTML code, starting with:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>[TRANSLATED_TITLE]</title>
    <script type="application/ld+json">
    [SCHEMA_JSON]
    </script>
</head>
<body>
    [OPTIMIZED_CONTENT]
</body>
</html>
```

Do not include any explanations or comments outside the HTML code.
"""
                    
                    if engine_choice == "Google Gemini" and uploaded_images:
                        content_parts = [main_prompt]
                        for img_file in uploaded_images:
                            img = Image.open(img_file)
                            content_parts.append(img)
                        
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(sel_model)
                        response = model.generate_content(content_parts)
                        html_output = response.text
                    
                    else:
                        html_output = run_ai_text(engine_choice, main_prompt, api_key, sel_model)
                    
                    html_output = re.sub(r'^```html\s*', '', html_output)
                    html_output = re.sub(r'\s*```$', '', html_output)
                    html_output = html_output.strip()
                    
                    # SEO元数据生成
                    url_slug = article_title.lower()
                    url_slug = re.sub(r'[^\w\s-]', '', url_slug)
                    url_slug = re.sub(r'[\s_]+', '-', url_slug)
                    url_slug = f"{cinfo['name'].lower()}-{url_slug}"
                    
                    meta_desc_prompt = f"""
Generate a compelling Meta Description (150-155 characters) for this article:
Title: {article_title}
Content: {chinese_content[:300]}

Requirements:
- Include main keyword
- Compelling call-to-action
- Exactly 150-155 characters
- In English

Output only the meta description text, no explanations.
"""
                    meta_description = run_ai_text(engine_choice, meta_desc_prompt, api_key, sel_model).strip()
                    
                    keywords_prompt = f"""
Extract 8-12 relevant SEO keywords from this content:
{chinese_content}

Requirements:
- Mix of short-tail and long-tail keywords
- Include brand name: {cinfo['name']}
- Comma-separated list
- In English

Output only the keyword list, no explanations.
"""
                    meta_keywords = run_ai_text(engine_choice, keywords_prompt, api_key, sel_model).strip()
                    
                    excerpt_prompt = f"""
Write a compelling excerpt/summary (180-220 words) for this article:
Title: {article_title}
Content: {chinese_content}

Requirements:
- Engaging opening hook
- Summarize main points
- Include call-to-action
- 180-220 words
- In English

Output only the excerpt text, no explanations.
"""
                    excerpt_text = run_ai_text(engine_choice, excerpt_prompt, api_key, sel_model).strip()
                    
                    image_alts = []
                    if uploaded_images:
                        for img_file in uploaded_images:
                            alt_prompt = f"""
Generate SEO-optimized alt text for this image in the context of:
Article: {article_title}
Company: {cinfo['name']}

Requirements:
- Descriptive and specific
- Include relevant keywords naturally
- 10-15 words
- In English

Output only the alt text, no explanations.
"""
                            img_pil = Image.open(img_file)
                            alt_text = run_ai_vision(engine_choice, img_pil, alt_prompt, api_key, sel_model).strip()
                            image_alts.append({"filename": img_file.name, "alt": alt_text})
                    
                    st.session_state.seo_metadata = {
                        "url_slug": url_slug,
                        "meta_description": meta_description,
                        "meta_keywords": meta_keywords,
                        "excerpt": excerpt_text,
                        "image_alts": image_alts,
                        "html_content": html_output
                    }
                    
                    st.success("✅ 生成完成！")
                
                except Exception as e:
                    st.error(f"❌ 生成失败: {str(e)}")
                    st.exception(e)
    
    if st.session_state.seo_metadata:
        st.divider()
        st.markdown("## 📊 生成结果")
        
        with st.expander("📝 SEO元数据（复制到WordPress/CMS）", expanded=True):
            col_meta1, col_meta2 = st.columns(2)
            
            with col_meta1:
                st.text_input(
                    "🔗 自定义URL Slug",
                    value=st.session_state.seo_metadata['url_slug'],
                    help="用于永久链接"
                )
                
                st.text_area(
                    "📄 Meta Description (155字符)",
                    value=st.session_state.seo_metadata['meta_description'],
                    height=80,
                    help="显示在搜索结果中"
                )
            
            with col_meta2:
                st.text_area(
                    "🏷️ Meta Keywords",
                    value=st.session_state.seo_metadata['meta_keywords'],
                    height=80,
                    help="逗号分隔的关键词列表"
                )
                
                st.text_area(
                    "📌 文章摘要 Excerpt",
                    value=st.session_state.seo_metadata['excerpt'],
                    height=100,
                    help="用于文章预览和分享"
                )
            
            if st.session_state.seo_metadata['image_alts']:
                st.markdown("**🖼️ 图片Alt文本优化:**")
                for idx, img_alt in enumerate(st.session_state.seo_metadata['image_alts']):
                    st.text_input(
                        f"图片 {idx+1}: {img_alt['filename']}",
                        value=img_alt['alt'],
                        key=f"alt_{idx}"
                    )
            
            all_metadata = f"""
=== SEO元数据 ===
URL Slug: {st.session_state.seo_metadata['url_slug']}

Meta Description:
{st.session_state.seo_metadata['meta_description']}

Meta Keywords:
{st.session_state.seo_metadata['meta_keywords']}

Excerpt:
{st.session_state.seo_metadata['excerpt']}

{"=" * 50}
图片Alt文本:
{"=" * 50}
{"".join([f"{i+1}. {img['filename']}: {img['alt']}\n" for i, img in enumerate(st.session_state.seo_metadata['image_alts'])])}
"""
            
            st.download_button(
                "📋 下载完整元数据.txt",
                data=all_metadata,
                file_name=f"{st.session_state.seo_metadata['url_slug']}-metadata.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        tab_preview, tab_code, tab_score = st.tabs(["👁️ 预览", "💻 HTML代码", "📊 SEO评分"])
        
        with tab_preview:
            st.markdown(
                st.session_state.seo_metadata['html_content'],
                unsafe_allow_html=True
            )
        
        with tab_code:
            st.code(
                st.session_state.seo_metadata['html_content'],
                language="html",
                line_numbers=True
            )
            
            st.download_button(
                "⬇️ 下载HTML文件",
                data=st.session_state.seo_metadata['html_content'],
                file_name=f"{st.session_state.seo_metadata['url_slug']}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with tab_score:
            score, checks = analyze_seo_score(st.session_state.seo_metadata['html_content'])
            
            st.metric("SEO优化得分", f"{score:.0f}/100")
            
            st.markdown("**检查项目:**")
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                st.markdown(f"{status} {check_name}")
            
            if score < 80:
                st.warning("⚠️ 建议优化未通过的检查项以提升SEO效果")
            else:
                st.success("🎉 SEO优化良好！")

st.divider()
st.caption(f"🦁 {cinfo['name']} 运营中台 V29.2 | Powered by {engine_choice} ({sel_model})")




