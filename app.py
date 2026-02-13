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
st.set_page_config(page_title="Wellucky & VastLog 运营中台 V29.1", layout="wide", page_icon="🦁")

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
        "description": "Professional international logistics and shipping solutions"
    },
    "house": {
        "name": "Wellucky", 
        "website": "www.wellucky.com", 
        "color": "#0066CC", 
        "type": "Product",
        "description": "Innovative container house and modular building solutions"
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

def get_clean_seo_name(ai_res, brand):
    """生成SEO友好的文件名"""
    if not ai_res or "Error" in ai_res: 
        return f"{brand.lower()}-item-{uuid.uuid4().hex[:4]}"
    
    name = ai_res.lower()
    name = re.sub(r'[^a-z0-9]', ' ', name)
    words = [w for w in name.split() if len(w) > 2 and w not in {'this','image','photo','view','the','and','for'}]
    
    brand_low = brand.lower()
    if brand_low in words: 
        words.remove(brand_low)
    words.insert(0, brand_low)
    
    return "-".join(words[:6])

def run_ai_text(engine, prompt, key, model_name):
    """纯文本AI调用（用于生成文案、SEO等）"""
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
            # 智谱纯文本用 glm-4-plus 或 glm-4
            text_model = "glm-4-plus" if "plus" in model_name else "glm-4"
            response = client.chat.completions.create(
                model=text_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        
        elif engine == "阿里通义":
            dashscope.api_key = key
            messages = [{"role": "user", "content": prompt}]
            # 阿里纯文本用qwen-max
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
            # 智谱图片识别必须用 glm-4v 或你的 glm-4-6v
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
                    model=model_name,  # qwen-vl-max 或 qwen-vl-plus
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
    st.title("⚙️ 系统配置 V29.1")
    
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
    
    # 根据引擎显示不同模型（修正后的模型列表）
    if engine_choice == "Google Gemini":
        model_options = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-lite", 
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b"
        ]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = GOOGLE_API_KEY
        api_status = "✅ 已配置" if GOOGLE_API_KEY else "❌ 未配置"
    
    elif engine_choice == "智谱清言":
        model_options = [
            "glm-4-6v",      # 你实际使用的模型
            "glm-4v",        # 图片识别
            "glm-4-plus",    # 纯文本
            "glm-4"          # 标准版
        ]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = ZHIPU_API_KEY
        api_status = "✅ 已配置" if ZHIPU_API_KEY else "❌ 未配置"
        
        # 提示：图片识别需要v系列模型
        if "v" not in sel_model.lower():
            st.caption("⚠️ 图片识别需选择带'v'的模型")
    
    else:  # 阿里通义
        model_options = [
            "qwen-vl-max",    # 图片识别
            "qwen-vl-plus",   # 图片识别
            "qwen-max"        # 纯文本
        ]
        sel_model = st.selectbox("模型版本", model_options, index=0)
        api_key = ALI_API_KEY
        api_status = "✅ 已配置" if ALI_API_KEY else "❌ 未配置"
    
    st.caption(f"API状态: {api_status}")
    
    st.divider()
    
    # 系统信息
    st.markdown("### 📊 系统状态")
    st.caption(f"• 引擎: {engine_choice}")
    st.caption(f"• 模型: {sel_model}")
    st.caption(f"• 品牌: {cinfo['name']}")

# ==========================================
# 3. 主界面
# ==========================================
st.markdown(
    f"## 🦁 {cinfo['name']} 数字化运营台", 
    unsafe_allow_html=True
)

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
            
            # Prompt设计
            prompt_naming = f"""
            Analyze this product image and generate a SEO-friendly filename.
            Format: {cinfo['name'].lower()}-keyword1-keyword2-keyword3
            Rules:
            - Use lowercase only
            - Use hyphens to separate words
            - Include 3-5 descriptive keywords
            - Focus on product type, material, use case
            - No generic words like 'image', 'photo', 'product'
            
            Output only the filename, nothing else.
            """
            
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
            
            for idx, uploaded_file in enumerate(files_t1):
                status_text.text(f"处理中: {uploaded_file.name} ({idx+1}/{len(files_t1)})")
                
                img = Image.open(uploaded_file).convert("RGB")
                
                # 1. 图片识别并重命名
                raw_name = run_ai_vision(engine_choice, img, prompt_naming, api_key, sel_model)
                clean_filename = get_clean_seo_name(raw_name, cinfo['name']) + ".webp"
                
                # 2. 生成文案（如果选择完整处理）
                copywriting_text = ""
                if btn_full_process:
                    copywriting_text = run_ai_vision(engine_choice, img, prompt_copywriting, api_key, sel_model)
                
                # 3. 转换为WebP
                webp_data = convert_to_webp(img)
                
                st.session_state.results_tab1.append({
                    "original_name": uploaded_file.name,
                    "img": img,
                    "new_name": clean_filename,
                    "copy_text": copywriting_text,
                    "webp_data": webp_data
                })
                
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
                        use_container_width=True
                    )
                
                with col_content:
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

# ==========================================
# Tab 2: 封面工厂
# ==========================================
with tab2:
    st.markdown("### 🎨 YouTube/社媒封面设计工具")
    
    col_bg, col_text = st.columns([1, 1])
    
    # 左侧：背景层
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
        
        else:  # AI生成
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
    
    # 右侧：文字层
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
    
    # 预览与导出
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
            
            # 计算文字宽度
            try:
                text_width = draw.textlength(text, font=font)
            except:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
            
            x = (W - text_width) / 2
            
            # 绘制阴影
            draw.text((x + 4, y_pos + 4), text, font=font, fill="black")
            # 绘制文字
            draw.text((x, y_pos), text, font=font, fill=color)
        
        draw_text_with_shadow(text1, size1, color1, y1)
        draw_text_with_shadow(text2, size2, color2, y2)
        draw_text_with_shadow(text3, size3, color3, y3)
        
        # 如果是YouTube，绘制安全区参考线
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
        
        # 导出选项
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # 导出JPG
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
            # 导出PNG
            buf_png = io.BytesIO()
            final_cover.save(buf_png, format="PNG")
            st.download_button(
                "⬇️ 下载PNG (透明)",
                data=buf_png.getvalue(),
                file_name=f"{cinfo['name'].lower()}-cover.png",
                mime="image/png",
                use_container_width=True
            )

# ==========================================
# Tab 3: GEO/EEAT 优化专家
# ==========================================
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
    
    # 高级选项
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
    
    # 生成按钮
    if st.button("✨ 生成完整SEO内容", type="primary", use_container_width=True):
        if not chinese_content or not article_title:
            st.warning("⚠️ 请输入中文内容和文章标题")
        elif not api_key:
            st.error("❌ 请先配置API Key")
        else:
            with st.spinner("🤖 AI正在生成SEO优化内容..."):
                try:
                    # 准备图片文件名列表
                    image_filenames = []
                    image_alt_texts = []
                    
                    if uploaded_images:
                        for img_file in uploaded_images:
                            image_filenames.append(img_file.name)
                    
                    # 构建主Prompt
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
                    
                    # 调用AI生成HTML
                    if engine_choice == "Google Gemini" and uploaded_images:
                        # Google支持图文混合
                        content_parts = [main_prompt]
                        for img_file in uploaded_images:
                            img = Image.open(img_file)
                            content_parts.append(img)
                        
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(sel_model)
                        response = model.generate_content(content_parts)
                        html_output = response.text
                    
                    else:
                        # 其他引擎用纯文本
                        html_output = run_ai_text(engine_choice, main_prompt, api_key, sel_model)
                    
                    # 清理输出（移除markdown代码块标记）
                    html_output = re.sub(r'^```html\s*', '', html_output)
                    html_output = re.sub(r'\s*```$', '', html_output)
                    html_output = html_output.strip()
                    
                    # ============================================
                    # 生成SEO元数据
                    # ============================================
                    
                    # 1. 自定义URL
                    url_slug = article_title.lower()
                    url_slug = re.sub(r'[^\w\s-]', '', url_slug)
                    url_slug = re.sub(r'[\s_]+', '-', url_slug)
                    url_slug = f"{cinfo['name'].lower()}-{url_slug}"
                    
                    # 2. Meta Description
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
                    
                    # 3. Meta Keywords
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
                    
                    # 4. 摘要/Excerpt
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
                    
                    # 5. 图片Alt文本（如果有图片）
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
                    
                    # 保存到session state
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
                    st.exception(e)  # 显示详细错误信息
    
    # 显示结果
    if st.session_state.seo_metadata:
        st.divider()
        st.markdown("## 📊 生成结果")
        
        # SEO元数据展示
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
            
            # 图片Alt文本
            if st.session_state.seo_metadata['image_alts']:
                st.markdown("**🖼️ 图片Alt文本优化:**")
                for idx, img_alt in enumerate(st.session_state.seo_metadata['image_alts']):
                    st.text_input(
                        f"图片 {idx+1}: {img_alt['filename']}",
                        value=img_alt['alt'],
                        key=f"alt_{idx}"
                    )
            
            # 一键复制所有元数据
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
        
        # HTML内容展示
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

# ==========================================
# 底部信息
# ==========================================
st.divider()
st.caption(f"🦁 {cinfo['name']} 运营中台 V29.1 | Powered by {engine_choice} ({sel_model})")
