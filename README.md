# 部署指南（更新版）— Gemini + 通义千问

## 修改说明
本次更新将 Claude API 替换为：
- **图片识别** → Google Gemini（gemini-1.5-flash）
- **文字翻译+优化** → 通义千问（qwen-plus）

---

## 需要准备的 API Key

### 1. Google Gemini API Key
- 打开：https://aistudio.google.com/app/apikey
- 点击 "Create API key"，复制

### 2. 通义千问 API Key
- 打开：https://dashscope.console.aliyun.com
- 左侧菜单 → "API-KEY管理" → 创建新Key，复制

---

## Streamlit Cloud 配置

在 Streamlit Cloud → App Settings → Secrets 中填入：

```toml
GEMINI_API_KEY = "AIzaxxxxxxxxxxxxxxxx"
QIANWEN_API_KEY = "sk-xxxxxxxxxxxxxxxx"
APP_PASSWORD = "你设置的访问密码"
```

---

## 更新步骤

1. 将新的 `app.py` 和 `requirements.txt` 上传到 GitHub 仓库（覆盖旧文件）
2. 在 Streamlit Cloud Secrets 中添加 `GEMINI_API_KEY` 和 `QIANWEN_API_KEY`
3. 删除旧的 `ANTHROPIC_API_KEY`（不再需要）
4. Streamlit 会自动重新部署

---

## 费用参考（仅供参考）
- Gemini 1.5 Flash：有免费额度，超出后约 $0.075/百万tokens
- 通义千问 Plus：约 ¥0.004/千tokens（非常便宜）
- 每次处理一篇内容大约花费 ¥0.05–0.2
