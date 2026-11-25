# ComfyUI-YesAPI

面向 **[Yesapi AI Model Router (`https://yesapi.online`)](https://yesapi.online)** 的全功能 ComfyUI 自定义节点套件。

一套插件完整覆盖 **独立纯文生图、多图参考生图与局部遮罩重绘（Nano-banana / GPT-Image 全系列）**、**CSV/Excel 与文件夹自动化批量流水线**、**服务端异步任务队列**、**OpenAI + Anthropic 双协议 LLM/VLM 多模态对话与写词导表**、以及 **`/v1/videos` 视频同步与异步生成**，共计 **19 个节点**。

---

## ✨ 核心特性

- **统一生图与编辑引擎**：无需第三方图床中转，纯文生图自动走 `/v1/images/generations`，连接 1~14 张参考图或 `MASK` 遮罩时自动走 `/v1/images/edits`（`multipart/form-data`）；**画幅比例（`aspect_ratio`：`auto` / `1:1` / `16:9` / `9:16` / `4:3` / `3:4` / `3:2` / `2:3` / `21:9`）与分辨率档位（`resolution`：`默认` / `512 (小)` / `1K` / `2K` / `4K`）完全独立分开选择**，底层自动映射为与官方工作台一致的精确像素分辨率，同时保留 `custom_size` 自定义像素覆盖。
- **防截断自动压图**：自动检测参考图体积，超过 `19MB` 时自动高质量缩放/压缩，避免触发平台 `20MB` 截断导致识图失败。
- **分组 API Key 隔离与防泄露**：针对 Yesapi「API Key 绑定模型分组」机制，支持在本地 `config.json`（已被 `.gitignore` 严格排除，绝不会误传 GitHub）中按 Gemini、Gemini-VT、ChatGPT、Claude、国模CN 分别配置 LLM Key，并保留 image、通用 LLM、video Key 作为回退。
- **模型永不过时**：内置平台模型广场最新主力型号下拉框，同时所有节点均提供 `custom_model` 输入框（填写后优先覆盖下拉选项），并附带 `YesAPIModelList` 节点一键拉取当前 Key 可用模型列表。
- **批量防崩与智能熔断**：CSV 自动兼容 UTF-8 与中文 Windows Excel 的 GBK 编码；路径输入自动剥离首尾双引号；批量任务遇到 `402` / `API_KEY_QUOTA_EXHAUSTED` / `分组不匹配` 立即触发熔断停止后续请求，避免无效重试。
- **双协议大语言与视觉模型**：原生支持 OpenAI `/chat/completions` 与 Anthropic `/messages`（`claude-*` 模型自动匹配），最多可连 6 张参考图做视觉反推，并可直接批量写词导出 `.csv` / `.xlsx` 串联下游批量生图。

---

## 📦 安装方法

### 方式一：通过 Git 克隆（推荐）
进入你的 `ComfyUI/custom_nodes` 目录执行：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-YesAPI.git
cd ComfyUI-YesAPI
pip install -r requirements.txt
```

### 方式二：手动复制目录
将 `ComfyUI-YesAPI` 文件夹整体复制到 `ComfyUI/custom_nodes/ComfyUI-YesAPI`，使用 ComfyUI 对应的 Python 环境安装 `requirements.txt` 后重启 ComfyUI。

---

## 🔑 配置 API Key（重要：分组说明）

在 Yesapi 平台上，**不同令牌（API Key）绑定不同的模型分组**：
- 生图模型（`Nano-banana-Pro`、`Nano-banana-2.1`、`gpt-image-2.x` 等）需要使用绑定 **`[生图Image]`** 分组的 Key；
- 文本/多模态模型（Gemini、GPT、Claude、DeepSeek、Kimi、GLM）需要使用绑定对应文本分组的 Key。

### 本地安全配置步骤
1. 将插件目录下的 `config.example.json` 复制一份并重命名为 **`config.json`**（该文件已在 `.gitignore` 中忽略，推送到 GitHub 时不会上传）：

```json
{
  "base_url": "https://yesapi.online",
  "default_api_key": "",
  "image_api_key": "sk-填入你的[生图Image]分组Key",
  "llm_api_key": "sk-通用LLM回退Key",
  "gemini_api_key": "sk-Gemini分组Key",
  "gemini_vt_api_key": "sk-Gemini-VT分组Key（可选）",
  "openai_api_key": "sk-ChatGPT分组Key",
  "claude_api_key": "sk-Claude分组Key",
  "cn_api_key": "sk-国模CN分组Key",
  "video_api_key": "sk-填入你的视频分组Key",
  "timeout_seconds": 180
}
```

2. **Key 读取优先级**（从高到低）：
   1. 节点面板上的 `api_key` 输入框（非空时最高优先）
   2. 对应模型组 `config.json` Key（`gemini_api_key` / `gemini_vt_api_key` / `openai_api_key` / `claude_api_key` / `cn_api_key`）
   3. 对应模型组环境变量（`YESAPI_GEMINI_KEY` / `YESAPI_GEMINI_VT_KEY` / `YESAPI_OPENAI_KEY` / `YESAPI_CLAUDE_KEY` / `YESAPI_CN_KEY`）
   4. `llm_api_key` / `YESAPI_LLM_KEY` 通用 LLM Key
   5. `default_api_key` / `YESAPI_API_KEY`（image/video 节点会先查各自类别 Key）

下拉框中的 `[Gemini]`、`[ChatGPT]`、`[Claude]` 等标签只用于区分分组，发送请求前会移除标签。LLM 节点按真实模型 ID 自动选组 Key；留空专用 Key 时回退到通用 `llm_api_key`。模型目录已按当前公开模型广场接口去重；平台更新后仍可用 `custom_model` 或模型查询节点。

---

## 🧩 全量 19 个节点一览

### 一、`YesAPI/图像`（同步与批量，6 个节点）
| 节点类名 | ComfyUI 显示名称 | 功能说明 |
|---|---|---|
| `YesAPIImageGenerate` | 🎨 YesAPI 图像生成与编辑 | 最多 14 张参考图融合 + `MASK` 局部重绘；支持并发多次生成与自定义尺寸/模型 |
| `YesAPIImageText2Image` | 🎨 YesAPI 纯文生图 | 独立纯文生图输入，不带参考图或遮罩接口 |
| `YesAPIImageBatchCSV` | 📊 YesAPI 图像批量生成 (CSV/Excel) | 读取 `.csv`/`.xls`/`.xlsx` 指定列批量并发生成，支持单条多次执行、最多 4 张公共参考图与配额熔断 |
| `YesAPIImageSaveWithPrompt` | 🏷️ YesAPI 图像批量生成 (命名细化版) | 批量生成并按清洗后的提示词自动命名保存无黑边原图 PNG（支持保存/预览双模式，输出文件路径列表） |
| `YesAPIImageBatchDir` | 📂 YesAPI 文件夹批量处理 (CSV) | 遍历本地图片文件夹 × CSV/固定提示词批量图生图，含坏图预检、3 次重试、熔断保护与错误日志输出 |
| `YesAPISaveImageByFilename` | 💾 YesAPI 按文件名智能保存 | 配合上游 `filenames` 列表（例如高清放大后）按原对应名称追加后缀防覆盖保存 |

### 二、`YesAPI/图像异步`（服务端异步队列，3 个节点）
| 节点类名 | ComfyUI 显示名称 | 功能说明 |
|---|---|---|
| `YesAPIImageAsyncSubmit` | 🚀 YesAPI 图像异步提交 | 以 `async=true` 秒级提交生图/重绘任务到本地队列，不阻塞 ComfyUI 画布 |
| `YesAPIImageAsyncBatchSubmit` | 🚀 YesAPI 图像异步批量提交 (CSV/Excel) | 将表格大批量提示词快速投递至服务端异步队列 |
| `YesAPIImageAsyncQuery` | 📥 YesAPI 图像异步查询下载 | 刷新所有异步任务进度并自动下载已完成图像（注：平台异步结果保留 **2 小时**，超时会有醒目提示） |

### 三、`YesAPI/语言模型`（LLM / VLM / 表格批处理，5 个节点）
| 节点类名 | ComfyUI 显示名称 | 功能说明 |
|---|---|---|
| `YesAPILLMChat` | 💬 YesAPI LLM 文本对话 | 支持 `auto` / `openai`（`/chat/completions`）/ `anthropic`（`/messages`）双协议切换 |
| `YesAPIVLMChat` | 👁️ YesAPI VLM 多图视觉对话 | 最多连接 6 张图像进行多图对比、反推提示词或视觉问答（未连图片时自动作为纯文本对话运行） |
| `YesAPILLMWriter` | ✍️ YesAPI AI 批量写词导出表格 | 让大模型按主题或参考图批量编写提示词并直接保存为 `.csv`/`.xlsx`，无缝连入下游批量生图节点 |
| `YesAPILLMBatchCSV` | 📊 YesAPI LLM 批量表格处理 (CSV/Excel) | 并发读取表格列批量翻译、改写或扩写分镜提示词，导出新表格 |
| `YesAPIModelList` | 🔍 YesAPI 可用模型查询 | 调用 `/v1/models` 列出当前分类 Key 实际可用的完整模型 ID，支持关键字筛选 |

### 四、`YesAPI/视频`（`/v1/videos` 协议，5 个节点）
| 节点类名 | ComfyUI 显示名称 | 功能说明 |
|---|---|---|
| `YesAPIVideoGenerate` | 🎬 YesAPI 视频生成 (同步等待) | 提交视频任务、带 ComfyUI 进度条轮询并流式下载 `.mp4`，输出原生 `VIDEO` 对象（支持首尾帧与音频直链） |
| `YesAPIVideoAsyncSubmit` | 🚀 YesAPI 视频异步提交 | 提交单条视频生成任务并记录至本地视频队列 |
| `YesAPIVideoAsyncBatchSubmit` | 🚀 YesAPI 视频异步批量提交 (CSV/Excel) | 从 CSV/Excel 批量投递视频生成任务 |
| `YesAPIVideoQueryTasks` | 📋 YesAPI 视频任务状态查询 | 刷新并展示本地视频任务队列进度 |
| `YesAPIVideoGetNext` | 📥 YesAPI 获取下一个完成视频 | 自动流式下载队列中最早完成的视频任务并输出 `VIDEO` 对象 |

> **提示**：视频节点已按 Yesapi `/v1/videos` 官方接口规范全量就绪；若平台模型广场「视频」分类当前暂无在售渠道，调用时会给出明确的模型未上架提示，待渠道开启或在 `custom_model` 填入在售视频模型名即可直接使用。

---

## 📂 示例工作流

`workflows/` 目录下内置了 5 个纯原生节点搭配的示例工作流（无第三方缺失节点依赖），直接拖入 ComfyUI 画布即可体验：
1. `1_图像生成与多图遮罩编辑.json`
2. `2_AI写词与CSV批量生图流水线.json`
3. `3_图像异步提交与下载队列.json`
4. `4_LLM与VLM多图对话及模型查询.json`
5. `5_视频生成与异步队列.json`

---

## 🧪 运行自动化测试

```bash
cd ComfyUI-YesAPI
pytest -v
```

---

## 📄 开源协议

本项目基于 [MIT License](LICENSE) 开源。
