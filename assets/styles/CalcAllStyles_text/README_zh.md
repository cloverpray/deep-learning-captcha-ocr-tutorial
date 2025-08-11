# deep-learning-captcha-ocr-tutorial

> 从零开始训练一个基于深度神经网络的**图形验证码识别**项目（学习/练习用）。仅供学习与研究，**不得用于任何非法用途**。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c)](https://pytorch.org/)
[![Server](https://img.shields.io/badge/Server-FastAPI%20%7C%20Flask-00b894)](#step1)
[![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](#)

---

<a id="toc"></a>
## 目录

- [特性概览](#features)
- [仓库结构](#tree)
- [Step 1 · Ready‑to‑Run（开箱即用）](#step1)
- [Step 2 · Train‑with‑Dataset（本地训练）](#step2)
- [Step 3 · Build‑from‑Scratch（数据合成 → 训练 → 使用）](#step3)
- [评估与基准（可选）](#bench)
- [导出与部署（可选）](#deploy)
- [常见问题（FAQ）](#faq)
- [路线图（Roadmap）](#roadmap)
- [致谢](#ack)
- [许可](#license)

---

<a id="features"></a>
## ✨ 特性概览
- 支持多种验证码风格：EasyCaptcha、CalcAllStyles_text、Ruoyi 风格
- 模型可选：`vanilla_crnn` / `crnn_resnet34` / `svtr_tiny`
- 统一的训练接口与评估指标（EM/CER）
- 可选 **FastAPI** 或 **Flask** 作为服务端；含最小化前端 Demo
- 可扩展的 **Java 生成器** 与 **数据合并去重脚本**

---

<a id="tree"></a>
## 🗂️ 仓库结构
```
.
├─ demo/                      # Step 1：开箱即用（最小服务 + 简单前端）
│  ├─ server_fastapi.py       # 选择其一运行
│  ├─ server_flask.py         # 选择其一运行
│  └─ requirements.txt        # 推理/服务端最小依赖（FastAPI 或 Flask 版本）
├─ training-ready/            # Step 2：数据就绪后直接训练
│  ├─ dataset/                # 训练/验证集（或下载脚本）
│  ├─ train.py                # 训练入口（调用 full-pipeline 中的核心逻辑）
│  └─ eval.py                 # 评估入口（EM/CER）
├─ full-pipeline/             # Step 3：完整流程（原始工程）
│  ├─ captcha_ocr/            # 数据集/模型/工具等
│  ├─ tools/                  # Java 生成器、合并去重、导出脚本
│  ├─ train.py                # 训练
│  ├─ infer.py                # 推理（含解码逻辑）
│  ├─ server_fastapi.py       # FastAPI 服务端
│  ├─ server_flask.py         # Flask 服务端
│  └─ requirements.txt        # 全量依赖（训练 + 推理 + 数据处理）
├─ hf_spaces/                 # Hugging Face Spaces Demo（Gradio）
│  ├─ app.py
│  └─ requirements.txt
├─ .github/workflows/ci.yml   # Actions：最小冒烟测试（可选）
├─ README.md                  # ← 本文件
├─ LICENSE                    # MIT
└─ requirements.txt           # 可选：引用 full-pipeline/requirements.txt
```

---

## 支持的验证码风格（示例）
<p>
  <img src="assets/styles/CalcAllStyles_text/te_00000413.png" width="120" />
  <img src="assets/styles/CalcAllStyles_text/te_00000441.png" width="120" />
  <img src="assets/styles/CalcAllStyles_text/te_00000497.png" width="120" />
  <img src="assets/styles/CalcAllStyles_text/te_00001181.png" width="120" />
  <img src="assets/styles/CalcAllStyles_text/te_00001623.png" width="120" />
  <img src="assets/styles/CalcAllStyles_text/te_00001675.png" width="120" />
  <img src="assets/styles/CalcAllStyles_text/te_00001846.png" width="120" />
  <img src="assets/styles/EasyCaptcha/te_00000002.png" width="120" />
  <img src="assets/styles/EasyCaptcha/te_00000004.png" width="120" />
  <img src="assets/styles/EasyCaptcha/te_00000028.png" width="120" />
  <img src="assets/styles/EasyCaptcha/te_00000048.png" width="120" />
  <img src="assets/styles/EasyCaptcha/te_00000463.png" width="120" />
</p>

---

## 前端页面 Demo 截图
<p><img src="assets/page_demo.jpg" width="780" /></p>

---

<a id="step1"></a>
## 🚀 Step 1 · Ready‑to‑Run（开箱即用）
选择 **FastAPI** 或 **Flask** 方式其一：

### 方式 A：FastAPI + Uvicorn
**demo/requirements.txt（示例）**
```txt
torch>=2.3.0
Pillow>=10.2.0
fastapi>=0.111.0
uvicorn>=0.30.0
python-multipart>=0.0.9
```
**启动服务**
```bash
cd demo
pip install -r requirements.txt
uvicorn server_fastapi:app --host 0.0.0.0 --port 8000
```
> 如遇 `Form data requires "python-multipart"` 报错，请安装/确认 `python-multipart` 已就绪。

### 方式 B：Flask（最小依赖）
**demo/requirements.txt（示例）**
```txt
torch>=2.3.0
Pillow>=10.2.0
flask>=3.0.0
```
**启动服务**
```bash
cd demo
pip install -r requirements.txt
python server_flask.py --host 0.0.0.0 --port 8001
```
**（可选）前后端分离**
```bash
# 内网：FastAPI 8000；对外：Flask 8001，Flask 通过 --upstream 代理后端
python server_flask.py --host 0.0.0.0 --port 8001 --upstream http://127.0.0.1:8000
```
浏览器打开 `http://127.0.0.1:8001`，支持拖拽上传。

---

<a id="step2"></a>
## 🧪 Step 2 · Train‑with‑Dataset（本地训练）
### 依赖
参考 `full-pipeline/requirements.txt`（包含训练/增强/评估所需）：
```txt
torch>=2.3.0
torchvision>=0.18.0
albumentations>=1.4.0
opencv-python>=4.9.0
numpy>=1.26.0
Pillow>=10.2.0
tqdm>=4.66.0
rich>=13.7.0
pandas>=2.2.0
fastapi>=0.111.0
uvicorn>=0.30.0
python-multipart>=0.0.9
flask>=3.0.0
```

### 训练命令（示例）
```bash
# vanilla_crnn（基础 CRNN）
python train.py --arch vanilla_crnn --train-root dataset/train --val-root dataset/test \
  --imgH 60 --imgW 256 --pad-multiple 4 --downsample 4 --batch 192 --epochs 30 --aug --num-workers 1

# crnn_resnet34（ResNet34 特征）
python train.py --arch crnn_resnet34 --train-root dataset/train --val-root dataset/test \
  --imgH 60 --imgW 256 --pad-multiple 4 --downsample 4 --batch 192 --epochs 30 --aug --num-workers 1

# svtr_tiny（轻量视觉 Transformer）
python train.py --arch svtr_tiny --train-root dataset/train --val-root dataset/test \
  --imgH 60 --imgW 256 --pad-multiple 4 --downsample 4 --batch 192 --epochs 30 --aug --num-workers 1

# 一次性连续训练三种
python train.py --arch all --train-root dataset/train --val-root dataset/test \
  --imgH 60 --imgW 256 --pad-multiple 4 --downsample 4 --batch 192 --epochs 30 --aug --num-workers 1
```
> ✅ 批大小建议：`--batch 128`（更省显存）通常与 `--batch 192` 精度相近。若降至 `--batch 64`，建议适当增加 `--epochs`（+10–20%）以对齐效果。

---

<a id="step3"></a>
## 🧬 Step 3 · Build‑from‑Scratch（数据合成 → 训练 → 使用）
### 3.1 生成 EasyCaptcha 风格
```bash
# 编译
javac -encoding UTF-8 -cp ".;lib/*" EasyCaptchaRunner.java

# 列出可用类型
java -Dfile.encoding=UTF-8 -cp ".;lib/*" EasyCaptchaRunner list

# 基于配置文件批量生成（混合）
java -Dfile.encoding=UTF-8 -cp ".;lib/*" EasyCaptchaRunner fromjson config.json easy_train 200000 --threads 8
java -Dfile.encoding=UTF-8 -cp ".;lib/*" EasyCaptchaRunner fromjson config.json easy_test  20000  --threads 8
```

### 3.2 生成 CalcAllStyles_text 风格
```bash
# 编译
javac -encoding UTF-8 -cp "CalculateCaptcha-1.1.jar" CalcAllStylesRunner.java

# 示例：全部文本类型一次生成
java -cp ".;CalculateCaptcha-1.1.jar" CalcAllStylesRunner all out_all_text 10 160 60 0

# 大规模生成（示例）
java -cp ".;CalculateCaptcha-1.1.jar" CalcAllStylesRunner smart ctext_train 162000 --fonts "Arial,Microsoft YaHei,SimSun,Noto Sans CJK SC" --threads 8
java -cp ".;CalculateCaptcha-1.1.jar" CalcAllStylesRunner smart ctext_test   16200  --fonts "Arial,Microsoft YaHei,SimSun,Noto Sans CJK SC" --threads 8
```

### 3.3 生成 Ruoyi 风格
```bash
# 编译
javac -encoding UTF-8 RuoyiJarNormalizeSmart.java

# 训练/测试集（自动尺寸分布，示例线程数 8）
java -Dfile.encoding=UTF-8 -cp ".;CalculateCaptcha-1.1.jar" RuoyiJarNormalizeSmart smart ruoyi_train 100000 --jar CalculateCaptcha-1.1.jar --sizes "160x60:40,180x64:20,200x64:15,256x64:10,120x40:10,96x32:5" --threads 8
java -Dfile.encoding=UTF-8 -cp ".;CalculateCaptcha-1.1.jar" RuoyiJarNormalizeSmart smart ruoyi_test  10000  --jar CalculateCaptcha-1.1.jar --sizes "160x60:40,180x64:20,200x64:15,256x64:10,120x40:10,96x32:5" --threads 8
```

### 3.4 合并与去重
```bash
python tools/merge_datasets.py \
  --train-dest dataset/train --test-dest dataset/test \
  --train-src easy  tools/java_captcha/EasyCaptcha/images/easy_train \
  --train-src ctext tools/java_captcha/CalculateCaptch/images/ctext_train \
  --train-src ruoyi tools/java_captcha/CalculateCaptch/images/ruoyi_train \
  --test-src  easy  tools/java_captcha/EasyCaptcha/images/easy_test \
  --test-src  ctext tools/java_captcha/CalculateCaptch/images/ctext_test \
  --test-src  ruoyi tools/java_captcha/CalculateCaptch/images/ruoyi_test \
  --copy-mode copy --dedupe --seed 123
```
> 提示：合并/去重可能较慢，耐心等待；确保磁盘有足够空间。

---

<a id="bench"></a>
## 📊 评估与基准（可选）
- **指标**：EM（exact match）、CER（character error rate）
- **建议提供**：CPU/GPU 推理延迟、吞吐、显存占用（不同 batch）、不同风格/尺寸下的鲁棒性对比

| Model           | EM@test | CER@test | Latency (CPU) | Latency (GPU) | Params |
|-----------------|---------|----------|---------------|---------------|--------|
| vanilla_crnn    |         |          |               |               |        |
| crnn_resnet34   |         |          |               |               |        |
| svtr_tiny       |         |          |               |               |        |

> 可提供 `scripts/benchmark.py` 自动生成表格到 README。

---

<a id="deploy"></a>
## 📦 导出与部署（可选）
- **TorchScript/ONNX** 导出：便于跨语言/跨平台部署
- **Docker**：提供 `Dockerfile` 与 `docker-compose.yml`（可选）
- **Hugging Face Spaces**：使用 `hf_spaces/` 的 Gradio App 一键上线 Demo

---

<a id="faq"></a>
## 🔧 常见问题（FAQ）
- **启动 FastAPI 报错 `python-multipart`**：请安装 `python-multipart>=0.0.9`。
- **Windows 上 Java 编译乱码**：加 `-encoding UTF-8`，或设置 `-Dfile.encoding=UTF-8`。
- **端口占用**：修改 `--port` 或释放对应端口。
- **显存不足**：减小 `--batch`，或降低 `--imgW`，或启用混合精度训练。

---

<a id="roadmap"></a>
## 🗺️ 路线图（Roadmap）
- [ ] Web 前端 Demo 增强（拖拽、批量推理、可视化）
- [ ] 导出 ONNX / NCNN / TFLite 示例
- [ ] 统一基准脚本与排行榜（不同风格/复杂度）
- [ ] 多语言文档完善（EN/中文）

---

<a id="ack"></a>
## 🙏 致谢
- EasyCaptcha、CalculateCaptcha 等开源项目作者与贡献者
- 社区反馈与 PR

---

<a id="license"></a>
## 📝 许可
- 代码：MIT（见 [LICENSE](LICENSE)）
- 数据/权重：请遵循其各自来源许可
