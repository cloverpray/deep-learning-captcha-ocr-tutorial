# -*- coding: utf-8 -*-
from flask import Flask, Response, request, jsonify
import requests
import os

# ================== 反向代理配置 ==================
# 默认把 /api/* 转发到本机 FastAPI（内网），可用环境变量或启动参数覆盖
API_UPSTREAM = os.environ.get("API_UPSTREAM", "http://127.0.0.1:8000")

# 前端纯页面：浏览器 JS 调用同源 /api，再由本服务转发到 FastAPI
app = Flask(__name__)

HTML_PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>验证码识别（前端 Demo）</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
body{font-family: system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;margin:40px;color:#222;}
h1{font-size:22px}
fieldset{border:1px solid #ddd;padding:16px;border-radius:10px;max-width:960px;}
label{display:inline-block;min-width:120px;}
input,select{padding:6px 8px;margin:6px 0;}
button{padding:8px 14px;cursor:pointer;}
.small{color:#666;font-size:12px;}
.controls{margin-bottom:12px;}
.badge{display:inline-block;padding:2px 8px;margin-left:8px;border:1px solid #ddd;border-radius:20px;font-size:12px;color:#666;}
/* 拖拽上传 + 预览条 */
.dropzone{border:2px dashed #bbb;border-radius:10px;padding:14px;text-align:center;color:#666;}
.dropzone.drag{border-color:#4a90e2;background:#eef5ff;}
.preview-strip{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.preview-strip img{width:80px;height:auto;border:1px solid #ddd;border-radius:6px;}
/* 历史/结果表 */
table{border-collapse:collapse;margin-top:20px;width:100%;}
td,th{border:1px solid #ccc;padding:6px 10px;vertical-align:top;}
.thumb-s{width:110px;height:auto;border:1px solid #ddd;border-radius:6px;}
.success{color:#0a7a0a;}
.error{color:#b00020;}
/* 预测列不换行并适当放大 */
th.pred-col,td.pred-col{min-width:220px;white-space:nowrap;}
.predtxt{white-space:nowrap;font-weight:700;font-size:16px;letter-spacing:.5px;}
</style>
<script>
// ==================== 配置与工具 ====================
// 说明：如果 URL 有 ?api=xxx，则直连该地址；否则默认同源 '/api'（Flask 反向代理到内网 FastAPI）
const LS_ITEMS_KEY = 'captcha_hist_items_v4';     // 单条记录：{ts, arch, srcW, srcH, filename, dataUrl, pred, error}
const LS_LAST_ARCH = 'captcha_last_arch';
const MAX_HISTORY  = 500;

let filesMap = new Map();  // key -> File（避免重复）
let filesMeta = {};        // filename -> { dataUrl, width, height }

function getApiBase(){
  const p = new URLSearchParams(location.search).get('api');
  if (p && p.trim()) return p.trim();     // 显式覆盖
  return location.origin + '/api';        // 默认同源反代
}
function fileKey(f){ return `${f.name}::${f.size}::${f.lastModified}`; }

// 将文件读取为 dataURL，并获取真实尺寸（width/height）
function filesToMetaMap(fileList){
  const arr = Array.from(fileList || []);
  return Promise.all(arr.map(f => new Promise((resolve,reject)=>{
    const fr = new FileReader();
    fr.onload = () => {
      const img = new Image();
      img.onload = ()=> resolve({name:f.name, dataUrl: fr.result, width: img.naturalWidth, height: img.naturalHeight});
      img.onerror = ()=> resolve({name:f.name, dataUrl: fr.result, width: 0, height: 0});
      img.src = fr.result;
    };
    fr.onerror = reject;
    fr.readAsDataURL(f);
  }))).then(list=>{
    const map = {};
    list.forEach(o => { map[o.name] = {dataUrl:o.dataUrl, width:o.width, height:o.height}; });
    return map;
  });
}

// ==================== 模型列表 ====================
async function refreshModels(){
  const API_BASE = getApiBase();
  try{
    const res = await fetch(API_BASE + '/models');
    const data = await res.json();
    const sel = document.getElementById('arch');
    const cur = localStorage.getItem(LS_LAST_ARCH) || sel.value;
    sel.innerHTML = '';
    (data || []).forEach(a=>{
      const o=document.createElement('option'); o.value=a; o.textContent=a; sel.appendChild(o);
    });
    if ((data || []).includes(cur)) sel.value = cur;
    document.getElementById('status').textContent = '模型列表已刷新（来自 ' + API_BASE + '）';
  }catch(e){
    document.getElementById('status').textContent = '刷新失败：' + e;
  }
}
function bindModelChange(){
  const sel = document.getElementById('arch');
  sel.addEventListener('change', ()=> localStorage.setItem(LS_LAST_ARCH, sel.value || ''));
}

// ==================== 历史记录（localStorage） ====================
function loadHistory(){
  try{ return JSON.parse(localStorage.getItem(LS_ITEMS_KEY) || '[]'); }
  catch(e){ return []; }
}
function saveHistory(list){
  // 全局去重（按 dataUrl 优先，其次文件名+尺寸+模型+预测）
  const seen = new Set();
  const deduped = [];
  for (const it of list){
    const key = it.dataUrl ? ('U:'+it.dataUrl) : ('F:'+[it.filename,it.srcW,it.srcH,it.arch,it.pred].join('|'));
    if (seen.has(key)) continue;
    seen.add(key); deduped.push(it);
    if (deduped.length >= MAX_HISTORY) break;
  }
  localStorage.setItem(LS_ITEMS_KEY, JSON.stringify(deduped));
}
function addHistoryItemsFirst(items){
  const cur = loadHistory();
  saveHistory((items || []).concat(cur)); // 新的在前
}
function clearHistory(){
  localStorage.removeItem(LS_ITEMS_KEY);
  renderHistory();
}
function fmtTime(ts){
  const d=new Date(ts); const p=n=> (n<10?'0'+n:n);
  return d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate())+' '+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());
}
function renderHistory(){
  const list = loadHistory();
  const tbody = document.getElementById('hist_body');
  tbody.innerHTML = '';
  list.forEach(it=>{
    const tr = document.createElement('tr');

    const tdImg = document.createElement('td');
    const img = document.createElement('img');
    img.src = it.dataUrl || '';
    img.className = 'thumb-s';
    tdImg.appendChild(img);

    const tdName = document.createElement('td'); tdName.textContent = it.filename || '';

    const tdPred = document.createElement('td'); tdPred.className='pred-col';
    if (it.error){
      tdPred.innerHTML = '<span class="error predtxt">'+ (it.error||'') +'</span>';
    }else{
      tdPred.innerHTML = '<span class="success predtxt">'+ (it.pred||'') +'</span>';
    }

    const tdModel = document.createElement('td'); tdModel.textContent = it.arch || '';
    const tdSize  = document.createElement('td');
    const sw = it.srcW || '-'; const sh = it.srcH || '-';
    tdSize.textContent = (sw||'-') + '×' + (sh||'-');

    const tdTime  = document.createElement('td'); tdTime.textContent = fmtTime(it.ts || Date.now());

    tr.appendChild(tdImg);
    tr.appendChild(tdName);
    tr.appendChild(tdPred);
    tr.appendChild(tdModel);
    tr.appendChild(tdSize);
    tr.appendChild(tdTime);
    tbody.appendChild(tr);
  });
}

// ==================== 拖拽上传 & 选择上传（带预览条） ====================
function attachDropzone(){
  const dz = document.getElementById('dropzone');
  const fileInput = document.getElementById('files');

  dz.addEventListener('click', ()=> fileInput.click());
  dz.addEventListener('dragover', e=>{ e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', e=> dz.classList.remove('drag'));
  dz.addEventListener('drop', async e=>{
    e.preventDefault(); dz.classList.remove('drag');
    await addFiles(e.dataTransfer.files);
  });

  fileInput.addEventListener('change', async e=>{
    await addFiles(e.target.files);
    fileInput.value=''; // 允许再次选择同一文件
  });
}

function fileKey(f){ return `${f.name}::${f.size}::${f.lastModified}`; }

async function addFiles(list){
  const arr = Array.from(list || []);
  if (!arr.length) return;

  // 去重加入 filesMap
  const newOnes = [];
  for (const f of arr){
    const k = fileKey(f);
    if (!filesMap.has(k)){ filesMap.set(k, f); newOnes.push(f); }
  }
  if (!newOnes.length) return;

  // 读取 meta（dataUrl + 宽高）
  const metas = await filesToMetaMap(newOnes);
  Object.assign(filesMeta, metas);

  // 若是首次加入，自动用第一张图的真实尺寸填入 H/W
  if (filesMap.size === newOnes.length){
    const first = newOnes[0];
    const m = filesMeta[first.name];
    if (m && m.width && m.height){
      document.getElementById('imgH').value = m.height;
      document.getElementById('imgW').value = m.width;
    }
  }

  // 更新预览条
  renderPreviewStrip();
}

function renderPreviewStrip(){
  const strip = document.getElementById('previewStrip');
  strip.innerHTML = '';
  for (const [,f] of filesMap){
    const m = filesMeta[f.name] || {};
    const url = m.dataUrl;
    const img = document.createElement('img');
    img.src = url; img.title = f.name + (m.width&&m.height?`（${m.width}×${m.height}）`:'');
    strip.appendChild(img);
  }
}

// ==================== 识别流程（结果写入“历史/结果表”） ====================
async function doPredict(ev){
  ev.preventDefault();
  const API_BASE = getApiBase();
  const arch = document.getElementById('arch').value || 'svtr_tiny';
  const imgH = parseInt(document.getElementById('imgH').value || '32', 10);
  const imgW = parseInt(document.getElementById('imgW').value || '160', 10);

  if (!filesMap.size){
    const inputFiles = document.getElementById('files').files;
    if (!inputFiles || !inputFiles.length){ alert('请先选择或拖拽图片'); return; }
    await addFiles(inputFiles);
  }

  // 发送请求（后端会固定高、宽自适应）
  const fd = new FormData();
  fd.append('arch', arch);
  fd.append('imgH', imgH);
  fd.append('imgW', imgW);
  for (const [,f] of filesMap){ fd.append('files', f); }

  document.getElementById('status').textContent = '识别中...';
  try{
    const res = await fetch(API_BASE + '/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error){
      document.getElementById('status').textContent = '出错：' + data.error;
      return;
    }

    // 写入历史（置顶），并全局去重
    const now = Date.now();
    const items = (data.results || []).map(r=>{
      const m = filesMeta[r.filename] || {};
      return {
        ts: now,
        arch: arch,
        srcW: m.width || null,
        srcH: m.height || null,
        filename: r.filename,
        dataUrl: (m.dataUrl || ''),
        pred: r.error ? '' : (r.pred || ''),
        error: r.error || ''
      };
    });
    addHistoryItemsFirst(items);
    renderHistory();

    // 记录最近一次选择的模型
    localStorage.setItem(LS_LAST_ARCH, arch);

    document.getElementById('status').textContent =
      '完成：'+ (data.count || 0) +' 张（模型：'+ arch +'，输入尺寸：'+ imgW+'×'+imgH +'）';

    // ✅ 识别后清空选择，避免下次重复加入同一批文件
    filesMap.clear();
    filesMeta = {};
    renderPreviewStrip();

  }catch(e){
    document.getElementById('status').textContent = '请求失败：' + e;
  }
}

// ==================== 启动 ====================
document.addEventListener('DOMContentLoaded', ()=>{
  attachDropzone();
  bindModelChange();
  refreshModels();
  renderHistory();
});
</script>
</head>
<body>
<h1>验证码识别（前端 Demo）</h1>
<p class="small">
  默认通过同源 <code>/api</code> 反向代理到内网 FastAPI（无需暴露 8000）。<br/>
  如需直连其它后端，可在 URL 加 <code>?api=后端地址</code> 覆盖，比如 <code>?api=http://192.168.1.8:9000</code>。
</p>

<div class="controls">
  <button type="button" onclick="refreshModels()">刷新模型列表</button>
  <span id="status" class="small"></span>
</div>

<form id="frm" onsubmit="doPredict(event)" enctype="multipart/form-data">
  <fieldset>
    <legend>上传图片</legend>
    <div>
      <label for="arch">模型</label>
      <select name="arch" id="arch"><!-- 刷新后填充，默认选中上一次用过的模型 --></select>
      <span class="small">（来自 /api/models 或 ?api=/models）</span>
    </div>
    <div>
      <label for="imgH">图高 H</label>
      <input type="number" name="imgH" id="imgH" value="32" min="16" max="1024" />
    </div>
    <div>
      <label for="imgW">图宽 W</label>
      <input type="number" name="imgW" id="imgW" value="160" min="32" max="4096" />
    </div>
    <div>
      <label for="files">选择文件</label>
      <input type="file" name="files" id="files" accept="image/*" multiple />
      <div id="dropzone" class="dropzone small" style="margin-top:6px;">将图片拖拽到此处，或点击此区域选择文件</div>
      <!-- ✅ 缩略图预览条：选择/拖拽后显示在“开始识别”旁 -->
      <div id="previewStrip" class="preview-strip"></div>
    </div>
    <div>
      <button type="submit">开始识别</button>
    </div>
  </fieldset>
</form>

<!-- ✅ 新的“历史/结果表”放在老位置 -->
<fieldset>
  <legend>识别结果（保存在浏览器，最新在最上方）</legend>
  <div class="small">最多保存 <b>500</b> 条 · 仅保存在你的浏览器（显示真实尺寸）</div>
  <div class="controls">
    <button type="button" onclick="renderHistory()">刷新记录</button>
    <button type="button" onclick="clearHistory()">清空记录</button>
  </div>
  <table id="history">
    <colgroup>
      <col style="width:140px">
      <col style="width:360px">
      <col style="width:260px">
      <col style="width:120px">
      <col style="width:120px">
      <col style="width:160px">
    </colgroup>
    <thead>
      <tr><th>缩略图</th><th>文件名</th><th class="pred-col">预测</th><th>模型</th><th>尺寸</th><th>时间</th></tr>
    </thead>
    <tbody id="hist_body"></tbody>
  </table>
</fieldset>

</body>
</html>
'''

# ================== 反向代理到 FastAPI（同源 /api/*） ==================

@app.route("/api/models", methods=["GET"])
def proxy_models():
    try:
        r = requests.get(f"{API_UPSTREAM}/models", timeout=15)
        return Response(r.content, status=r.status_code,
                        mimetype=r.headers.get("content-type","application/json"))
    except Exception as e:
        return jsonify({"error": f"proxy /models failed: {e}"}), 502

@app.route("/api/predict", methods=["POST"])
def proxy_predict():
    try:
        # 普通字段
        data = {k: v for k, v in request.form.items()}
        # 文件字段（读成 bytes 再转发，避免边界问题）
        files = []
        for f in request.files.getlist("files"):
            content = f.read()
            files.append(("files", (f.filename, content, f.mimetype or "application/octet-stream")))
        r = requests.post(f"{API_UPSTREAM}/predict", data=data, files=files, timeout=300)
        return Response(r.content, status=r.status_code,
                        mimetype=r.headers.get("content-type","application/json"))
    except Exception as e:
        return jsonify({"error": f"proxy /predict failed: {e}"}), 502

# 健康检查
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "upstream": API_UPSTREAM})

# 前端页面
@app.route("/", methods=['GET'])
def index():
    return Response(HTML_PAGE, mimetype="text/html; charset=utf-8")

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8001)
    ap.add_argument('--upstream', default=None, help='FastAPI 内网地址，例如 http://127.0.0.1:8000')
    args=ap.parse_args()
    if args.upstream:
        API_UPSTREAM = args.upstream
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
