"""训练监控页面：把指标流渲染成自包含的 HTML + 原生 JS 图表。

页面结构（两列 × 六行，窄屏自动单列）与状态栏按用户给的参考图实现；
所有图表都是原生 canvas 绘制，**不引用任何外部资源**（离线可看、无 CDN），
数据来自 :mod:`beamng_autopilot.experiments.metrics` 的记录流：

* 实时模式：页面从 ``<base>/metrics?since=<seq>`` 拉增量，按 ``seq`` 去重，
  断线自动重连；刷新后从 0 重新拉取历史。
* 快照模式：把已有记录直接嵌进 HTML（训练结束/离线回看），不联网。

三条前端纪律（都来自方案）：

1. **不假装有数据**：字段缺失显示"未采集"、任务不适用显示"不适用"、设备
   不提供显示"该设备未提供功耗数据"，只有真实记录到的 0 才画 0。
2. **平滑不改原始值**：原始曲线始终画在最底层，平滑窗口只影响叠加线；
   统计量（均值/中位/极值/峰值）一律基于原始记录。
3. **轴不混用**：训练曲线用 step 轴，硬件曲线用已运行时间（秒）。

页面只读：不向 autopilot 发任何控制命令，也不写训练产物。
"""

from __future__ import annotations

import json
from pathlib import Path

#: 12 张图（顺序 = 参考图的行列顺序）。前端按这张表建网格，测试也按它断言。
CHART_SPECS = (
    ("loss", "Loss", "loss 值", "step", False, "line"),
    ("acc", "Accuracy（像素准确率）", "%", "step", False, "line"),
    ("grad", "Gradient Norm", "L2 范数", "step", True, "line"),
    ("lr", "Learning Rate", "学习率", "step", True, "line"),
    ("loss_hist", "Loss Distribution", "帧数", "loss 分箱", False, "hist"),
    ("grad_hist", "Gradient Norm Distribution", "帧数", "梯度范数分箱", True,
     "hist"),
    ("gpu_mem", "GPU Memory", "GiB", "已运行时间 (s)", False, "line"),
    ("speed", "Training Speed", "s/it", "step", False, "line"),
    ("gpu_power", "GPU Power", "W", "已运行时间 (s)", False, "line"),
    ("gpu_util", "GPU Utilization", "%", "已运行时间 (s)", False, "line"),
    ("cpu_util", "CPU Utilization", "%", "已运行时间 (s)", False, "line"),
    ("sys_mem", "Memory Usage", "GiB", "已运行时间 (s)", False, "line"),
)


def render_html(*, run_id: str, title: str = "训练监控",
                mode: str = "live", records: list | None = None,
                poll_ms: int = 2000, sample_s: float = 2.0,
                base: str = "") -> str:
    """``mode="live"`` 走增量接口；``mode="snapshot"`` 用嵌入记录。"""
    payload = {
        "mode": mode,
        "run_id": run_id,
        "title": title,
        "poll_ms": int(poll_ms),
        "sample_s": float(sample_s),
        "base": base.rstrip("/"),
        "charts": [{"key": k, "title": t, "unit": u, "xlabel": x,
                    "log": lg, "kind": kind}
                   for (k, t, u, x, lg, kind) in CHART_SPECS],
        "initial": list(records or []),
    }
    return _TEMPLATE.replace("__PAYLOAD__", json.dumps(
        payload, ensure_ascii=False)).replace("__TITLE__", title)


def write_snapshot(path: Path | str, *, run_id: str, records: list,
                   title: str = "训练监控（快照）", sample_s: float = 2.0) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(run_id=run_id, title=title, mode="snapshot",
                             records=records, sample_s=sample_s),
                 encoding="utf-8")
    return p


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{--bg:#f5f6f8;--card:#ffffff;--ink:#1f2430;--muted:#6b7280;
        --line:#e3e6ec;--accent:#2f6df6;--raw:#b8c2d6;--ok:#2e9e6b;
        --warn:#c2410c;--bad:#b91c1c;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:13px/1.5 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  header{position:sticky;top:0;z-index:5;background:var(--card);
         border-bottom:1px solid var(--line);padding:10px 16px}
  .statusbar{display:flex;flex-wrap:wrap;gap:6px 22px;align-items:baseline}
  .statusbar b{font-weight:600}
  .item{white-space:nowrap}
  .item .k{color:var(--muted);margin-right:6px}
  .item .v{font-variant-numeric:tabular-nums;font-weight:600}
  .pill{padding:1px 9px;border-radius:10px;font-size:12px;font-weight:600;
        background:#eef1f6;color:#374151}
  .pill.running{background:#e6f0ff;color:#1d4ed8}
  .pill.completed{background:#e7f7ee;color:#1d7a4d}
  .pill.failed{background:#fdeaea;color:#b91c1c}
  .pill.paused,.pill.waiting{background:#f4efe3;color:#8a6d1f}
  .banner{margin:8px 16px 0;padding:8px 12px;border-radius:8px;font-size:13px}
  .banner.demo{background:#fff7e6;border:1px solid #f3d9a4;color:#8a6d1f}
  .banner.err{background:#fdeaea;border:1px solid #f0b4b4;color:#8c1c1c}
  .banner.note{background:#eef4ff;border:1px solid #cfe0ff;color:#26406f}
  .toolbar{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;
           padding:8px 16px 0;color:var(--muted)}
  .toolbar label{display:flex;gap:6px;align-items:center;white-space:nowrap}
  .toolbar input[type=number]{width:64px;padding:2px 6px;border:1px solid var(--line);
           border-radius:6px;background:#fff;color:var(--ink)}
  .toolbar button{padding:3px 10px;border:1px solid var(--line);background:#fff;
           border-radius:6px;cursor:pointer;color:var(--ink)}
  .toolbar button.on{background:#e6f0ff;border-color:#9dbdf7;color:#1d4ed8}
  main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px 16px 28px}
  @media (max-width:1100px){main{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:8px 10px 6px;min-width:0}
  .card h3{margin:0 0 2px;font-size:13px;font-weight:600}
  .card .sub{color:var(--muted);font-size:12px;margin-bottom:4px}
  .cv{position:relative;width:100%;height:190px}
  .cv canvas{width:100%;height:100%;display:block}
  .empty{position:absolute;inset:0;display:flex;align-items:center;
         justify-content:center;color:var(--muted);font-size:12px;
         text-align:center;padding:0 12px}
  .stats{color:var(--muted);font-size:12px;margin-top:2px;
         font-variant-numeric:tabular-nums}
  .tip{position:fixed;pointer-events:none;background:#1f2430ee;color:#fff;
       padding:5px 8px;border-radius:6px;font-size:12px;display:none;z-index:20;
       font-variant-numeric:tabular-nums}
</style></head>
<body>
<header>
  <div class="statusbar" id="statusbar"></div>
</header>
<div id="banners"></div>
<div class="toolbar">
  <label>平滑窗口（loss）<input id="smoothLoss" type="number" min="1" step="1" value="24"></label>
  <label>平滑窗口（硬件）<input id="smoothHw" type="number" min="1" step="1" value="32"></label>
  <span>窗口</span>
  <button data-win="all" class="on">全部</button>
  <button data-win="500">最近 500 步</button>
  <button data-win="100">最近 100 步</button>
  <label id="devWrap" style="display:none">设备<select id="devSel"></select></label>
  <span id="zoomHint">滚轮缩放时间/step 轴，双击复位</span>
</div>
<main id="grid"></main>
<div class="tip" id="tip"></div>
<script>
const P = __PAYLOAD__;
const state = {recs: [], bySeq: new Set(), task: null, errors: [], problems: [],
               win: 'all', smooth: {loss: 24, hw: 32}, zoom: {}, device: null,
               connected: P.mode === 'snapshot', lastSeq: 0, note: ''};

function put(records){
  for (const r of records || []){
    const s = r.seq;
    if (s === undefined || s === null){ state.recs.push(r); continue; }
    if (state.bySeq.has(s)) continue;         // 按 seq 去重（断线重连/重复拉取）
    state.bySeq.add(s); state.recs.push(r);
    if (s > state.lastSeq) state.lastSeq = s;
  }
  state.recs.sort((a,b)=>((a.seq??0)-(b.seq??0)));
  for (const r of records || []){
    if (r.kind === 'task') state.task = r;
    if (r.error) state.errors.push(r.error);
  }
}

async function poll(){
  if (P.mode !== 'live') return;
  try{
    const u = `${P.base}/metrics?since=${state.lastSeq}`;
    const res = await fetch(u, {cache:'no-store'});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const j = await res.json();
    put(j.records); state.problems = j.problems || [];
    state.connected = true; state.note = '';
  }catch(e){
    state.connected = false;
    state.note = '连接断开，正在自动重试：' + e.message;
  }
  render();
}

function series(kind){ return state.recs.filter(r=>r.kind===kind); }
function field(recs, f, xf){
  const out = [];
  for (const r of recs){
    const y = r[f];
    if (y === undefined || y === null || !isFinite(y)) { out.push([xf(r), null]); continue; }
    out.push([xf(r), +y]);
  }
  return out;
}
function fmt(v, d){ if (v===null||v===undefined||!isFinite(v)) return '—';
  return (Math.abs(v)>=100? v.toFixed(d??0) : v.toFixed(d??3)); }
function dur(s){ if (s===null||s===undefined||!isFinite(s)) return '—';
  s = Math.max(0, Math.floor(s)); const h=Math.floor(s/3600), m=Math.floor(s%3600/60);
  return (h? h+'h ':'') + m + 'm ' + (s%60) + 's'; }

function stats(vals){
  const f = vals.filter(v=>v!==null && isFinite(v)).map(Number).sort((a,b)=>a-b);
  if (!f.length) return {n:0, missing:'无有效记录'};
  const q = p => f[Math.min(f.length-1, Math.round(p*(f.length-1)))];
  return {n:f.length, mean:f.reduce((a,b)=>a+b,0)/f.length,
          median:(f.length%2? f[(f.length-1)/2] : 0.5*(f[f.length/2-1]+f[f.length/2])),
          min:f[0], max:f[f.length-1], p95:q(0.95)};
}
function smooth(vals, w){
  if (w<=1) return vals.slice();
  const out=[]; let buf=[];
  for (const v of vals){ if (v===null||!isFinite(v)){ out.push(null); continue; }
    buf.push(v); if (buf.length>w) buf.shift(); out.push(buf.reduce((a,b)=>a+b,0)/buf.length); }
  return out;
}
function applyWindow(pts){
  if (state.win==='all' || !pts.length) return pts;
  const k = parseInt(state.win,10);
  return pts.slice(Math.max(0, pts.length-k));
}
function zoomed(pts, key){
  const z = state.zoom[key]; if (!z || !pts.length) return pts;
  return pts.filter(p=>p[0]>=z[0] && p[0]<=z[1]);
}

// ---- 画图 -----------------------------------------------------------------
function baseFrame(cv, spec){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const g = cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,w,h);
  const m = {l:52, r:8, t:8, b:22};
  return {g, w, h, m, iw: w-m.l-m.r, ih: h-m.t-m.b};
}
function drawAxes(f, spec, xs, ys, opts){
  const {g,m,iw,ih} = f;
  const freq = opts && opts.freqAxis;      // 直方图：整数频次轴
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (!isFinite(lo) || !isFinite(hi)) return null;
  if (freq){ lo = 0; }                      // 频次从 0 起，不外推负数
  else {
    if (spec.log && lo>0){ lo = Math.log10(lo); hi = Math.log10(hi); }
    if (hi-lo < 1e-9){ hi = lo + 1e-9; }
    const pad = (hi-lo)*0.08; lo -= pad; hi += pad;
  }
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const sx = x => m.l + (x1-x0<1e-9? iw/2 : (x-x0)/(x1-x0)*iw);
  const sy = y => { const v = spec.log && y>0? Math.log10(y) : y;
                    return m.t + ih - (v-lo)/(hi-lo)*ih; };
  g.strokeStyle='#e3e6ec'; g.lineWidth=1;
  for (let i=0;i<=4;i++){ const y=m.t+ih*i/4; g.beginPath(); g.moveTo(m.l,y); g.lineTo(m.l+iw,y); g.stroke(); }
  for (let i=0;i<=4;i++){ const x=m.l+iw*i/4; g.beginPath(); g.moveTo(x,m.t); g.lineTo(x,m.t+ih); g.stroke(); }
  g.fillStyle='#6b7280'; g.font='11px system-ui'; g.textAlign='right';
  for (let i=0;i<=4;i++){ const v = hi-(hi-lo)*i/4;
    let lab;
    if (freq) lab = String(Math.round(v));                       // 频次是计数，取整
    else if (spec.log) lab = (Math.pow(10,v)).toPrecision(3);
    else lab = (Math.abs(v)>=100? v.toFixed(0): v.toFixed(2));
    g.fillText(lab, m.l-6, m.t+ih*i/4+3); }
  g.textAlign='center';
  const xf = v => (Math.abs(v)>=1000? (v/1000).toFixed(1)+'k' : v.toFixed(0));
  for (let i=0;i<=4;i++){ const v = x0+(x1-x0)*i/4; g.fillText(xf(v), m.l+iw*i/4, m.t+ih+14); }
  return {sx, sy, x0, x1, lo, hi};
}
function drawSeries(f, axis, pts, color, width, dash){
  const {g} = f; let started=false;
  g.save(); g.strokeStyle=color; g.lineWidth=width; g.setLineDash(dash||[]);
  g.beginPath();
  for (const [x,y] of pts){
    if (y===null || !isFinite(y)){ started=false; continue; }
    const px=axis.sx(x), py=axis.sy(y);
    if (!started){ g.moveTo(px,py); started=true; } else g.lineTo(px,py);
  }
  g.stroke(); g.restore();
}
function drawMarker(f, axis, y, label, color){
  const {g,m,iw}=f; if (y===null||!isFinite(y)) return;
  const py=axis.sy(y); if (!isFinite(py)) return;
  g.save(); g.strokeStyle=color; g.lineWidth=1; g.setLineDash([4,3]);
  g.beginPath(); g.moveTo(m.l,py); g.lineTo(m.l+iw,py); g.stroke(); g.restore();
  g.fillStyle=color; g.font='11px system-ui'; g.textAlign='left';
  g.fillText(label, m.l+4, py-3);
}
function drawLineChart(card, spec, ptsAll, key){
  const cv = card.querySelector('canvas'); const f = baseFrame(cv, spec);
  const {g,m,iw,ih} = f; g.fillStyle='#6b7280'; g.font='12px system-ui';
  if (!ptsAll.length){ g.fillText(spec.unit, m.l, m.t+12); return; }
  const pts = zoomed(applyWindow(ptsAll), key);
  const ys = pts.map(p=>p[1]).filter(v=>v!==null && isFinite(v));
  const xs = pts.map(p=>p[0]);
  if (!ys.length){ emptyOverlay(card, '没有有限值可画（记录里该字段为空或非有限）'); return; }
  const axis = drawAxes(f, spec, xs, ys); if (!axis){ return; }
  const win = spec.hardware? state.smooth.hw : state.smooth.loss;
  if (spec.raw){ drawSeries(f, axis, pts, '#b8c2d6', 1); }        // 原始值始终画
  const sm = smooth(pts.map(p=>p[1]), win);
  drawSeries(f, axis, pts.map((p,i)=>[p[0], sm[i]]), spec.color, 2);
  const st = stats(ys);
  if (spec.marker === 'min'){
    const i = pts.findIndex(p=>p[1]===st.min);
    const best = i>=0? st.min : null;
    if (best!==null){ const xb=pts[i][0];
      g.fillStyle=spec.color; g.beginPath(); g.arc(axis.sx(xb), axis.sy(best), 3, 0, 6.283); g.fill();
      g.fillText(`历史最低 ${fmt(best)} @ ${spec.xlabel} ${fmt(xb,0)}`, axis.sx(xb)+6, axis.sy(best)-6); }
  }
  if (spec.marker === 'max'){
    let bi=-1, bv=-Infinity;
    pts.forEach((p,i)=>{ if (p[1]!==null && p[1]>bv){ bv=p[1]; bi=i; } });
    if (bi>=0){ g.fillStyle=spec.color;
      g.beginPath(); g.arc(axis.sx(pts[bi][0]), axis.sy(bv), 3, 0, 6.283); g.fill();
      g.fillText(`历史最高 ${fmt(bv,4)} @ ${spec.xlabel} ${fmt(pts[bi][0],0)}`,
                 axis.sx(pts[bi][0])+6, axis.sy(bv)-6); }
  }
  if (spec.marker === 'median'){ drawMarker(f, axis, st.median, `中位 ${fmt(st.median)}`, '#6b7280'); }
  const spikes = [];
  const med = st.median;
  pts.forEach(p=>{ if (p[1]!==null && med && p[1] > med*3) spikes.push(p); });
  for (const p of spikes.slice(0,6)){ g.fillStyle='#c2410c';
    g.beginPath(); g.arc(axis.sx(p[0]), axis.sy(p[1]), 2.5, 0, 6.283); g.fill(); }
  if (spec.hardware && state.win==='all'){
    g.fillStyle='#6b7280'; g.font='11px system-ui'; g.textAlign='right';
    g.fillText(`原始点 ${ptsAll.length}（绘图抽稀后 ${pts.length}）`, m.l+iw-2, m.t+11);
    g.textAlign='left';
  }
  card.dataset.stats = JSON.stringify(st);
}
function drawHist(card, spec, values){
  const cv = card.querySelector('canvas'); const f = baseFrame(cv, spec);
  const {g,m,iw,ih} = f;
  const vals = values.filter(v=>v!==null && isFinite(v));
  if (!vals.length){ emptyOverlay(card, '没有有限值可统计'); return; }
  const st = stats(vals);
  const lo0 = st.min, hi0 = st.max;
  // 裁剪规则：只影响绘图范围，统计量仍按原始值算，并在图注里写明
  const clipP = 0.995;
  const sorted = vals.slice().sort((a,b)=>a-b);
  const hiC = sorted[Math.min(sorted.length-1, Math.floor(clipP*(sorted.length-1)))];
  const clipped = Math.max(0, vals.length - vals.filter(v=>v<=hiC).length);
  const lo = lo0, hi = Math.max(hiC, lo0 + 1e-9);
  const bins = 40, w = (hi-lo)/bins, counts = new Array(bins).fill(0);
  for (const v of vals){ const i = Math.max(0, Math.min(bins-1, Math.floor((v-lo)/w))); counts[i]++; }
  const hiC2 = Math.max(...counts);
  const axis = drawAxes(f, spec, [lo, hi], [0, hiC2],
                        {freqAxis: true});             // 真轴 + 整数频次刻度
  if (!axis) return;
  g.fillStyle = spec.color; g.globalAlpha = 0.75;
  for (let i=0;i<bins;i++){ const x0=axis.sx(lo+i*w), x1=axis.sx(lo+(i+1)*w);
    const y0=axis.sy(counts[i]); g.fillRect(x0+0.5, y0, Math.max(1,x1-x0-1), m.t+ih-y0); }
  g.globalAlpha = 1;
  // 标注放在底部：与顶部刻度标签错开，避免互相覆盖
  // 只画虚线，文字留给下方统计行（画布底部与刻度/裁剪说明重叠过）
  const vline = (v, col) => { const x=axis.sx(v); g.save();
    g.strokeStyle=col; g.setLineDash([4,3]); g.beginPath();
    g.moveTo(x,m.t); g.lineTo(x,m.t+ih); g.stroke(); g.restore(); };
  vline(st.mean, '#2f6df6');
  vline(st.median, '#2e9e6b');
  if (spec.key === 'loss_hist') vline(st.min, '#8a6d1f');
  card.dataset.marks = JSON.stringify({mean: st.mean, median: st.median,
                                       min: st.min, max: st.max});
  g.fillStyle='#6b7280'; g.font='11px system-ui';
  if (clipped) g.fillText(`裁剪：绘图范围到 p99.5=${fmt(hiC)}（${clipped} 个更大值仍计入统计）`, m.l+2, m.t+11);
  card.dataset.stats = JSON.stringify(st);
}
function emptyOverlay(card, text){
  let el = card.querySelector('.empty');
  if (!el){ el = document.createElement('div'); el.className='empty'; card.querySelector('.cv').appendChild(el); }
  el.textContent = text;
}
function clearOverlay(card){ const el=card.querySelector('.empty'); if (el) el.remove(); }

// ---- 卡片表 ---------------------------------------------------------------
const SPECS = {
  loss:      {unit:'loss', color:'#2f6df6', raw:true, marker:'min', get:()=>field(series('train'),'loss',r=>r.step||0)},
  acc:       {unit:'%', color:'#2e9e6b', raw:true, marker:'max', get:()=>field(series('train'),'acc',r=>r.step||0).map(p=>[p[0], p[1]===null?null:p[1]*100])},
  grad:      {unit:'L2', log:true, color:'#c2410c', raw:true, marker:'median', get:()=>field(series('train'),'grad_norm',r=>r.step||0)},
  lr:        {unit:'lr', log:true, color:'#7c3aed', raw:false, marker:null, get:()=>field(series('train'),'lr',r=>r.step||0)},
  loss_hist: {unit:'帧数', color:'#2f6df6', hist:true, get:()=>series('train').map(r=>r.loss)},
  grad_hist: {unit:'帧数', color:'#c2410c', hist:true, get:()=>series('train').map(r=>r.grad_norm)},
  gpu_mem:   {unit:'GiB', hardware:true, color:'#0f766e', markMax:true, get:()=>sysSeries('gpu_mem_gib')},
  speed:     {unit:'s/it', color:'#db2777', raw:true, marker:null, get:()=>field(series('train'),'step_s',r=>r.step||0)},
  gpu_power: {unit:'W', hardware:true, color:'#b45309', markMax:true, get:()=>sysSeries('gpu_power_w')},
  gpu_util:  {unit:'%', hardware:true, color:'#15803d', markMax:true, get:()=>sysSeries('gpu_util_pct')},
  cpu_util:  {unit:'%', hardware:true, color:'#0369a1', markMax:true, get:()=>sysSeries('cpu_util_pct')},
  sys_mem:   {unit:'GiB', hardware:true, color:'#9333ea', markMax:true, get:()=>sysSeries('sys_mem_gib')},
};
function sysSeries(field){
  const t0 = state.task && state.task.started_at? state.task.started_at :
             (series('system')[0]? series('system')[0].t : 0);
  const out = [];
  for (const r of series('system')){
    let v = r[field];
    if (state.device !== null && Array.isArray(r.devices)){
      const d = r.devices.find(x=>x.device===state.device);
      v = d? (field==='gpu_mem_gib'? d.mem_gib :
              field==='gpu_power_w'? d.power_w : d.util_pct) : null;
    }
    out.push([r.t - t0, (v===undefined||v===null||!isFinite(v))? null : +v]);
  }
  return out;
}
function unavailableReason(key){
  const map = {gpu_power:'gpu_power_w', gpu_util:'gpu_util_pct', gpu_mem:'gpu_mem_gib',
               cpu_util:'cpu_util_pct', sys_mem:'sys_mem_gib'};
  const f = map[key]; if (!f) return '';
  for (const r of series('system')){
    const u = r.unavailable || {};
    for (const k in u){ if (k.endsWith(f) || k===f) return u[k]; }
  }
  return '';
}

function statusbar(){
  const t = state.task || {};
  const tr = series('train');
  const last = tr.length? tr[tr.length-1] : {};
  const steps = tr.map(r=>r.step_s).filter(v=>isFinite(v));
  const meanStep = steps.length? steps.reduce((a,b)=>a+b,0)/steps.length : null;
  const elapsed = t.started_at? (Date.now()/1000 - t.started_at) : null;
  const cls = t.status || 'waiting';
  const names = {waiting:'等待中', running:'运行中', paused:'已暂停',
                 completed:'已完成', failed:'已失败'};
  const rows = [
    ['任务名称', t.name || P.title],
    ['运行状态', `<span class="pill ${cls}">${names[cls]||cls}${state.connected?'':'（离线）'}</span>`],
    ['step', `${last.step ?? t.current_step ?? '—'} / ${t.total_steps ?? '—'}`],
    ['epoch', `${last.epoch ?? t.epoch ?? '—'}`],
    ['已运行时间', dur(elapsed)],
    ['当前 loss', fmt(last.loss)],
    ['平均每步耗时', meanStep===null? '—' : meanStep.toFixed(4)+' s/it'],
    ['当前时间', new Date().toLocaleTimeString()],
  ];
  document.getElementById('statusbar').innerHTML = rows.map(
    ([k,v])=>`<span class="item"><span class="k">${k}</span><span class="v">${v}</span></span>`).join('');
}

function banners(){
  const box = document.getElementById('banners'); const html = [];
  if (P.initial.length && P.initial[0].demo || series('train').some(r=>r.demo))
    html.push('<div class="banner demo"><b>DEMO 演示数据</b>：本页展示的是演示模式生成的合成记录，不是真实训练结果。</div>');
  if ((state.task||{}).status === 'failed')
    html.push(`<div class="banner err"><b>训练失败</b>：${(state.task.error||'原因见日志')}（图表保留失败前的记录）</div>`);
  if (state.problems && state.problems.length)
    html.push(`<div class="banner err">记录文件有 ${state.problems.length} 行不可读（半写行/损坏）：${state.problems[0]}</div>`);
  if (!state.connected && P.mode === 'live')
    html.push(`<div class="banner note">${state.note || '正在连接…'}</div>`);
  const reason = unavailableReason('gpu_power');
  if (reason) html.push(`<div class="banner note">GPU 功耗：该设备未提供功耗数据（${reason}）</div>`);
  if (state.task && state.task.aggregate)
    html.push(`<div class="banner note">多设备汇总口径：${state.task.aggregate}</div>`);
  box.innerHTML = html.join('');
}

function buildGrid(){
  const grid = document.getElementById('grid');
  grid.innerHTML = P.charts.map(c=>`
    <div class="card" id="card-${c.key}">
      <h3>${c.title}</h3>
      <div class="sub">${c.unit? '单位：'+c.unit+'　':''}横轴：${c.xlabel}${c.log? '　纵轴：对数':''}</div>
      <div class="cv"><canvas></canvas></div>
      <div class="stats"></div>
    </div>`).join('');
  grid.querySelectorAll('canvas').forEach((cv,i)=>{
    const key = P.charts[i].key;
    cv.addEventListener('mousemove', ev=>hover(ev, key));
    cv.addEventListener('mouseleave', ()=>{ document.getElementById('tip').style.display='none'; });
    cv.addEventListener('wheel', ev=>{
      ev.preventDefault();
      const spec = SPECS[key]; const ptsAll = applyWindow(spec.get());
      if (!ptsAll.length) return;
      const cur = state.zoom[key] || [Math.min(...ptsAll.map(p=>p[0])), Math.max(...ptsAll.map(p=>p[0]))];
      const span = (cur[1]-cur[0]) || 1; const k = ev.deltaY>0? 1.15 : 1/1.15;
      const mid = (cur[0]+cur[1])/2;
      state.zoom[key] = [mid-span*k/2, mid+span*k/2];
      render();
    }, {passive:false});
    cv.addEventListener('dblclick', ()=>{ delete state.zoom[key]; render(); });
  });
}

function hover(ev, key){
  const spec = SPECS[key]; const card = document.getElementById('card-'+key);
  const cv = card.querySelector('canvas'); const rect = cv.getBoundingClientRect();
  const pts = zoomed(applyWindow(spec.get()), key).filter(p=>p[1]!==null && isFinite(p[1]));
  if (!pts.length) return;
  const m = {l:52, r:8}; const iw = rect.width-m.l-m.r;
  const x0 = Math.min(...pts.map(p=>p[0])), x1 = Math.max(...pts.map(p=>p[0]));
  const frac = Math.max(0, Math.min(1, (ev.clientX-rect.left-m.l)/iw));
  const xv = x0 + frac*(x1-x0);
  let best = pts[0];
  for (const p of pts) if (Math.abs(p[0]-xv) < Math.abs(best[0]-xv)) best = p;
  const tip = document.getElementById('tip');
  const t0 = state.task && state.task.started_at? state.task.started_at : 0;
  const when = spec.hardware && t0? new Date((t0+best[0])*1000).toLocaleTimeString() : ('step '+best[0]);
  tip.innerHTML = `${P.charts.find(c=>c.key===key).title}<br>${spec.xlabel}: ${fmt(best[0],0)}<br>` +
                  (spec.hardware? `时间: ${when}<br>`:'') + `值: ${fmt(best[1],4)} ${spec.unit||''}`;
  tip.style.display='block'; tip.style.left=(ev.clientX+12)+'px'; tip.style.top=(ev.clientY+12)+'px';
}

function deviceSelector(){
  const devs = new Set();
  for (const r of series('system')) for (const d of (r.devices||[])) devs.add(d.device);
  const wrap = document.getElementById('devWrap'); const sel = document.getElementById('devSel');
  if (devs.size <= 1){ wrap.style.display='none'; return; }
  wrap.style.display='flex';
  sel.innerHTML = `<option value="">全部（汇总）</option>` +
    [...devs].sort().map(d=>`<option value="${d}">GPU ${d}</option>`).join('');
  sel.onchange = ()=>{ state.device = sel.value===''? null : parseInt(sel.value,10); render(); };
}

function render(){
  statusbar(); banners();
  for (const c of P.charts){
    const spec = SPECS[c.key]; const card = document.getElementById('card-'+c.key);
    const merged = Object.assign({}, c, spec);
    clearOverlay(card);
    delete card.dataset.stats;
    // 一张图出错不能拖垮整页（实测：drawHist 抛一次异常，后面所有硬件图都
    // 停在上一次的"未采集"文字上，看起来像没有数据）
    try {
      if (spec.hist){
        const vals = spec.get();
        if (!vals.some(v=>v!==null && isFinite(v))) emptyOverlay(card, '没有有限值可统计');
        else drawHist(card, merged, vals);
      } else {
        const pts = spec.get();
        const reason = spec.hardware? unavailableReason(c.key) : '';
        if (!pts.length){
          emptyOverlay(card, spec.hardware? (reason||'未采集到硬件数据') : '暂无该指标的记录');
        } else if (pts.every(p=>p[1]===null)){
          emptyOverlay(card, reason || '记录里该字段全为空（未采集或不适用）');
        } else drawLineChart(card, merged, pts, c.key);
      }
    } catch (e) {
      emptyOverlay(card, '该图渲染失败：' + (e && e.message ? e.message : e));
    }
    const st = card.dataset.stats? JSON.parse(card.dataset.stats) : null;
    card.querySelector('.stats').textContent = st?
      (st.missing? st.missing :
       (spec.hist?
        `均值 ${fmt(st.mean)} · 中位 ${fmt(st.median)} · 最小 ${fmt(st.min)} · 最大 ${fmt(st.max)} · n=${st.n}（蓝=均值 绿=中位 虚线）`
        : `当前 ${fmt(pts_last(c.key),4)??'—'} · 均值 ${fmt(st.mean)} · 峰值 ${fmt(st.max)} · 中位 ${fmt(st.median)} · n=${st.n}${st.dropped_non_finite? '（忽略非有限 '+st.dropped_non_finite+'）':''}`))
      : '无统计（无有效记录）';
  }
}
function pts_last(key){
  const spec = SPECS[key];
  const pts = spec.hist? [] : spec.get();
  const finite = pts.filter(p=>p[1]!==null && isFinite(p[1]));
  return finite.length? finite[finite.length-1][1] : null;
}

document.querySelectorAll('.toolbar button[data-win]').forEach(b=>{
  b.onclick = ()=>{ state.win = b.dataset.win;
    document.querySelectorAll('.toolbar button[data-win]').forEach(x=>x.classList.toggle('on', x===b));
    render(); };
});
document.getElementById('smoothLoss').oninput = e=>{ state.smooth.loss = Math.max(1, parseInt(e.target.value||'1',10)); render(); };
document.getElementById('smoothHw').oninput = e=>{ state.smooth.hw = Math.max(1, parseInt(e.target.value||'1',10)); render(); };
window.addEventListener('resize', ()=>render());

put(P.initial);
buildGrid(); deviceSelector(); render();
if (P.mode === 'live'){ setInterval(poll, P.poll_ms); poll(); }
else { setInterval(()=>{ statusbar(); }, 1000); }
</script>
</body></html>
"""
