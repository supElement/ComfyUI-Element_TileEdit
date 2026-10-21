// tile_split_ui.js — ElementTileSplit 交互面板
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js"; 
import { eeMarkForward } from "./ee_canvas_utils.js";
import { eeRunIsolatedPromptCore, eeAddWindowListener, isNodes2_0, hideWidgetAndSlot } from "./ee_node_utils.js";

const STYLE_ID = "ets-style";
const MIN_DIVS = 2;
/* ================================================================
   Gap 规划 v4（与后端逐位镜像）
   ================================================================ */
const GAP_MAX_ASPECT = 3.0;
const GAP_MIN_SHORT_DIVS = 2;
const GAP_COVER_STEPS = [0, 8, 16, 25];
const GAP_TARGET_TILES = 8;
const GAP_SEARCH_BUDGET = 1200;

const PREVIEW_MAX_EDGE = 2048; 
const HIT = 8; // 显示空间命中容差（crop 式）
const ETS_UPSTREAM = new Set();
const PALETTE = [
  "#ff4d4d", "#ff9a3d", "#ffd93d", "#b8e63d", "#4ade80", "#2dd4bf", "#38bdf8", "#5b8cff",
  "#8b5cf6", "#d946ef", "#f43f8e", "#c2410c", "#166534", "#0e7490", "#1d4ed8", "#6d28d9",
  "#a16207", "#be123c", "#334155", "#7c2d12",
];
const CUR = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", ne: "nesw-resize", sw: "nesw-resize", nw: "nwse-resize", se: "nwse-resize" };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const snap = (v, d) => Math.round(v / d) * d;
const snapCeil = (v, d) => Math.max(d, Math.ceil(v / d - 1e-6) * d);
const roundDiv = (v, d) => Math.max(d, Math.round(v / d) * d);
function djb2(s) { let h = 5381; for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0; return (h >>> 0).toString(36); }
function hexA(hex, a) {
  const h = String(hex).replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16) || 0;
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function defaultState() {
  return {
    version: 1, mode: "manual", div: 16,
    unified: { upscale: 2.0, columns: 2, rows: 2, tile_size: 1024 },
    overlap_mode: "px", overlap_px: 64, overlap_pct: 15,
    fill_gaps: true, gap_max_px: 1536,
    global_prompt: "enhance quality preserving all the details", 
    global_append: true, 
    global_negative: "",
	enable_layer_upscale: false,
    source: null, src_w: 0, src_h: 0,
    layers: [], selected: null, prompt_tab: "global",
  };
}

/* ---------- 与后端逐位一致的几何镜像 ---------- */
function overlapPx(mode, pxVal, pctVal, edge) {
  if (mode === "pct") return Math.round(clamp(Number(pctVal) || 0, 0, 45) / 100 * edge);
  return clamp(Math.round(Number(pxVal) || 0), 0, Math.floor(edge / 2));
}
function splitRegion(x, y, w, h, cols, rows, ov) {
  cols = Math.max(1, cols | 0); rows = Math.max(1, rows | 0); ov = Math.max(0, ov | 0);
  if (cols === 1 && rows === 1) return [{ x, y, w, h, col: 0, row: 0 }];
  const axis = (total, n, o) => {
    if (n === 1) return [[0, total]];
    const cw = Math.max(1, Math.floor((total + o * (n - 1)) / n + 0.5));
    const step = (total - cw) / (n - 1);
    const out = [];
    for (let i = 0; i < n; i++) {
      const a = Math.floor(i * step + 0.5);
      const end = i < n - 1 ? a + cw : total;
      out.push([a, end - a]);
    }
    return out;
  };
  const xs = axis(w, cols, ov), ys = axis(h, rows, ov);
  const out = [];
  ys.forEach(([ty, th], j) => xs.forEach(([tx, tw], i) =>
    out.push({ x: x + tx, y: y + ty, w: tw, h: th, col: i, row: j })));
  return out;
}
function largestFreeRect(free) {
  const rows = free.length, cols = free[0].length;
  const heights = new Array(cols).fill(0);
  let widest = null;   
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) heights[c] = free[r][c] ? heights[c] + 1 : 0;
    const stack = [];
    for (let c = 0; c <= cols; c++) {
      const hcur = c < cols ? heights[c] : 0;
      let start = c;
      while (stack.length && stack[stack.length - 1][1] >= hcur) {
        const [s, hgt] = stack.pop();
        const area = hgt * (c - s);
        if (area > 0) {
          if (!widest || area > widest[0]) widest = [area, r - hgt + 1, s, hgt, c - s];
        }
        start = s;
      }
      stack.push([start, hcur]);
    }
  }
  return widest && widest[0] > 0 ? widest : null;
}

function snapV(v, s) { return Math.round(v / s) * s; }

function balancedSubsplit(rects, gapMax) {
  const out = [];
  for (const { x: rx, y: ry, w: rw, h: rh } of rects) {
    if (rw <= gapMax && rh <= gapMax) { out.push({ x: rx, y: ry, w: rw, h: rh }); continue; }
    const maxSc = Math.max(1, Math.ceil(rw / gapMax)), maxSr = Math.max(1, Math.ceil(rh / gapMax));
    let best = null;
    for (let sc = 1; sc <= maxSc; sc++) {
      const sw = Math.ceil(rw / sc); if (sw > gapMax) continue;
      for (let sr = 1; sr <= maxSr; sr++) {
        const sh = Math.ceil(rh / sr); if (sh > gapMax) continue;
        const a = Math.max(sw, sh) / Math.max(1, Math.min(sw, sh));
        if (!best || a < best[0]) best = [a, sc, sr];
      }
    }
    const [, sc, sr] = best || [0, maxSc, maxSr];
    const sw = Math.ceil(rw / sc), sh = Math.ceil(rh / sr);
    for (let i = 0; i < sc; i++) for (let j = 0; j < sr; j++) {
      const x = rx + i * sw, y = ry + j * sh;
      out.push({ x, y, w: Math.min(sw, rx + rw - x), h: Math.min(sh, ry + rh - y) });
    }
  }
  return out;
}

function mergeAdjacent(rects, gapMax, minShort, aspect = GAP_MAX_ASPECT) {
  const rs = rects.map(r => r.slice());
  let changed = true;
  while (changed) {
    changed = false;
    outer:
    for (let i = 0; i < rs.length; i++) {
      for (let j = i + 1; j < rs.length; j++) {
        const a = rs[i], b = rs[j];
        let m = null;
        if (a[1] === b[1] && a[3] === b[3] && (a[0] + a[2] === b[0] || b[0] + b[2] === a[0]))
          m = [Math.min(a[0], b[0]), a[1], a[2] + b[2], a[3]];
        else if (a[0] === b[0] && a[2] === b[2] && (a[1] + a[3] === b[1] || b[1] + b[3] === a[1]))
          m = [a[0], Math.min(a[1], b[1]), a[2], a[3] + b[3]];
        if (!m) continue;
        if (m[2] > gapMax || m[3] > gapMax ||
            Math.max(m[2], m[3]) / Math.max(1, Math.min(m[2], m[3])) > aspect ||
            Math.min(m[2], m[3]) < minShort) continue;
        rs[i] = m; rs.splice(j, 1); changed = true;
        break outer;
      }
    }
  }
  return rs;
}

function rectOverlap(a, b) {
  const w = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  return w * h;
}

function rectSubtract(a, b) {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  const ix1 = Math.max(ax, bx), iy1 = Math.max(ay, by);
  const ix2 = Math.min(ax + aw, bx + bw), iy2 = Math.min(ay + ah, by + bh);
  if (ix1 >= ix2 || iy1 >= iy2) return [a];
  if (ix1 <= ax && iy1 <= ay && ix2 >= ax + aw && iy2 >= ay + ah) return [];
  const out = [];
  if (iy1 > ay) out.push([ax, ay, aw, iy1 - ay]);
  if (iy2 < ay + ah) out.push([ax, iy2, aw, ay + ah - iy2]);
  if (ix1 > ax) out.push([ax, iy1, ix1 - ax, iy2 - iy1]);
  if (ix2 < ax + aw) out.push([ix2, iy1, ax + aw - ix2, iy2 - iy1]);
  return out;
}

function edgeFullyCovered(B, R) {
  const [bx, by, bw, bh] = B;
  const [rx, ry, rw, rh] = R;
  const ix1 = Math.max(bx, rx), iy1 = Math.max(by, ry);
  const ix2 = Math.min(bx + bw, rx + rw), iy2 = Math.min(by + bh, ry + rh);
  if (ix1 >= ix2 || iy1 >= iy2) return false;

  if (rx <= bx && rx + rw >= bx + bw && ry <= by && ry + rh > by) return true;
  if (rx <= bx && rx + rw >= bx + bw && ry < by + bh && ry + rh >= by + bh) return true;
  if (ry <= by && ry + rh >= by + bh && rx <= bx && rx + rw > bx) return true;
  if (ry <= by && ry + rh >= by + bh && rx < bx + bw && rx + rw >= bx + bw) return true;
  return false;
}

function planGaps(covered, div, outW, outH, gapMax, regions = [], targetTiles = GAP_TARGET_TILES) {
  if (!regions.length) return balancedSubsplit([{ x: 0, y: 0, w: outW, h: outH }], gapMax);

  const baseGrid = balancedSubsplit([{ x: 0, y: 0, w: outW, h: outH }], gapMax);

  let pieces = [];
  for (const b of baseGrid) {
    let parts = [[b.x, b.y, b.w, b.h]];
    for (const rg of regions) {
      const rr = [rg.x, rg.y, rg.w, rg.h];
      const nxt = [];
      for (const p of parts) {
        if (edgeFullyCovered(p, rr)) {
          nxt.push(...rectSubtract(p, rr));
        } else {
          nxt.push(p);
        }
      }
      parts = nxt;
      if (!parts.length) break;
    }
    pieces.push(...parts);
  }

  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < pieces.length; i++) {
      if (!pieces[i]) continue;
      let [x, y, w, h] = pieces[i];
      if (Math.max(w, h) / Math.max(1, Math.min(w, h)) <= 5.0) continue;

      const ox = x, oy = y, ow = w, oh = h;
      if (w > h) {
        const diff = w / 5.0 - h;
        let expandT = Math.min(y, diff / 2);
        let expandB = diff - expandT;
        if (expandB > outH - (y + h)) {
          expandB = outH - (y + h);
          expandT = diff - expandB;
        }
        y = Math.max(0, Math.floor(y - expandT));
        h = Math.min(outH - y, Math.ceil(h + diff));
      } else {
        const diff = h / 5.0 - w;
        let expandL = Math.min(x, diff / 2);
        let expandR = diff - expandL;
        if (expandR > outW - (x + w)) {
          expandR = outW - (x + w);
          expandL = diff - expandR;
        }
        x = Math.max(0, Math.floor(x - expandL));
        w = Math.min(outW - x, Math.ceil(w + diff));
      }

      let nx = x, ny = y, nw = w, nh = h;
      for (let j = 0; j < pieces.length; j++) {
        if (i === j || !pieces[j]) continue;
        const [qx, qy, qw, qh] = pieces[j];
        const ix = Math.max(nx, qx), iy = Math.max(ny, qy);
        const ix2 = Math.min(nx + nw, qx + qw), iy2 = Math.min(ny + nh, qy + qh);
        if (ix < ix2 && iy < iy2) {
          const nx2 = Math.min(nx, qx), ny2 = Math.min(ny, qy);
          nx = nx2; ny = ny2;
          nw = Math.max(nx + nw, qx + qw) - nx;
          nh = Math.max(ny + nh, qy + qh) - ny;
          pieces[j] = null; 
        }
      }
      
      if (nx !== ox || ny !== oy || nw !== ow || nh !== oh) {
        pieces[i] = [nx, ny, nw, nh];
        changed = true;
      }
    }
    if (changed) pieces = pieces.filter(p => p !== null);
  }

  let finalPieces = [];
  for (const p of pieces) {
    if (p[2] > gapMax || p[3] > gapMax) {
      balancedSubsplit([{ x: p[0], y: p[1], w: p[2], h: p[3] }], gapMax)
        .forEach(sub => finalPieces.push([sub.x, sub.y, sub.w, sub.h]));
    } else {
      finalPieces.push(p);
    }
  }

  const cleanPieces = [];
  for (const p of finalPieces) {
    let parts = [p];
    for (const c of cleanPieces) {
      const nxt = [];
      for (const pt of parts) {
        nxt.push(...rectSubtract(pt, c));
      }
      parts = nxt;
      if (!parts.length) break;
    }
    cleanPieces.push(...parts);
  }

  return cleanPieces.map(([x, y, w, h]) => ({ x, y, w, h }));
}


/* ---------- 样式 ---------- */
function installStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const st = document.createElement("style");
  st.id = STYLE_ID;
  st.textContent = `
.ets{display:flex;flex-direction:column;width:100%;height:100%;min-height:320px;background:#11141b;color:#e9eef8;border-radius:9px;overflow:hidden;font:12px/1.4 Inter,Segoe UI,Arial,sans-serif;user-select:none;position:relative}
.ets *{box-sizing:border-box}.ets button{font:inherit}
.ets-head{display:flex;align-items:center;gap:6px;padding:6px 8px;background:#1c222d;border-bottom:1px solid #30384a;flex-wrap:wrap;flex-shrink:0}
.ets-run{width:28px;height:24px;border:1px solid #2f7d55;background:#17301f;color:#7cf5b0;border-radius:5px;cursor:pointer;flex-shrink:0}
.ets-run:hover{background:#1f4a2e}
.ets-load{width:32px;border-color:#3b4558;background:#252d3a;color:#dce5f5}
.ets-load:hover{background:#2f3a4c}
.ets-tabbtn{padding:3px 10px;border:1px solid #3b4558;background:#252d3a;color:#dce5f5;border-radius:5px;cursor:pointer}
.ets-tabbtn.on{background:#2a4b7a;border-color:#4e7fc0}
.ets-field{display:inline-flex;align-items:center;gap:3px;color:#919bad;white-space:nowrap}
.ets-field input[type=number]{width:44px;height:22px;padding:1px 4px;background:#0d1118;color:#e9eef8;border:1px solid #333d50;border-radius:4px;outline:none}
.ets-field input#ets-gmax, .ets-field input#ets-ovpx, .ets-field input#ets-ovpct, .ets-field input#ets-utile{width:56px} /* 扩宽特定输入框防止数字显示不全 */
.ets-field input.ovon{border-color:#43d9d1}

.ets-ov-toggle {
  padding: 0 6px;
  height: 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #3b4558;
  background: #252d3a;
  color: #919bad;
  border-radius: 4px;
  cursor: pointer;
  user-select: none;
  font-size: 11px;
  margin-right: 4px; /* 与输入框拉开一点距离 */
  transition: all 0.2s;
}
.ets-ov-toggle:hover { 
  background: #2f3a4c; 
  color: #dce5f5; 
}
.ets-ov-toggle.ovon { 
  background: #2a4b7a; 
  border-color: #4e7fc0; 
  color: #fff; 
}
.ets-field input[type=checkbox]{accent-color:#43d9d1;width:14px;height:14px}
.ets-status{margin-left:auto;color:#919bad;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}

.ets .only-count,.ets .only-size,.ets .only-manual{display:none}
.ets.m-count .only-count{display:inline-flex; gap: 12px; align-items: center;}
.ets.m-size .only-size{display:inline-flex}
.ets.m-manual .only-manual{display:inline-flex; gap: 12px; align-items: center; margin-left: 12px;}

.ets-body{display:flex;min-height:0;flex:1}
.ets-canvaswrap{flex:1;min-width:0;display:flex}
.ets-stage{flex:1;display:flex;align-items:center;justify-content:center;position:relative;background:#0d1118;overflow:hidden}
.ets-wrap{position:relative;flex-shrink:0}
.ets-img{display:block;width:100%;height:100%}
.ets-cv{position:absolute;inset:0;width:100%;height:100%;touch-action:none}
.ets-placeholder{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#55627a;text-align:center;padding:0 16px;pointer-events:none;white-space:pre-line}
.ets-side{width:248px;flex-shrink:0;border-left:1px solid #30384a;display:flex;flex-direction:column;background:#141922;min-height:0}
.ets-sidehead{padding:5px 8px;color:#919bad;border-bottom:1px solid #262d3a;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;flex-shrink:0;gap:4px}
.ets-count{font-size:11px;text-align:right;flex:1;min-width:100px;line-height:1.2;word-break:break-all;color:#919bad;}
.ets-list{flex:1;overflow-y:auto;overflow-x:hidden;padding:4px;position:relative;scrollbar-color:#49556a #171c25}
.ets-row{display:flex;align-items:center;gap:6px;padding:4px;border:1px solid #262d3a;border-radius:6px;margin-bottom:4px;background:#191e28;cursor:grab}
.ets-row.sel{border-color:#43d9d1;box-shadow:0 0 0 1px #43d9d155}
.ets-thumb{flex-shrink:0}.ets-thumb canvas{display:block;width:44px;height:33px;border-radius:3px;background:#000}
.ets-badge{width:18px;height:18px;border-radius:4px;display:inline-flex;align-items:center;justify-content:center;color:#111;font-weight:700;flex-shrink:0}
.ets-rowmain{flex:1;min-width:0;display:flex;flex-direction:column;gap:2px}
.ets-rowparams{display:flex;gap:4px;flex-wrap:nowrap;align-items:center;color:#919bad}
.ets-rowparams input{width:32px;height:20px;padding:0 2px;background:#0d1118;color:#e9eef8;border:1px solid #333d50;border-radius:4px;outline:none}
.ets-rowparams input[data-f="upscale"]{width:52px} /* 单独加宽 upscale 输入框 */
.ets-rowparams label{display:inline-flex; align-items:center; gap:4px;}
.ets-del{border:none;background:transparent;color:#ef6a77;cursor:pointer;font-size:13px;padding:2px 4px;flex-shrink:0}
.ets-del:hover{color:#ff9aa5}
.ets-row.gaprow{cursor:default;opacity:.85}
.ets-insert{position:absolute;left:4px;right:4px;height:0;border-top:2px solid #43d9d1;box-shadow:0 0 6px #43d9d1;pointer-events:none;z-index:5}
.ets-prompts{border-top:1px solid #30384a;background:#141922;padding:4px 8px 8px;flex-shrink:0}
.ets-ptabs{display:flex;gap:4px;margin-bottom:4px;align-items:center}
.ets-ptabs .ets-hint{color:#718096;margin-left:auto}
.ets textarea{width:100%;resize:none;background:#0d1118;color:#e9eef8;border:1px solid #333d50;border-radius:4px;padding:3px 6px;font:inherit;outline:none}
.ets textarea.ets-p{height:72px;line-height:1.4}.ets textarea.ets-n{height:48px;margin-top:4px;line-height:1.4}
.ets textarea:disabled{opacity:.45}
.ets.inactive { opacity: 0.75; }
.ets input, .ets textarea { user-select: auto; }
body.ets-reordering,body.ets-reordering *{cursor:grabbing !important}
`;
  document.head.appendChild(st);
}

/* ---------- stateWidget 绑定（onNodeCreated 与 onConfigure 共用）---------- */
function bindStateWidget(node, sw) {
  if (!sw) return false;
  sw.serializeValue = () => sw.value;
  const origCb = sw.callback;
  sw.callback = function (v) {
    const rv = origCb?.call(this, v);
    return rv;
  };
  if (node.__ets && node.__ets.stateWidget !== sw) {
    node.__ets.stateWidget = sw;
    node.__ets._suppressLoadOnce = false;
    node.__ets.loadFromState();
    return true;
  }
  return false;
}

/* ---------- 主 UI 类 ---------- */
class TileSplitUI {
  constructor(node, root, stateWidget) {
    this.node = node; this.root = root; this.stateWidget = stateWidget;
    this.state = Object.assign(defaultState(), this.readState());
    this.state.unified = Object.assign(defaultState().unified, this.state.unified || {});
    if (!Array.isArray(this.state.layers)) this.state.layers = [];
    this.srcW = this.state.src_w || 0;
    this.srcH = this.state.src_h || 0;
    this.dispScale = 1; this.drag = null; this.previewNewRect = null;
    this.pending = null; this.reorder = null;
    this._unsubs = []; this._suppressLoadOnce = true; this._statusTimer = 0;
    this._imgKind = null; 
	this._previewRef = null;
	this._loadedMeta = null; 
    this.build(); this.attachEvents(); 
	this.bindImageSource(); this.render();
  }

  /* ===== 持久化（工作流指纹门禁）===== */
  workflowFp() {
    try {
      const g = app.graph;
      const name = g?.extra?.workflow_name || app.workflowManager?.activeWorkflow?.name || "";
      const nodes = (g?.nodes || g?._nodes || []);
      const parts = nodes.map(n => `${n.id}:${n.type}`).sort();
      const links = [];
      const lk = g?.links || {};
      for (const k in lk) {
        const l = lk[k];
        if (l && l.origin_id != null) links.push(`${l.origin_id}>${l.target_id}`);
      }
      links.sort();
      return djb2(name + "#" + parts.join(",") + "#" + links.join(","));
    } catch (_) { return "unknown"; }
  }
  lsKey() { return "ets_state_" + this.node.id; }
  readLocal() {
    try {
      const d = JSON.parse(localStorage.getItem(this.lsKey()) || "null");
      if (!d || d.__v !== 2) return null;                          
      if (String(d.fp) !== String(this.workflowFp())) return null; 
      return d.state || null;
    } catch (_) { return null; }
  }
  readState() {
    const raw = this.stateWidget?.value;
    try { return JSON.parse(typeof raw === "string" ? raw : "{}") || {}; }
    catch (_) { return {}; }
  }
  saveState() {
    if (!this.stateWidget) return;
    this._suppressLoadOnce = true;
    
    const ts = Date.now(); 
    const newVal = JSON.stringify(this.state);
    
    // 只有内容真正改变时才更新 widget 并触发画布变更
    if (this.stateWidget.value !== newVal) {
        this.stateWidget.value = newVal;
        this.stateWidget.callback?.(this.stateWidget.value);
        try { app.graph?.change?.(); } catch (_) {}
        this.node.setDirtyCanvas?.(true, true);
    }
    
    try {
      localStorage.setItem(this.lsKey(), JSON.stringify(
        { __v: 2, fp: this.workflowFp(), ts: ts, state: this.state }));
    } catch (_) {}
  }
  loadFromState() {
    if (this._suppressLoadOnce) { this._suppressLoadOnce = false; return; }
    let st = this.readState();
    const empty = !st || (!st.source && !(st.layers || []).length && !st.src_w);
    if (empty && !this.node.__etsFresh) {
      const cached = this.readLocal();
      if (cached && (cached.source || (cached.layers || []).length || cached.src_w)) st = cached;
    }
    this.state = Object.assign(defaultState(), st);
    this.state.unified = Object.assign(defaultState().unified, st.unified || {});
    this.state.layers = (st.layers || []).map((l, i) => Object.assign({
      id: i + 1, color: PALETTE[i % PALETTE.length], x: 0, y: 0, w: 0, h: 0,
      upscale: this.state.unified.upscale, cols: 1, rows: 1, prompt: "", negative: "",
    }, l));
    this.srcW = this.state.src_w || 0;
    this.srcH = this.state.src_h || 0;
    console.debug("[TileSplit] restore:", this.node.id, "src=", this.srcW, "x", this.srcH,
      "layers=", this.state.layers.length, "mode=", this.state.mode, "ref=", !!this.state.preview_ref);
    this.bindImageSource();
    this.render();
  }
  forceReload() { this._suppressLoadOnce = false; this.loadFromState(); }

  /* ===== DOM 构建 ===== */
  build() {
    this.root.className = "ets";
    this.root.innerHTML = `
<div class="ets-head">
  <button class="ets-run" title="仅运行到此节点 | Only run up to this node">▶</button>
  <button class="ets-run ets-load" title="加载本地图像文件 | Load image file">📂</button>
  <button class="ets-tabbtn" data-mode="manual" title="手动绘制框图层 | Manual draw boxes">by Manual</button>
  <button class="ets-tabbtn" data-mode="count" title="按列/行数量分块 | Split by column/row count">by count</button>
  <button class="ets-tabbtn" data-mode="size" title="按分块尺寸分块 | Split by tile size">by size</button>
  
  <label class="ets-field" title="对齐基数（分块尺寸必须是该值的倍数） | Alignment base (tile sizes are multiples of this)">
    Div <input type="number" id="ets-div" min="2" max="256" step="2">
  </label>
  
  <label class="ets-field" title="整体统一放大倍数 | Unified output upscale factor">
    U-upscale <input type="number" id="ets-uup" min="0.1" max="16" step="0.05">
  </label>
  
  <span class="only-count">
    <label class="ets-field" title="列数 | Columns count">
      Cols <input type="number" id="ets-ucols" min="1" max="16" step="1">
    </label>
    <label class="ets-field" title="行数 | Rows count">
      Rows <input type="number" id="ets-urows" min="1" max="16" step="1">
    </label>
  </span>
  
  <span class="only-size">
    <label class="ets-field" title="生成分块的目标尺寸（像素） | Target tile size (pixels)">
      tile <input type="number" id="ets-utile" min="8" max="8192" step="16">
    </label>
  </span>
  
  <span class="ets-field" title="相邻分块的重叠像素 | Overlap pixels between tiles">
    overlap <span class="ets-ov-toggle" data-ov="px">px</span><input type="number" id="ets-ovpx" min="0" max="2048" step="8">
  </span>
  <span class="ets-field" title="相邻分块的重叠百分比 | Overlap percentage between tiles">
    <span class="ets-ov-toggle" data-ov="pct">%</span><input type="number" id="ets-ovpct" min="0" max="45" step="1">
  </span>
  
  <span class="only-manual">
    <label class="ets-field" title="自动拆分未被手绘框覆盖的区域 | Smart-split uncovered area">
      fill gaps <input type="checkbox" id="ets-fg">
    </label>
    <label class="ets-field" title="补块的最大边长（像素） | Max side length for gap tiles (pixels)">
      max side <input type="number" id="ets-gmax" min="8" max="8192" step="16">
    </label>
  </span>
</div>
<div class="ets-body">
  <div class="ets-canvaswrap"><div class="ets-stage">
    <div class="ets-wrap"><img class="ets-img" draggable="false"><canvas class="ets-cv"></canvas></div>
    <div class="ets-placeholder"></div>
  </div></div>
  <div class="ets-side">
    <div class="ets-sidehead">
      <div style="display:flex; align-items:center; gap:6px;">
        <span>Layers</span>
        <label class="ets-field" title="开启时图层独立放大参数生效 | Enable independent layer upscale" style="font-size:11px; margin-left:4px;">
          <input type="checkbox" id="ets-lup" style="margin:0;"> alone
        </label>
      </div>
      <span class="ets-count"></span>
    </div>
    <div class="ets-list"></div>
  </div>
</div>
<div class="ets-prompts">
  <div class="ets-ptabs">
    <button class="ets-tabbtn" data-ptab="layer" title="当前选中图层的提示词 | Prompt for the selected layer">layer prompt</button>
    <button class="ets-tabbtn" data-ptab="global" title="应用于所有分块的全局提示词 | Global prompt applied to all tiles">global prompt</button>
	<label class="ets-field" style="margin-left: 8px;" title="开：追加至所有图层提示词后面。关：仅作为没有填写提示词的图层/背景的默认值。 | ON: append to all layer prompts. OFF: only use as default for empty layers.">
      add <input type="checkbox" id="ets-gappend">
    </label>
    <span class="ets-hint"></span>
    <span class="ets-status"></span> 
  </div>
  <textarea class="ets-p" placeholder="positive prompt" title="正向提示词 | Positive prompt"></textarea>
  <textarea class="ets-n" placeholder="negative prompt" title="负向提示词 | Negative prompt"></textarea>
</div>
<input type="file" accept="image/*" style="display:none" class="ets-file">`;
    const q = s => this.root.querySelector(s);
    this.stageEl = q(".ets-stage"); this.wrapEl = q(".ets-wrap");
    this.imgEl = q(".ets-img"); this.cv = q(".ets-cv");
    this.placeholderEl = q(".ets-placeholder"); this.statusEl = q(".ets-status");
    this.listEl = q(".ets-list"); this.countEl = q(".ets-count");
    this.pEl = q("textarea.ets-p"); this.nEl = q("textarea.ets-n");
    this.hintEl = q(".ets-hint"); this.fileEl = q(".ets-file");

    eeMarkForward(this.root);
    this.pEl.setAttribute("data-ee-native-scroll", "");
    this.nEl.setAttribute("data-ee-native-scroll", "");

  }
  _isActive() {
    try {
      const sn = app.canvas?.selected_nodes;
      if (sn && this.node.id != null && sn[this.node.id]) return true;
    } catch (_) {}
    return false;
  }
  
  updateScrollZones() {
    this.listEl.removeAttribute("data-ee-native-scroll");
  }

  /* ===== 事件 ===== */
  attachEvents() {
    this.root.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => {
      this.state.mode = b.dataset.mode; this.saveState(); this.render();
    });
    this.root.querySelectorAll("[data-ptab]").forEach(b => b.onclick = () => {
      this.state.prompt_tab = b.dataset.ptab; this.saveState(); this.render();
    });
	
    const gapp = this.root.querySelector("#ets-gappend");
    gapp.addEventListener("change", () => { 
        this.state.global_append = gapp.checked; 
        this.saveState(); 
        this.render(); 
    });

    const lup = this.root.querySelector("#ets-lup");
    lup.addEventListener("change", () => {
        this.state.enable_layer_upscale = lup.checked; 
        this.saveState(); 
        this.render(); 
    });


    this.root.querySelector(".ets-run").onclick = async () => {
      this.status("queueing isolated run…");
      try { await eeRunIsolatedPromptCore(this.node, "TileSplit"); this.status("queued"); }
      catch (e) { this.status("run failed: " + e.message); }
    };
    this.root.querySelector(".ets-load").onclick = () => this.fileEl.click();
    this.stageEl.addEventListener("click", () => {
      if (!this.srcW && !this.inputConnected()) this.fileEl.click();
    });
    const onNum = (id, fn) => {
      const el = this.root.querySelector("#" + id);
      el.addEventListener("change", () => { fn(parseFloat(el.value) || 0); this.saveState(); this.render(); });
    };
    onNum("ets-div", v => { this.state.div = clamp(Math.round(v) || 16, 2, 256); });
    onNum("ets-uup", v => { this.state.unified.upscale = clamp(v || 2, 0.1, 16); });
    onNum("ets-ucols", v => { this.state.unified.columns = clamp(Math.round(v) || 2, 1, 16); });
    onNum("ets-urows", v => { this.state.unified.rows = clamp(Math.round(v) || 2, 1, 16); });
    onNum("ets-utile", v => { this.state.unified.tile_size = clamp(roundDiv(v || 1024, this.state.div), this.state.div, 8192); });
    onNum("ets-gmax", v => { this.state.gap_max_px = clamp(roundDiv(v || 1536, this.state.div), this.state.div, 8192); });
    onNum("ets-ovpx", v => { this.state.overlap_px = Math.max(0, Math.round(v)); });
    onNum("ets-ovpct", v => { this.state.overlap_pct = clamp(v || 0, 0, 45); });
	
    this.root.querySelectorAll('.ets-ov-toggle').forEach(btn => {
      btn.onclick = () => {
        this.state.overlap_mode = btn.dataset.ov;
        this.saveState(); 
        this.render();
      };
    });
	
	const fg = this.root.querySelector("#ets-fg");
    fg.addEventListener("change", () => { this.state.fill_gaps = fg.checked; this.saveState(); this.render(); });

    this.cv.addEventListener("pointerdown", e => this.onCanvasDown(e));
    this.cv.addEventListener("pointermove", e => this.onCanvasMove(e));
    this.cv.addEventListener("pointerup", e => this.onCanvasUp(e));
    this.cv.addEventListener("pointercancel", () => { this.drag = null; this.previewNewRect = null; this.draw(); });
    this.root.addEventListener("dragover", e => e.preventDefault());
    this.root.addEventListener("drop", e => {
      e.preventDefault();
      const f = [...(e.dataTransfer?.files || [])].find(f => f.type.startsWith("image/"));
      if (f) this.uploadImage(f);
    });
    this.fileEl.addEventListener("change", e => {
      if (e.target.files?.[0]) this.uploadImage(e.target.files[0]);
      e.target.value = "";
    });
    this._unsubs.push(eeAddWindowListener("paste", (e) => {
      if (!this._isActive()) return;
      for (const it of (e.clipboardData?.items || [])) {
        if (it.type?.startsWith("image/")) {
          e.preventDefault(); e.stopPropagation();
          const f = it.getAsFile();
          if (f) this.uploadImage(f);
          return;
        }
      }
    }, true));
    this._unsubs.push(eeAddWindowListener("keydown", (e) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      if (!this._isActive() || this.state.selected == null) return;
      const t = e.target || {};
      const tag = (t.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || t.isContentEditable) return;
      e.preventDefault(); e.stopPropagation();
      this.deleteLayer(this.state.selected);
    }, true));
	
    const onCtrlKey = (down) => (e) => {
      if (e.key !== "Control" && e.key !== "Meta") return;
      if (!this._isActive() || !this._lastDisp) return;
      this.updateHoverCursor(down);
    };
    this._unsubs.push(eeAddWindowListener("keydown", onCtrlKey(true), true));
    this._unsubs.push(eeAddWindowListener("keyup", onCtrlKey(false), true));
	
    const pdeb = fn => { clearTimeout(this._ptimer); this._ptimer = setTimeout(fn, 350); };
    this.pEl.addEventListener("input", () => {
      const t = this.promptTarget();
      if (t) { t.obj[t.pk] = this.pEl.value; pdeb(() => this.saveState()); }
    });
    this.nEl.addEventListener("input", () => {
      const t = this.promptTarget();
      if (t) { t.obj[t.nk] = this.nEl.value; pdeb(() => this.saveState()); }
    });

    this._ro = new ResizeObserver(() => {
      clearTimeout(this._roT);
      this._roT = setTimeout(() => {
        this.layoutCanvas();
        this.draw();
        this.node.setDirtyCanvas?.(true, true);
      }, 100);
    });
    setTimeout(() => this._ro.observe(this.root), 100);
  }
  detachEvents() {
    this._unsubs.forEach(u => { try { u(); } catch (_) {} });
    this._unsubs = [];
    this._ro?.disconnect();
    clearTimeout(this._ptimer); 
	clearTimeout(this._statusTimer);
	clearTimeout(this._roT);
    this.removeDocDragListeners();
  }

  /* ===== 图像源 ===== */
  imgUrl(info) {
    const q = new URLSearchParams();
    q.set("filename", info?.filename || "");
    q.set("subfolder", info?.subfolder || "");
    q.set("type", info?.type || "input");
    return "/view?" + q.toString();
  }
  
  upstreamImageRef() {
    try {
      const inp = (this.node.inputs || []).find(i => i.name === "image");
      const link = inp && inp.link != null ? app.graph.links[inp.link] : null;
      const up = link ? app.graph.getNodeById(link.origin_id) : null;
      if (!up) return null;
      const v = up.widgets?.find(x => x.name === "image")?.value;
      if (Array.isArray(v) && v[0]) return { filename: v[0], subfolder: v[1] || "", type: v[2] || "input" };
      if (typeof v === "string" && v) {
        const m = v.match(/^(.*?)\s*\[([^\]]*)\]$/);
        if (m) return { filename: m[1].trim(), subfolder: m[2].trim(), type: "input" };
        return { filename: v, subfolder: "", type: "input" };
      }
    } catch (_) {}
    return null;
  }
  
  setImg(url, kind, meta) {
    const seq = (this._imgSeq = (this._imgSeq || 0) + 1);
    const loader = new Image();
    loader.onload = () => {
      if (seq !== this._imgSeq) return;
      this._imgKind = kind;
      this.imgEl.src = url;
      this._loadedMeta = (meta && meta.w > 0 && meta.h > 0)
        ? { w: meta.w | 0, h: meta.h | 0, url: this.imgEl.src } : null;
    };
    loader.onerror = () => {
      if (seq !== this._imgSeq) return;
      console.warn("[TileSplit] img failed:", (url || "").slice(0, 120));
      this.status("image unavailable (kept last one)");
    };
    loader.src = url;
  }

  inputConnected() {
    try {
      const inp = (this.node.inputs || []).find(i => i.name === "image");
      if (!inp) return false;
      if (inp.link != null) return true;
      if (typeof this.node.getInputLink === "function") return !!this.node.getInputLink(0);
    } catch (_) {}
    return false;
  }
  bindImageSource() {
    const previewRef = this.state.preview_ref || this._previewRef;
    if (this.inputConnected()) {
      const up = this.upstreamImageRef();
      if (up && up.filename) {
        this.setImg(this.imgUrl(up), "upstream", null);
        return;
      }
      if (previewRef && previewRef.filename) {
        this.setImg(this.imgUrl(previewRef) + "&cb=" + Date.now(), "preview", previewRef);
        return;
      }
      return;
    }
    const s = this.state.source;
    if (s && s.filename) { this.setImg(this.imgUrl(s), "source", null); return; }
    if (previewRef && previewRef.filename) {
      this.setImg(this.imgUrl(previewRef) + "&cb=" + Date.now(), "preview", previewRef);
      return;
    }
    this._imgKind = null;
    this._loadedMeta = null;
    this.imgEl.removeAttribute("src");
  }
  
  hookUpstreamWidget() {
    try {
      ETS_UPSTREAM.add(this);
      const node = this.node;
      const inp = (node.inputs || []).find(i => i.name === "image");
      const link = inp && inp.link != null ? app.graph.links[inp.link] : null;
      const up = link ? app.graph.getNodeById(link.origin_id) : null;
      const w = up?.widgets?.find(x => x.name === "image");
      if (!w || w.__etsUpHooked) return;
      w.__etsUpHooked = true;
      const cb = w.callback;
      w.callback = function (v) {
        const rv = cb?.call(this, v);
        ETS_UPSTREAM.forEach(ui => { try { ui.onUpstreamChanged(); } catch (_) {} });
        return rv;
      };
    } catch (_) {}
  }

  onUpstreamChanged() {
    if (!this.inputConnected()) return;
    const up = this.upstreamImageRef();
    if (!up || !up.filename) return;
    this.setImg(this.imgUrl(up) + "&t=" + Date.now(), "upstream", null);
    this.status("upstream image changed");
  }

  _handleExecuted(out) {
    out = out || {};
    let changed = false;
    
    if (out.src && out.src.w) {
      if (this.state.src_w !== out.src.w || this.state.src_h !== out.src.h) {
        this.state.src_w = out.src.w | 0;
        this.state.src_h = out.src.h | 0;
        this.srcW = this.state.src_w;
        this.srcH = this.state.src_h;
        changed = true;
      }
    }
    
    const ref = out.preview;
    if (ref && ref.filename) {
      const pr = { 
        filename: ref.filename, 
        subfolder: ref.subfolder || "", 
        type: ref.type || "input", 
        w: ref.w || 0, 
        h: ref.h || 0 
      };
      if (!this.state.preview_ref || 
          this.state.preview_ref.filename !== pr.filename || 
          this.state.preview_ref.w !== pr.w || 
          this.state.preview_ref.h !== pr.h) {
        this._previewRef = pr;
        this.state.preview_ref = pr;
        changed = true;
      }
    }
    
    const srcW = out.src ? out.src.w : 0, srcH = out.src ? out.src.h : 0;
    if (out.bg_image && out.bg_image.length > 0) {
      const img = out.bg_image[0];
      const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
      const w = (img.w || srcW || 0) | 0, h = (img.h || srcH || 0) | 0;
      this.setImg(url, "preview", { w, h });
    } else if (ref && ref.filename) {
      const w = (ref.w || srcW || 0) | 0, h = (ref.h || srcH || 0) | 0;
      this.setImg(this.imgUrl(ref) + "&t=" + Date.now(), "preview", { w, h });
    }
    
    // 仅在数据真实改变时保存状态
    if (changed) {
      this.saveState();
    }
    this.render();
  }

  async uploadImage(file) {
    this.status("uploading…");
    try {
      const fd = new FormData();
      fd.append("image", file, "ets_" + Date.now() + "_" + (file.name || "paste.png"));
      fd.append("subfolder", "element_tile_split");
      fd.append("type", "input");
      fd.append("overwrite", "0");
      const r = await fetch("/upload/image", { method: "POST", body: fd });
      const d = await r.json();
      if (!d?.name) throw new Error(d?.error || "upload rejected");
      this.state.source = { filename: d.name, subfolder: d.subfolder || "element_tile_split", type: d.type || "input" };
      this.state.src_w = 0; this.state.src_h = 0;
      this._imgKind = "source";
	  this._loadedMeta = null;
      this.setImg(this.imgUrl(this.state.source), "source", null);
      this.saveState();
      this.status("loaded " + d.name);
    } catch (err) { this.status("upload failed: " + err.message); }
  }

  layoutCanvas() {
    if (!this.srcW || !this.srcH) return;
    const availW = Math.max(80, this.stageEl.clientWidth - 16);
    const availH = Math.max(60, this.stageEl.clientHeight - 16);
    const ar = this.srcW / this.srcH;
    let w = availW, h = w / ar;
    if (h > availH) { h = availH; w = h * ar; }
    w = Math.floor(w); h = Math.floor(h);
    if (this.wrapEl.style.width === w + "px" && this.wrapEl.style.height === h + "px") return;
    this.wrapEl.style.width = w + "px";
    this.wrapEl.style.height = h + "px";
    const dpr = window.devicePixelRatio || 1;
    this.cv.width = Math.round(w * dpr);
    this.cv.height = Math.round(h * dpr);
    this.dispScale = (this.cv.clientWidth || w) / this.srcW;
  }
  outCanvasSize() {
    const s = this.state;
    return [snapCeil(this.srcW * s.unified.upscale, s.div), snapCeil(this.srcH * s.unified.upscale, s.div)];
  }
  layerRegionOut(l) {
    const s = this.state, up = s.unified.upscale, div = s.div;
    const [ow, oh] = this.outCanvasSize();
    const rx = clamp(snap(l.x * up, div), 0, ow - div);
    const ry = clamp(snap(l.y * up, div), 0, oh - div);
    const rw = (l.x + l.w >= this.srcW - 0.5) ? (ow - rx) : clamp(snap(l.w * up, div), div, ow - rx);
    const rh = (l.y + l.h >= this.srcH - 0.5) ? (oh - ry) : clamp(snap(l.h * up, div), div, oh - ry);
    return { x: rx, y: ry, w: rw, h: rh };
  }
  unifiedTiles() {
    const s = this.state;
    const [ow, oh] = this.outCanvasSize();
    let cols, rows;
    if (s.mode === "size") {
      cols = Math.max(1, Math.ceil(ow / s.unified.tile_size));
      rows = Math.max(1, Math.ceil(oh / s.unified.tile_size));
    } else { cols = s.unified.columns; rows = s.unified.rows; }
    const ov = overlapPx(s.overlap_mode, s.overlap_px, s.overlap_pct, Math.min(ow / cols, oh / rows));
    return splitRegion(0, 0, ow, oh, cols, rows, ov);
  }
  gapRectsOut(regions) {
    const s = this.state, div = s.div;
    const [ow, oh] = this.outCanvasSize();
    if (ow <= 0 || oh <= 0 || div <= 0) return [];
    const cacheKey = JSON.stringify(regions) + `|${div}|${s.gap_max_px}|${ow}|${oh}`;
    if (this._gapCache && this._gapCache.key === cacheKey) return this._gapCache.rects;
    if (this.drag && this._gapCache) return this._gapCache.rects;

    const colsN = Math.floor(ow / div), rowsN = Math.floor(oh / div);
    if (colsN <= 0 || rowsN <= 0) return [];
    const covered = Array.from({ length: rowsN }, () => new Array(colsN).fill(false));
    for (const rg of regions) {
      const c0 = Math.floor(rg.x / div), r0 = Math.floor(rg.y / div);
      const c1 = Math.ceil((rg.x + rg.w) / div), r1 = Math.ceil((rg.y + rg.h) / div);
      for (let r = Math.max(0, r0); r < Math.min(rowsN, r1); r++)
        for (let c = Math.max(0, c0); c < Math.min(colsN, c1); c++) covered[r][c] = true;
    }

    const rects = planGaps(covered, div, ow, oh, s.gap_max_px, regions);
    this._gapCache = { key: cacheKey, rects };
    return rects;
  }
  totalTileCount() {
    const s = this.state;
    if (s.mode !== "manual") return this.unifiedTiles().length;
    let n = 0;
    for (const l of s.layers) n += Math.max(1, l.cols) * Math.max(1, l.rows);
    if (s.fill_gaps) n += this.gapRectsOut(s.layers.map(l => this.layerRegionOut(l))).length;
    return n;
  }
  draw() {
    const ctx = this.cv.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = this.cv.width / dpr, H = this.cv.height / dpr;
    ctx.clearRect(0, 0, W, H);
    if (!this.srcW) return;
    const k = this.dispScale || 1;
    const kOut = k / (this.state.unified.upscale || 1);
    const s = this.state;
    ctx.font = "bold 11px Inter, Arial";
    ctx.textBaseline = "middle";
    if (s.mode === "manual") {
      if (s.fill_gaps) {
        const gaps = this.gapRectsOut(s.layers.map(l => this.layerRegionOut(l)));
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = "#7b8494";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const g of gaps) {
          ctx.rect(g.x * kOut, g.y * kOut, g.w * kOut, g.h * kOut);
        }
        ctx.stroke();
        for (const g of gaps) {
          ctx.fillStyle = "rgba(123,132,148,.07)";
          ctx.fillRect(g.x * kOut, g.y * kOut, g.w * kOut, g.h * kOut);
          ctx.fillStyle = "#7b8494";
          ctx.textAlign = "left";
          ctx.fillText("G", g.x * kOut + 5, g.y * kOut + 10);
        }
        ctx.setLineDash([]);
      }
      s.layers.forEach((l, i) => this.drawBox(ctx, l, i, k, l.id === s.selected));
    } else {
      const tiles = this.unifiedTiles();
      ctx.setLineDash([5, 4]);
      for (let i = 0; i < tiles.length; i++) {
        const t = tiles[i];
        ctx.strokeStyle = "#93a7c9"; ctx.lineWidth = 1;
        ctx.strokeRect(t.x * kOut + .5, t.y * kOut + .5, t.w * kOut - 1, t.h * kOut - 1);
        ctx.fillStyle = "#93a7c9"; ctx.textAlign = "left";
        ctx.fillText(String(i + 1), t.x * kOut + 5, t.y * kOut + 10);
      }
      ctx.setLineDash([]);
    }
    if (this.previewNewRect) {
      const d = this.previewNewRect;
      const x = Math.min(d.x0, d.x1) * k, y = Math.min(d.y0, d.y1) * k;
      const w = Math.abs(d.x1 - d.x0) * k, h = Math.abs(d.y1 - d.y0) * k;
      ctx.setLineDash([4, 3]); ctx.strokeStyle = "#43d9d1"; ctx.lineWidth = 1.5;
      ctx.strokeRect(x + .5, y + .5, w, h);
      ctx.setLineDash([]);
    }
    this.refreshStatus();
  }
  drawBox(ctx, l, i, k, sel) {
    const x = l.x * k, y = l.y * k, w = l.w * k, h = l.h * k;
    ctx.fillStyle = hexA(l.color, .12);
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = l.color;
    ctx.lineWidth = sel ? 2.5 : 1.5;
    ctx.strokeRect(x + .5, y + .5, w - 1, h - 1);
    if (sel) {
      const pts = [[x, y], [x + w / 2, y], [x + w, y], [x, y + h / 2], [x + w, y + h / 2], [x, y + h], [x + w / 2, y + h], [x + w, y + h]];
      for (const [px, py] of pts) {
        ctx.fillStyle = "#fff";
        ctx.fillRect(px - 3.5, py - 3.5, 7, 7);
        ctx.strokeStyle = l.color; ctx.lineWidth = 1.2;
        ctx.strokeRect(px - 3.5, py - 3.5, 7, 7);
      }
    }
    const s = 18, bx = x + 2, by = y + 2;
    ctx.fillStyle = l.color;
    ctx.fillRect(bx, by, s, s);
    ctx.strokeStyle = "rgba(0,0,0,.45)"; ctx.lineWidth = 1;
    ctx.strokeRect(bx + .5, by + .5, s - 1, s - 1);
    ctx.fillStyle = "#111"; ctx.textAlign = "center";
    ctx.fillText(String(i + 1), bx + s / 2, by + s / 2 + .5);
  }

  toDisp(e) {
    const r = this.cv.getBoundingClientRect();
    const cw = this.cv.clientWidth || r.width;
    const ch = this.cv.clientHeight || r.height;
    return [(e.clientX - r.left) * (cw / r.width),
            (e.clientY - r.top) * (ch / r.height)];
  }
  toSrc(e) {
    const k = this.dispScale || 1;
    const [dx, dy] = this.toDisp(e);
    return [dx / k, dy / k];
  }
  boxDisp(l) {
    const k = this.dispScale || 1;
    return { x: l.x * k, y: l.y * k, w: l.w * k, h: l.h * k };
  }
  edgeHit(dp, b) {
    const x2 = b.x + b.w, y2 = b.y + b.h;
    if (dp[0] < b.x - HIT || dp[0] > x2 + HIT || dp[1] < b.y - HIT || dp[1] > y2 + HIT) return null;
    const nL = Math.abs(dp[0] - b.x) <= HIT, nR = Math.abs(dp[0] - x2) <= HIT;
    const nT = Math.abs(dp[1] - b.y) <= HIT, nB = Math.abs(dp[1] - y2) <= HIT;
    if (nT && nL) return "nw";
    if (nT && nR) return "ne";
    if (nB && nL) return "sw";
    if (nB && nR) return "se";
    if (nT) return "n";
    if (nB) return "s";
    if (nL) return "w";
    if (nR) return "e";
    return "move";
  }
  hitAt(dp) {
    for (let i = this.state.layers.length - 1; i >= 0; i--) {
      const l = this.state.layers[i];
      const h = this.edgeHit(dp, this.boxDisp(l));
      if (h) return { handle: h, layer: l };
    }
    return null;
  }
  onCanvasDown(e) {
    if (e.button !== 0 || !this._isActive() || !this.srcW) return;
    if (this.state.mode !== "manual") return;
    const dp = this.toDisp(e);
    const [sx, sy] = this.toSrc(e);
    try { this.cv.setPointerCapture(e.pointerId); } catch (_) {}
    if (e.ctrlKey || e.metaKey) {
      this.drag = { type: "new", x0: sx, y0: sy, x1: sx, y1: sy };
      this._lastCursor = "crosshair";
      this._lastHoverId = null;
      this.cv.style.cursor = "crosshair";
      return;
    }
    const hit = this.hitAt(dp);
    if (hit) {
      this.selectLayer(hit.layer.id, false);
      if (hit.handle === "move") {
        this.drag = { type: "move", id: hit.layer.id, ox: sx, oy: sy, orig: { x: hit.layer.x, y: hit.layer.y } };
      } else {
        this.drag = { type: "resize", id: hit.layer.id, handle: hit.handle, orig: { ...hit.layer } };
      }
    } else {
      this.drag = null;
      this.selectLayer(null, true);
    }
  }
  
  updateHoverCursor(forceCrosshair = false) {
    if (this.drag) return;
    const s = this.state;
    if (s.mode !== "manual" || !this._isActive() || !this._lastDisp) return;
    const hit = forceCrosshair ? null : this.hitAt(this._lastDisp);
    const cur = !hit ? "crosshair" : (hit.handle === "move" ? "move" : CUR[hit.handle]);
    const hoverId = hit?.layer.id ?? null;
    if (this._lastCursor !== cur || this._lastHoverId !== hoverId) {
      this._lastCursor = cur;
      this._lastHoverId = hoverId;
      this.cv.style.cursor = cur;
    }
  }
  
  onCanvasMove(e) {
    const dp = this.toDisp(e);
    const [sx, sy] = this.toSrc(e);
    const d = this.drag, s = this.state, div = s.div, min = div * MIN_DIVS;
    this._lastDisp = dp;                       

    if (!d) {
      this.updateHoverCursor(e.ctrlKey || e.metaKey);
      return;
    }

    if (d.type === "new") {
      if (this._lastCursor !== "crosshair") {
        this._lastCursor = "crosshair";
        this._lastHoverId = null;
        this.cv.style.cursor = "crosshair";
      }
      d.x1 = sx; d.y1 = sy;
      this.previewNewRect = { ...d };
      this.draw();
      return;
    }

    // if (d.type === "new") { d.x1 = sx; d.y1 = sy; this.previewNewRect = { ...d }; this.draw(); return; }
    const l = s.layers.find(x => x.id === d.id);
    if (!l) return;
    if (d.type === "move") {
      l.x = clamp(snap(d.orig.x + (sx - d.ox), div), 0, Math.max(0, this.srcW - l.w));
      l.y = clamp(snap(d.orig.y + (sy - d.oy), div), 0, Math.max(0, this.srcH - l.h));
    } else {
      const o = d.orig, h = d.handle;
      let x = o.x, y = o.y, x2 = o.x + o.w, y2 = o.y + o.h;
      if (h.includes("e")) x2 = clamp(snap(sx, div), o.x + min, this.srcW);
      if (h.includes("s")) y2 = clamp(snap(sy, div), o.y + min, this.srcH);
      if (h.includes("w")) x = clamp(snap(sx, div), 0, o.x + o.w - min);
      if (h.includes("n")) y = clamp(snap(sy, div), 0, o.y + o.h - min);
      l.x = x; l.y = y; l.w = x2 - x; l.h = y2 - y;
    }
    this.draw();
  }
  onCanvasUp(e) {
    const d = this.drag;
    this.drag = null; this.previewNewRect = null;
    if (!d) return;
    if (d.type === "new") this.finalizeNew(d);
    else {
      this.clampLayer(this.state.layers.find(x => x.id === d.id));
      this.saveState(); this.render();
    }
    this.updateHoverCursor(!!(e && (e.ctrlKey || e.metaKey)));
  }
  finalizeNew(d) {
    const div = this.state.div, min = div * MIN_DIVS;
    const rx0 = Math.min(d.x0, d.x1), ry0 = Math.min(d.y0, d.y1);
    const rx1 = Math.max(d.x0, d.x1), ry1 = Math.max(d.y0, d.y1);
    if (rx1 - rx0 < div * 0.5 || ry1 - ry0 < div * 0.5) return;
    const x = clamp(snap(rx0, div), 0, Math.max(0, this.srcW - min));
    const y = clamp(snap(ry0, div), 0, Math.max(0, this.srcH - min));
    const w = clamp(snap(rx1 - rx0, div), min, this.srcW - x);
    const h = clamp(snap(ry1 - ry0, div), min, this.srcH - y);
    const layer = {
      id: Date.now() + Math.floor(Math.random() * 1000), color: this.nextColor(),
      x, y, w, h, upscale: this.state.unified.upscale, cols: 1, rows: 1, prompt: "", negative: "",
    };
    this.state.layers.push(layer);
    this.state.selected = layer.id;
    this.state.prompt_tab = "layer";
    this.saveState(); this.render();
  }
  nextColor() {
    const used = new Set(this.state.layers.map(l => l.color));
    for (const c of PALETTE) if (!used.has(c)) return c;
    return PALETTE[this.state.layers.length % PALETTE.length];
  }
  clampLayer(l) {
    if (!l) return;
    const min = this.state.div * MIN_DIVS;
    l.w = clamp(l.w, min, this.srcW); l.h = clamp(l.h, min, this.srcH);
    l.x = clamp(l.x, 0, this.srcW - l.w); l.y = clamp(l.y, 0, this.srcH - l.h);
  }
  clampLayersToBounds() {
    if (!this.srcW) return;
    this.state.layers.forEach(l => this.clampLayer(l));
    this.state.layers = this.state.layers.filter(l => l.w > 0 && l.h > 0);
  }
  deleteLayer(id) {
    const i = this.state.layers.findIndex(l => String(l.id) === String(id));
    if (i < 0) return;
    this.state.layers.splice(i, 1);
    if (String(this.state.selected) === String(id)) this.state.selected = null;
    this.saveState(); this.render();
  }
  selectLayer(id, save = true) {
    this.state.selected = id;
    this.listEl.querySelectorAll(".ets-row").forEach(r => r.classList.toggle("sel", String(r.dataset.id) === String(id)));
    this.syncPromptPanel(); this.draw();
    if (save) this.saveState();
  }

  /* ===== 层列表 ===== */
  buildList() {
    const s = this.state;
    const manual = s.mode === "manual";
    let html = "";
    if (!manual) {
      const tiles = this.unifiedTiles();
      html += `<div class="ets-row gaprow" data-id="u">
<span class="ets-badge" style="background:#93a7c9">U</span>
<div class="ets-rowmain"><div class="ets-rowparams">
<span>Unified grid · ${tiles.length} tiles · read-only</span></div></div>
</div>`;
    } else {
      s.layers.forEach((l, i) => {
        const upDisabled = !s.enable_layer_upscale; 
        html += `<div class="ets-row ${l.id === s.selected ? "sel" : ""}" data-id="${l.id}">
<div class="ets-thumb"><canvas width="44" height="33" data-thumb="${l.id}"></canvas></div>
<span class="ets-badge" style="background:${l.color}">${i + 1}</span>
<div class="ets-rowmain">
  <div class="ets-rowparams">
    <label title="该图层独立的放大倍数 | Independent upscale factor for this layer">upscale<input type="number" step="0.1" min="0.1" max="16" value="${l.upscale}" data-f="upscale" ${upDisabled ? 'disabled style="opacity:0.4"' : ''}></label>
  </div>
  <div class="ets-rowparams">
    <label title="该图层内部横向切分的列数 | Number of columns to split this layer">cols<input type="number" min="1" max="16" value="${l.cols}" data-f="cols"></label>
    <label title="该图层内部纵向切分的行数 | Number of rows to split this layer">rows<input type="number" min="1" max="16" value="${l.rows}" data-f="rows"></label>
  </div>
</div>
<button class="ets-del" title="删除此图层 | Delete this layer">✕</button>
</div>`;
      });
      if (s.fill_gaps) {
        const n = this.gapRectsOut(s.layers.map(l => this.layerRegionOut(l))).length;
        html += `<div class="ets-row gaprow" data-id="g">
<span class="ets-badge" style="background:#7b8494">G</span>
<div class="ets-rowmain"><div class="ets-rowparams">
<span>gap fill · ${n} tiles · max ${s.gap_max_px}px</span>
</div></div>
</div>`;
      }
      if (!s.layers.length) html += `<div class="ets-hint" style="padding:8px;color:#718096">
Ctrl + drag on canvas to draw a layer box</div>`;
    }
    
    this.listEl.innerHTML = html;
    this.countEl.textContent = `${this.totalTileCount()} tiles · ${this.outCanvasSize().join("×")}px`;

    if (manual) this.listEl.querySelectorAll("canvas[data-thumb]").forEach(cv => {
      const l = s.layers.find(x => String(x.id) === cv.dataset.thumb);
      if (l) this.drawThumb(cv, l);
    });
    if (manual) this.listEl.querySelectorAll(".ets-row[data-id]:not(.gaprow)").forEach(row => {
      const id = Number(row.dataset.id) || row.dataset.id;
      row.addEventListener("pointerdown", e => this.onRowDown(e, id));
      
      row.querySelector(".ets-del").onclick = e => { e.stopPropagation(); this.deleteLayer(id); };
      
      row.querySelectorAll("input[data-f]").forEach(inp => {
        // 使用 input 实时更新，不触发全量重绘
        inp.addEventListener("input", () => {
          const l = s.layers.find(x => String(x.id) === String(id));
          if (!l) return;
          const f = inp.dataset.f;
          if (f === "upscale") l.upscale = clamp(parseFloat(inp.value) || 1, 0.1, 16);
          if (f === "cols") l.cols = clamp(Math.round(parseFloat(inp.value)) || 1, 1, 16);
          if (f === "rows") l.rows = clamp(Math.round(parseFloat(inp.value)) || 1, 1, 16);
          this.saveState(); 
          this.draw(); 
        });
        // 失去焦点时同步数值，且不触发整个页面的重建
        inp.addEventListener("change", () => {
          const l = s.layers.find(x => String(x.id) === String(id));
          if (!l) return;
          const f = inp.dataset.f;
          if (f === "upscale") inp.value = l.upscale;
          if (f === "cols") inp.value = l.cols;
          if (f === "rows") inp.value = l.rows;
        });
      });
    });
    this.updateScrollZones();
  }
  drawThumb(cv, l) {
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    const im = this.imgEl;
    if (!im?.naturalWidth || !this.srcW) return;
    const fx = im.naturalWidth / this.srcW, fy = im.naturalHeight / this.srcH;
    const sx = l.x * fx, sy = l.y * fy, sw = Math.max(1, l.w * fx), sh = Math.max(1, l.h * fy);
    const ar = sw / sh, cw = cv.width, ch = cv.height;
    let dw = cw, dh = cw / ar;
    if (dh > ch) { dh = ch; dw = ch * ar; }
    try { ctx.drawImage(im, sx, sy, sw, sh, (cw - dw) / 2, (ch - dh) / 2, dw, dh); } catch (_) {}
  }

  /* ---- 行选中 + 拖拽排序 ---- */
  onRowDown(e, id) {
    if (e.button !== 0) return;
    if (e.target.closest("input,button,textarea")) return;
    this.pending = { id, y0: e.clientY, moved: false };
    this.selectLayer(id, true);
    this._docMove = (ev) => this.onDocMove(ev);
    this._docUp = (ev) => this.onDocUp(ev);
    document.addEventListener("pointermove", this._docMove);
    document.addEventListener("pointerup", this._docUp);
  }
  onDocMove(ev) {
    const p = this.pending;
    if (!p) return;
    if (!p.moved) {
      if (Math.abs(ev.clientY - p.y0) <= 4 || this.state.mode !== "manual") return;
      p.moved = true;
      document.body.classList.add("ets-reordering");
      this.insertEl = document.createElement("div");
      this.insertEl.className = "ets-insert";
      this.listEl.appendChild(this.insertEl);
    }
    const rows = [...this.listEl.querySelectorAll('.ets-row[data-id]:not(.gaprow)')];
    let idx = rows.length, refY = null;
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i].getBoundingClientRect();
      if (ev.clientY < r.top + r.height / 2) { idx = i; refY = r.top; break; }
    }
    if (refY == null && rows.length) refY = rows[rows.length - 1].getBoundingClientRect().bottom;
    if (refY != null) {
      const lr = this.listEl.getBoundingClientRect();
      this.insertEl.style.top = (refY - lr.top + this.listEl.scrollTop) + "px";
    }
    this.reorder = { id: p.id, idx };
  }
  onDocUp() {
    const r = this.reorder;
    this.removeDocDragListeners();
    if (r) {
      const from = this.state.layers.findIndex(l => String(l.id) === String(r.id));
      if (from >= 0) {
        const [item] = this.state.layers.splice(from, 1);
        const to = clamp(r.idx > from ? r.idx - 1 : r.idx, 0, this.state.layers.length);
        this.state.layers.splice(to, 0, item);
      }
    }
    this.reorder = null; this.pending = null;
    this.saveState(); this.render();
  }
  removeDocDragListeners() {
    if (this._docMove) document.removeEventListener("pointermove", this._docMove);
    if (this._docUp) document.removeEventListener("pointerup", this._docUp);
    this._docMove = this._docUp = null;
    document.body.classList.remove("ets-reordering");
    this.insertEl?.remove(); this.insertEl = null;
  }

  /* ===== 提示词面板 ===== */
  promptTarget() {
    const s = this.state;
    if (s.prompt_tab === "layer") {
      const l = s.layers.find(x => String(x.id) === String(s.selected));
      return l ? { obj: l, pk: "prompt", nk: "negative" } : null;
    }
    return { obj: s, pk: "global_prompt", nk: "global_negative" };
  }
  syncPromptPanel() {
    const s = this.state, t = this.promptTarget();
    this.root.querySelectorAll("[data-ptab]").forEach(b => b.classList.toggle("on", b.dataset.ptab === s.prompt_tab));
    const layerMode = s.prompt_tab === "layer";
    this.pEl.disabled = this.nEl.disabled = (layerMode && !t);
    this.pEl.value = t ? (t.obj[t.pk] || "") : "";
    this.nEl.value = t ? (t.obj[t.nk] || "") : "";
    this.hintEl.textContent = layerMode
      ? (t ? "editing selected layer" : "select a layer to edit its prompt")
      : "applies to every tile";
  }

  /* ===== 状态/头部同步 ===== */
  syncHead() {
    const s = this.state;
    this.root.classList.toggle("m-manual", s.mode === "manual");
    this.root.classList.toggle("m-count", s.mode === "count");
    this.root.classList.toggle("m-size", s.mode === "size");
    this.root.querySelectorAll("[data-mode]").forEach(b => b.classList.toggle("on", b.dataset.mode === s.mode));
    const set = (id, v) => {
      const el = this.root.querySelector("#" + id);
      if (el && document.activeElement !== el) el.value = v;
    };
    set("ets-div", s.div); set("ets-uup", s.unified.upscale);
    set("ets-ucols", s.unified.columns); set("ets-urows", s.unified.rows);
    set("ets-utile", s.unified.tile_size);
    set("ets-ovpx", s.overlap_px); set("ets-ovpct", s.overlap_pct);
    set("ets-gmax", s.gap_max_px);
	
    const fg = this.root.querySelector("#ets-fg");
    if (document.activeElement !== fg) fg.checked = !!s.fill_gaps;

    const lup = this.root.querySelector("#ets-lup");
    if (document.activeElement !== lup) lup.checked = !!s.enable_layer_upscale;

    const gapp = this.root.querySelector("#ets-gappend");
    if (document.activeElement !== gapp) gapp.checked = !!s.global_append;
	
    this.root.querySelectorAll('.ets-ov-toggle').forEach(btn => {
      btn.classList.toggle("ovon", s.overlap_mode === btn.dataset.ov);
    });
  }
  render() {
    this.syncHead(); this.layoutCanvas(); this.buildList(); this.draw(); this.syncPromptPanel();
    if (!this.srcW) {
      this.placeholderEl.style.display = "flex";
      this.placeholderEl.textContent = this.inputConnected()
        ? "image input connected — press ▶ (run to this node) once to display it\nor drop / paste / click here to load an image"
        : "drop / paste an image here, or click to load a file\n(image input port takes priority)";
    } else this.placeholderEl.style.display = "none";
  }
  status(msg) {
    this._statusAt = Date.now();
    this.statusEl.textContent = msg;
    clearTimeout(this._statusTimer);
    this._statusTimer = setTimeout(() => this.refreshStatus(), 2500);
  }
  refreshStatus() {
    if (Date.now() - (this._statusAt || 0) < 2500) return;
    this.statusEl.textContent = "";
  }
}

TileSplitUI.prototype._adoptDims = function (w, h) {
  const changed = (this.state.src_w | 0) !== w || (this.state.src_h | 0) !== h;
  this.srcW = w;
  this.srcH = h;
  if (changed) {
    this.state.src_w = w;
    this.state.src_h = h;
    this.state.layers = [];
    this.state.selected = null;
  }
  this.clampLayersToBounds();
  this.saveState();
  this.layoutCanvas();
  this.render();
};

TileSplitUI.prototype.attachImageHooks = function () {
  this.imgEl.addEventListener("load", () => {
    const nw = this.imgEl.naturalWidth, nh = this.imgEl.naturalHeight;
    if (!nw || !nh) return;
    if (this._loadedMeta && this._loadedMeta.url === this.imgEl.src
      && this._loadedMeta.w > 0 && this._loadedMeta.h > 0) {
      this._adoptDims(this._loadedMeta.w, this._loadedMeta.h);
      return;
    }
    const rw = this.state.src_w | 0, rh = this.state.src_h | 0;
    if (rw === nw && rh === nh) { this._adoptDims(nw, nh); return; }
    if (rw && rh && Math.max(rw, rh) > PREVIEW_MAX_EDGE) {
      const rRec = rw / rh, rNat = nw / nh;
      if (nw <= rw && nh <= rh && Math.abs(rRec - rNat) / rRec < 0.02) {
        this._adoptDims(rw, rh);
        return;
      }
    }
    this._adoptDims(nw, nh);
  });

  this.imgEl.addEventListener("error", () => {
    console.warn("[TileSplit] img load failed:", (this.imgEl.src || "").slice(0, 120));
    if (this._imgKind === "source") {
      const ref = this.state.preview_ref || this._previewRef;
      if (ref && ref.filename) {
        this.setImg(this.imgUrl(ref) + "&cb=" + Date.now(), "preview", ref);
        return;
      }
    }
    this.status("image failed to load");
  });
};

app.registerExtension({
  name: "Element.TileSplitUI",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ElementTileSplit") return;
    installStyles();

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      hideWidgetAndSlot(node, "layout_data");
      node.__etsFresh = true;

      const root = document.createElement("div");
      root.style.position = "absolute";
      root.style.inset = "0";
      const wrapper = document.createElement("div");
      wrapper.style.position = "relative";
      wrapper.style.width = "100%";
      wrapper.style.height = "100%";
      wrapper.appendChild(root);

      const domWidget = node.addDOMWidget("tile_split_ui", "div", wrapper, { serialize: false });
      domWidget.serialize = false;
      domWidget.serializeValue = () => undefined;

      let stateWidget = node.widgets?.find(w => w.name === "layout_data");
      if (stateWidget) bindStateWidget(node, stateWidget);
      else setTimeout(() => bindStateWidget(node, node.widgets?.find(w => w.name === "layout_data")), 0);

      node.__ets = new TileSplitUI(node, root, stateWidget);
	  node.__ets.hookUpstreamWidget();
      node.__ets.attachImageHooks();

      const MIN_NODE_WIDTH = 420;
      const MIN_NODE_HEIGHT = 460;
      const origComputeSize = node.computeSize;
      node.computeSize = function (out) {
        const size = origComputeSize ? origComputeSize.apply(this, arguments) : [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];
        if (size[0] < MIN_NODE_WIDTH) size[0] = MIN_NODE_WIDTH;
        if (size[1] < MIN_NODE_HEIGHT) size[1] = MIN_NODE_HEIGHT;
        return size;
      };
      if (typeof node.id !== "number" || node.id < 0) node.size = [900, 640];
      node.onResize = function () {
        node.__ets?.layoutCanvas();
        node.__ets?.draw();
        node.setDirtyCanvas?.(true, true);
      };
      return r;
    };

    const onConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConn?.apply(this, arguments);
      clearTimeout(this.__etsConnT);
      this.__etsConnT = setTimeout(() => {
        this.__ets?.hookUpstreamWidget();
        this.__ets?.bindImageSource();
      }, 250);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      this.__etsFresh = false;
      if (this.__ets && !this.__ets.stateWidget) {
        bindStateWidget(this, this.widgets?.find(w => w.name === "layout_data"));
      }
      this.__ets?.forceReload();
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      this.__ets?._handleExecuted(output);
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this.__ets) { try { ETS_UPSTREAM.delete(this.__ets); } catch (_) {} }
      this.__ets?.detachEvents();
      this.__ets = null;
      return onRemoved?.apply(this, arguments);
    };
  },
});