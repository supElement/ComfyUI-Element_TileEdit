/* ============================================================
 * ee_canvas_utils.js —— ComfyUI 画布事件透传（滚轮缩放 + 中键平移）
 * ============================================================ */
import { app } from "../../scripts/app.js"; // 

const EDITABLE = new Set(["INPUT", "TEXTAREA", "SELECT"]);
const isEditable = (t) => !!t && (EDITABLE.has(t.tagName) || t.isContentEditable);
const canScroll = (el) => !!el &&
  (el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1);
const inCanvas = (t) => !!app.canvasEl && app.canvasEl.contains(t);

function localXY(e) {
  const r = app.canvasEl.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function syncMouse(e) {
  if (!app.canvas || !app.canvas.mouse) return;
  const [x, y] = localXY(e);
  app.canvas.mouse[0] = x;
  app.canvas.mouse[1] = y;
  if (app.canvas.graph_mouse) {
    app.canvas.graph_mouse[0] = x;
    app.canvas.graph_mouse[1] = y;
  }
}

function dispatchWheel(e) {
  const ev = new WheelEvent("wheel", {
    clientX: e.clientX, clientY: e.clientY,
    deltaX: e.deltaX, deltaY: e.deltaY, deltaZ: e.deltaZ, deltaMode: e.deltaMode,
    ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, altKey: e.altKey, metaKey: e.metaKey,
    bubbles: true, cancelable: true,
  });
  syncMouse(e);
  const lc = app.canvas;
  try { if (lc && typeof lc.processWheel === "function") { lc.processWheel(ev); return true; } } catch (_) {}
  try { if (lc && typeof lc.processMouseWheel === "function") { lc.processMouseWheel(ev); return true; } } catch (_) {}
  try { if (app.canvasEl) { app.canvasEl.dispatchEvent(ev); return true; } } catch (_) {}
  const ds = lc && lc.ds;
  if (ds && typeof ds.changeScale === "function") {
    const [lx, ly] = localXY(e);
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const ns = Math.min(ds.max_scale ?? 10, Math.max(ds.min_scale ?? 0.05, ds.scale * factor));
    ds.changeScale(ns, [lx, ly]);
    lc.setDirty && lc.setDirty(true, true);
    return true;
  }
  return false;
}

export function eeInstallCanvasPassthrough() {
  if (window.__eeCanvasPassthrough) return;
  window.__eeCanvasPassthrough = true;

  window.addEventListener("wheel", (e) => {
    if (inCanvas(e.target)) return;
    if (!e.target.closest || !e.target.closest("[data-ee-forward-wheel]")) return;
    if (e.target.closest("[data-ee-forward-force]")) {
      if (dispatchWheel(e)) { e.preventDefault(); e.stopPropagation(); }
      return;
    }
    if (e.target.closest("[data-ee-native-scroll]")) return;
    if (isEditable(e.target) && canScroll(e.target)) return;
    if (dispatchWheel(e)) { e.preventDefault(); e.stopPropagation(); }
  }, { capture: true, passive: false });

let panId = null, lastX = 0, lastY = 0, mode = 0, snap = null;

if (!document.getElementById("ee-pan-cursor-style")) {
  const st = document.createElement("style");
  st.id = "ee-pan-cursor-style";
  st.textContent = "body.ee-panning, body.ee-panning * { cursor: grabbing !important; }";
  document.head.appendChild(st);
}

window.addEventListener("pointerdown", (e) => {
  if (e.button !== 1 || panId !== null) return;
  if (inCanvas(e.target)) return;
  const zone = e.target.closest && e.target.closest("[data-ee-forward-wheel]");
  if (!zone) return;
  e.preventDefault();   
  const ds = app.canvas && app.canvas.ds;
  panId = e.pointerId; mode = 0;
  lastX = e.clientX; lastY = e.clientY;
  snap = ds ? [ds.offset[0], ds.offset[1]] : null;
  document.body.classList.add("ee-panning");   
}, true);


window.addEventListener("pointermove", (e) => {
  if (panId === null || e.pointerId !== panId) return;
  const ds = app.canvas && app.canvas.ds;
  if (!ds) return;

  if (mode === 0) {
    const nativeMoving = snap && (Math.abs(ds.offset[0] - snap[0]) > 0.01 ||
                                  Math.abs(ds.offset[1] - snap[1]) > 0.01);
    mode = nativeMoving ? 2 : 1;
  }
  if (mode === 2) return;      
  e.preventDefault();
  const k = 1 / (ds.scale || 1);
  ds.offset[0] += (e.clientX - lastX) * k;
  ds.offset[1] += (e.clientY - lastY) * k;
  lastX = e.clientX; lastY = e.clientY;
  app.canvas.setDirty && app.canvas.setDirty(true, true);
}, { passive: false });

const endPan = (e) => {
  if (panId === null) return;
  if (e && e.pointerId !== undefined && e.pointerId !== panId) return;
  panId = null; mode = 0; snap = null;
  document.body.classList.remove("ee-panning");   
};

window.addEventListener("pointerup", endPan, true);
window.addEventListener("pointercancel", endPan, true);
window.addEventListener("blur", () => endPan(null), true);

}

export function eeMarkForward(el) {
  if (!el) return;
  eeInstallCanvasPassthrough();
  el.setAttribute("data-ee-forward-wheel", "1");
  const wrap = el.parentElement;
  if (wrap && wrap.childElementCount === 1 && !wrap.hasAttribute("data-ee-forward-wheel")) {
    wrap.setAttribute("data-ee-forward-wheel", "1");
  }
}
