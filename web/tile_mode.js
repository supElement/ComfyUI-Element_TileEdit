import { app } from "../../scripts/app.js";

const TARGET_NODES = new Set(["TiledUpscale"]); // 仅一体化节点保留 tile_mode 切换
const SHOW_ON = ["columns", "rows"]; // tile_mode 打开时显示
const SHOW_OFF = ["tile_size"]; // tile_mode 关闭时显示
const HIDDEN_TYPE = "converted-widget";

function snapshot(node) {
  if (node._tileModeSnapshot) return;
  const types = {}, sizes = {};
  for (const w of node.widgets || []) { types[w.name] = w.type; sizes[w.name] = w.computeSize; }
  node._tileModeSnapshot = { types, sizes };
}

function setWidgetVisible(node, name, show) {
  const w = (node.widgets || []).find((x) => x.name === name);
  if (!w) return;
  const snap = node._tileModeSnapshot;
  w.hidden = !show;
  if (!w.options) w.options = {};
  w.options.hidden = !show;
  if (snap.types[name] !== undefined) w.type = show ? snap.types[name] : HIDDEN_TYPE;
  if (show) { if (snap.sizes[name]) w.computeSize = snap.sizes[name]; else delete w.computeSize; }
  else { w.computeSize = () => [0, -4]; }
  w.serializeValue = () => w.value;
  for (const key of ["element", "inputEl"]) {
    const el = w[key];
    if (el && el.style) el.style.display = show ? "" : "none";
  }
}

function applyMode(node) {
  snapshot(node);
  const mode = (node.widgets || []).find((x) => x.name === "tile_mode");
  if (!mode) return;
  const on = !!mode.value;
  for (const name of SHOW_ON) setWidgetVisible(node, name, on);
  for (const name of SHOW_OFF) setWidgetVisible(node, name, !on);
  if (node.setSize) node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
}

app.registerExtension({
  name: "Element.TileModeToggle",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODES.has(nodeData.name)) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const node = this;
      const mode = (node.widgets || []).find((x) => x.name === "tile_mode");
      if (mode) {
        const origCallback = mode.callback;
        mode.callback = function () {
          const rv = origCallback ? origCallback.apply(this, arguments) : undefined;
          applyMode(node); setTimeout(() => applyMode(node), 0);
          return rv;
        };
      }
      applyMode(node); setTimeout(() => applyMode(node), 0);
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      applyMode(this); setTimeout(() => applyMode(this), 0);
      return r;
    };
  },
});
