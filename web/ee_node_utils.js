// ee_node_utils.js —— ComfyUI_Element_easy 前端节点级共享工具。
//
// 收敛原先散落在 Hue×3 / ImageCurve / SigmaGraph / image_loader / pad_blur
// 等文件中的逐字拷贝：
// - isNodes2_0：Nodes2.0（comfy-vue 前端）检测，×6 份 → 1 份
// - hideWidgetAndSlot：隐藏 widget 并移除同名输入槽，×6 份 → 1 份
//   （注意：image_loader_previews.js 的变体使用 "hidden_custom" + Nodes2.0
//    分支，属有意分叉，未收敛）
// - eeRunIsolatedPromptCore：隔离执行核心（直提依赖子图 /prompt，失败回退
//   legacy mute 路径），×6 份 → 1 份
// - eeRunIsolatedPreview：Hue 系入口包装（image 输入连接检查）
// - eeAddWindowListener：window 捕获监听的单例分发器。原先每个节点实例都往
//   window 上各注册一份 wheel/paste 捕获监听（多实例 = N 个全局处理器，各自
//   靠 contains 兜底）；改为模块级只注册一个监听，按 Set 分发给各活跃实例。
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const _eeWinHandlers = new Map();

export function eeAddWindowListener(type, handler, opts) {
    let entry = _eeWinHandlers.get(type);
    if (!entry) {
        entry = {
            handlers: new Set(),
            wrapped: (e) => {
                for (const h of [...entry.handlers]) {
                    try { h(e); } catch (err) { console.error(err); }
                }
            },
        };
        _eeWinHandlers.set(type, entry);
    }
    if (entry.handlers.size === 0) {
        window.addEventListener(type, entry.wrapped, opts);
    }
    entry.handlers.add(handler);
    let removed = false;
    return () => {
        if (removed) return;
        removed = true;
        entry.handlers.delete(handler);
        if (entry.handlers.size === 0) {
            window.removeEventListener(type, entry.wrapped, opts);
        }
    };
}

export function isNodes2_0() {
    return (
        !!document.querySelector("comfy-app") ||
        !!document.querySelector(".comfy-vue") ||
        (window.comfyAPI && window.comfyAPI.vue)
    );
}

export function hideWidgetAndSlot(node, widgetName) {
    const w = node.widgets?.find(x => x.name === widgetName);
    if (w) {
        w.type = "hidden";
        w.hidden = true;
        w.computeSize = () => [0, 0];
        w.draw = () => {};
        if (w.inputEl) {
            w.inputEl.style.display = "none";
            if (w.inputEl.parentElement) w.inputEl.parentElement.style.display = "none";
        }
    }
    if (node.inputs) {
        const idx = node.inputs.findIndex(i => i.name === widgetName);
        if (idx !== -1) node.removeInput(idx);
    }
}

async function legacyIsolatedRun(node, logTag) {
    const p = await app.graphToPrompt();
    const prompt = p.output;
    const isolatedPrompt = {};
    const traceDependencies = (nodeId) => {
        if (!prompt[nodeId] || isolatedPrompt[nodeId]) return;
        isolatedPrompt[nodeId] = prompt[nodeId];
        const inputs = prompt[nodeId].inputs;
        for (let key in inputs) {
            const val = inputs[key];
            if (Array.isArray(val) && val.length === 2) {
                traceDependencies(String(val[0]));
            }
        }
    };
    traceDependencies(String(node.id));

    const originalGraphToPrompt = app.graphToPrompt;
    const virtualNodeTypes = [
        "NodeAny", "GetNodeAny",
        "SetImage", "GetImage", "SetLatent", "GetLatent"
    ];
    app.graphToPrompt = async function (...args) {
        const originalModes = new Map();
        for (const n of app.graph._nodes) {
            originalModes.set(n.id, n.mode);
            const isVirtual = virtualNodeTypes.some(t => n.type?.includes(t));
            if (!isolatedPrompt[String(n.id)] && !isVirtual) {
                n.mode = 2; // Mute 无关节点
            } else {
                n.mode = 0;
            }
        }
        try {
            return await originalGraphToPrompt.apply(this, args);
        } finally {
            for (const n of app.graph._nodes) {
                const m = originalModes.get(n.id);
                if (m !== undefined) n.mode = m;
            }
        }
    };
    try {
        await app.queuePrompt(0, 1);
    } finally {
        if (app.graphToPrompt !== originalGraphToPrompt) {
            app.graphToPrompt = originalGraphToPrompt;
        }
    }
}

export async function eeRunIsolatedPromptCore(node, logTag = "IsolatedPreview") {
    // 隔离执行核心：直提本节点依赖子图 → POST /prompt（front），
    // 失败时回退 legacy mute 路径（临时 Mute 无关节点再执行）。
    try {
        const p = await app.graphToPrompt();
        const prompt = p.output;
        const selectedNodeId = String(node.id);

        const isolatedPrompt = {};
        const traceDependencies = (nodeId) => {
            if (!prompt[nodeId] || isolatedPrompt[nodeId]) return;
            isolatedPrompt[nodeId] = prompt[nodeId];
            const inputs = prompt[nodeId].inputs;
            for (let key in inputs) {
                const val = inputs[key];
                if (Array.isArray(val) && val.length === 2) {
                    traceDependencies(String(val[0]));
                }
            }
        };
        traceDependencies(selectedNodeId);
        if (!isolatedPrompt[selectedNodeId]) {
            console.warn(`[${logTag}] node not found in prompt:`, selectedNodeId);
            return;
        }
        const res = await api.fetchApi("/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: isolatedPrompt,
                client_id: api.clientId ?? undefined,
                extra_data: { front: true },
            }),
        });
        if (!res.ok) {
            const errText = await res.text().catch(() => "");
            throw new Error(`/prompt ${res.status}: ${errText.slice(0, 300)}`);
        }
        try {
            const j = await res.json();
            console.log(`[${logTag}] queued:`, j?.prompt_id);
        } catch (e) {}
    } catch (err) {
        console.error(`[${logTag}] direct run failed, trying legacy mute path:`, err);
        try {
            await legacyIsolatedRun(node, logTag);
        } catch (e2) {
            console.error(`[${logTag}] legacy path failed:`, e2);
        }
    }
}

export async function eeRunIsolatedPreview(node, logTag = "IsolatedPreview") {
    const isImageConnected = () => {
        return node.inputs?.some(i => i.name === "image" && i.link !== null);
    };
    if (!isImageConnected()) {
        console.log("No image input connected, skipping preview");
        return;
    }
    return eeRunIsolatedPromptCore(node, logTag);
}
