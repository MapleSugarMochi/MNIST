"use strict";

const drawing = new window.MnistCanvas(document.getElementById("canvas"));
const writerInput = document.getElementById("writer-id");
const labelInput = document.getElementById("digit-label");
const statusElement = document.getElementById("status");
const samples = [];
writerInput.value = `writer-${crypto.randomUUID().slice(0, 8)}`;

function updateStatus(message = "") {
    statusElement.textContent = `已保存 ${samples.length} 条。${message}`;
}

document.getElementById("clear-button").addEventListener("click", () => drawing.clear());
document.getElementById("save-button").addEventListener("click", () => {
    const writer = writerInput.value.trim();
    if (!drawing.hasInk || !writer) {
        updateStatus("请填写匿名编号并书写。");
        return;
    }
    const image = drawing.exportPNG();
    if (samples.some((sample) => sample.image === image)) {
        updateStatus("这张笔迹已保存，请重新书写。");
        return;
    }
    samples.push({ id: crypto.randomUUID(), label: Number(labelInput.value),
        writer_id: writer, image, created_at: new Date().toISOString() });
    drawing.clear();
    updateStatus();
});
document.getElementById("undo-button").addEventListener("click", () => {
    samples.pop();
    updateStatus();
});
document.getElementById("export-button").addEventListener("click", () => {
    if (!samples.length) {
        updateStatus("请先保存样本。");
        return;
    }
    const corpus = { schema_version: 1, source: "human_canvas", samples };
    const url = URL.createObjectURL(new Blob([JSON.stringify(corpus, null, 2)],
        { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `canvas-${Date.now()}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    updateStatus("已导出；请保留文件。");
});
window.addEventListener("beforeunload", (event) => {
    if (samples.length) {
        event.preventDefault();
        event.returnValue = "";
    }
});
updateStatus();
