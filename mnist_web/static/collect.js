"use strict";

const drawing = new window.MnistCanvas(document.getElementById("canvas"));
const writerInput = document.getElementById("writer-id");
const labelInput = document.getElementById("digit-label");
const statusElement = document.getElementById("status");
const samples = [];
writerInput.value = `writer-${crypto.randomUUID().slice(0, 8)}`;

function updateStatus(message = "") {
    statusElement.textContent = `Saved ${samples.length} sample${samples.length === 1 ? "" : "s"}. ${message}`.trim();
}

document.getElementById("clear-button").addEventListener("click", () => drawing.clear());
document.getElementById("save-button").addEventListener("click", () => {
    const writer = writerInput.value.trim();
    if (!drawing.hasInk || !writer) {
        updateStatus("Enter an anonymous writer ID and draw a digit.");
        return;
    }
    const image = drawing.exportPNG();
    if (samples.some((sample) => sample.image === image)) {
        updateStatus("This drawing is already saved. Please draw a new one.");
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
        updateStatus("Save at least one sample first.");
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
    updateStatus("Exported. Keep the file safe.");
});
window.addEventListener("beforeunload", (event) => {
    if (samples.length) {
        event.preventDefault();
        event.returnValue = "";
    }
});
updateStatus();
