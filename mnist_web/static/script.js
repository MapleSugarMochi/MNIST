"use strict";

const clearButton = document.getElementById("clear-button");
const predictButton = document.getElementById("predict-button");
const statusElement = document.getElementById("status");
const resultElement = document.getElementById("result");
const confidenceElement = document.getElementById("confidence");
const topResultsElement = document.getElementById("top-results");

function resetResult(message = "等待输入") {
    statusElement.textContent = message;
    resultElement.textContent = "—";
    confidenceElement.textContent = "";
    topResultsElement.replaceChildren();
}

const drawing = new window.MnistCanvas(document.getElementById("canvas"), () => resetResult());

function showPrediction(data) {
    statusElement.textContent = data.uncertain === true
        ? "模型不太确定，可以尝试重新书写" : "识别完成";
    resultElement.textContent = data.prediction;
    confidenceElement.textContent = `模型分数 ${(data.confidence * 100).toFixed(1)}%`;
    topResultsElement.replaceChildren(...data.top3.map((item) => {
        const row = document.createElement("li");
        row.textContent = `${item.digit} · ${(item.confidence * 100).toFixed(1)}%`;
        return row;
    }));
}

async function predict() {
    if (drawing.busy) return;
    if (!drawing.hasInk) {
        resetResult("请先写一个数字");
        return;
    }
    drawing.setBusy(true);
    const version = drawing.version;
    predictButton.disabled = clearButton.disabled = true;
    resetResult("识别中…");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
        const response = await fetch("/predict", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: drawing.exportPNG() }), signal: controller.signal,
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || "识别服务暂时不可用。");
        if (!Number.isInteger(data.prediction) || data.prediction < 0 || data.prediction > 9
            || !Number.isFinite(data.confidence) || data.confidence < 0 || data.confidence > 1
            || !Array.isArray(data.top3) || data.top3.length !== 3
            || data.top3.some((item) => !Number.isInteger(item.digit)
                || item.digit < 0 || item.digit > 9 || !Number.isFinite(item.confidence)
                || item.confidence < 0 || item.confidence > 1)) {
            throw new Error("识别服务返回了无效结果。");
        }
        if (version === drawing.version) showPrediction(data);
    } catch (error) {
        resetResult(error.name === "AbortError" ? "识别超时，请重试。"
            : error instanceof Error ? error.message : "识别失败，请稍后重试。");
    } finally {
        clearTimeout(timeout);
        drawing.setBusy(false);
        predictButton.disabled = clearButton.disabled = false;
    }
}

clearButton.addEventListener("click", () => drawing.clear());
predictButton.addEventListener("click", predict);
resetResult();
