"use strict";

const clearButton = document.getElementById("clear-button");
const predictButton = document.getElementById("predict-button");
const statusElement = document.getElementById("status");
const resultElement = document.getElementById("result");
const confidenceElement = document.getElementById("confidence");
const topResultsElement = document.getElementById("top-results");

function resetResult(message = "Waiting for input") {
    statusElement.textContent = message;
    resultElement.textContent = "—";
    confidenceElement.textContent = "";
    topResultsElement.replaceChildren();
}

const drawing = new window.MnistCanvas(document.getElementById("canvas"), () => resetResult());

function showPrediction(data) {
    statusElement.textContent = data.uncertain === true
        ? "The model is unsure. Try drawing again." : "Done";
    resultElement.textContent = data.prediction;
    confidenceElement.textContent = `Model score ${(data.confidence * 100).toFixed(1)}%`;
    topResultsElement.replaceChildren(...data.top3.map((item) => {
        const row = document.createElement("li");
        row.textContent = `${item.digit} · ${(item.confidence * 100).toFixed(1)}%`;
        return row;
    }));
}

async function predict() {
    if (drawing.busy) return;
    if (!drawing.hasInk) {
        resetResult("Draw a digit first");
        return;
    }
    drawing.setBusy(true);
    const version = drawing.version;
    predictButton.disabled = clearButton.disabled = true;
    resetResult("Recognizing…");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
        const response = await fetch("/predict", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: drawing.exportPNG() }), signal: controller.signal,
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || "Recognition service is unavailable.");
        if (!Number.isInteger(data.prediction) || data.prediction < 0 || data.prediction > 9
            || !Number.isFinite(data.confidence) || data.confidence < 0 || data.confidence > 1
            || !Array.isArray(data.top3) || data.top3.length !== 3
            || data.top3.some((item) => !Number.isInteger(item.digit)
                || item.digit < 0 || item.digit > 9 || !Number.isFinite(item.confidence)
                || item.confidence < 0 || item.confidence > 1)) {
            throw new Error("The recognition service returned an invalid result.");
        }
        if (version === drawing.version) showPrediction(data);
    } catch (error) {
        resetResult(error.name === "AbortError" ? "Recognition timed out. Please try again."
            : error instanceof Error ? error.message : "Recognition failed. Please try again later.");
    } finally {
        clearTimeout(timeout);
        drawing.setBusy(false);
        predictButton.disabled = clearButton.disabled = false;
    }
}

clearButton.addEventListener("click", () => drawing.clear());
predictButton.addEventListener("click", predict);
resetResult();
