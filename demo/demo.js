"use strict";

// Static demo: the same canvas and preprocessing as the Flask app, with the
// CNN running in the browser through ONNX Runtime Web (WebAssembly backend).
(function () {
    const P = window.MnistPreprocessing;
    const statusElement = document.getElementById("status");
    const resultElement = document.getElementById("result");
    const confidenceElement = document.getElementById("confidence");
    const barsElement = document.getElementById("bars");
    const modelIdElement = document.getElementById("model-id");
    const predictButton = document.getElementById("predict-button");
    const clearButton = document.getElementById("clear-button");
    const liveToggle = document.getElementById("live-toggle");
    const preview = document.getElementById("preview").getContext("2d");
    const canvas = document.getElementById("canvas");

    let session = null;
    let inputName = "image";
    let running = false;
    let pending = false;

    const bars = Array.from({ length: 10 }, (_, digit) => {
        const row = document.createElement("li");
        const label = document.createElement("span");
        const track = document.createElement("span");
        const fill = document.createElement("span");
        const value = document.createElement("span");
        label.textContent = String(digit);
        track.className = "track";
        fill.className = "fill";
        fill.style.width = "0%";
        value.className = "value";
        value.textContent = "—";
        track.append(fill);
        row.append(label, track, value);
        barsElement.append(row);
        return { row, fill, value };
    });

    function resetOutput(message) {
        if (message !== undefined) statusElement.textContent = message;
        resultElement.textContent = "—";
        confidenceElement.textContent = "";
        for (const bar of bars) {
            bar.row.classList.remove("top");
            bar.fill.style.width = "0%";
            bar.value.textContent = "—";
        }
        preview.fillStyle = "black";
        preview.fillRect(0, 0, P.IMAGE_SIZE, P.IMAGE_SIZE);
    }

    const drawing = new window.MnistCanvas(canvas, () => {
        if (!liveToggle.checked && session) statusElement.textContent = "Ready";
    });

    function drawPreview(glyph) {
        const image = preview.createImageData(P.IMAGE_SIZE, P.IMAGE_SIZE);
        for (let i = 0; i < glyph.length; i++) {
            image.data.set([glyph[i], glyph[i], glyph[i], 255], i * 4);
        }
        preview.putImageData(image, 0, 0);
    }

    function validProbabilities(values) {
        if (values.length !== 10) return false;
        let total = 0;
        for (const value of values) {
            if (!Number.isFinite(value) || value < 0 || value > 1) return false;
            total += value;
        }
        return Math.abs(total - 1) < 1e-4;
    }

    function render(probabilities, elapsed) {
        let top = 0;
        for (let digit = 1; digit < 10; digit++) {
            if (probabilities[digit] > probabilities[top]) top = digit;
        }
        resultElement.textContent = String(top);
        confidenceElement.textContent = `Model score ${(probabilities[top] * 100).toFixed(1)}%`;
        bars.forEach((bar, digit) => {
            const percent = probabilities[digit] * 100;
            bar.row.classList.toggle("top", digit === top);
            bar.fill.style.width = `${percent.toFixed(2)}%`;
            bar.value.textContent = `${percent.toFixed(1)}%`;
        });
        statusElement.textContent = `Inference ran in your browser in ${elapsed.toFixed(1)} ms`;
    }

    async function predict() {
        if (!session) return;
        if (running) {
            pending = true;
            return;
        }
        if (!drawing.hasInk) {
            resetOutput("Draw a digit first");
            return;
        }
        running = true;
        try {
            const started = performance.now();
            const image = drawing.exportImageData();
            const gray = P.rgbaToGray(image.data, image.width, image.height);
            const glyph = P.preprocessGray(gray, image.width, image.height);
            drawPreview(glyph);
            const tensor = new ort.Tensor("float32", P.toModelInput(glyph), [1, 28, 28, 1]);
            const outputs = await session.run({ [inputName]: tensor });
            const probabilities = Array.from(outputs[session.outputNames[0]].data);
            if (!validProbabilities(probabilities)) throw new Error("The model returned invalid scores.");
            render(probabilities, performance.now() - started);
        } catch (error) {
            resetOutput(error instanceof P.EmptyDrawingError ? error.message
                : `Recognition failed: ${error.message}`);
        } finally {
            running = false;
            if (pending) {
                pending = false;
                predict();
            }
        }
    }

    function strokeEnded() {
        if (liveToggle.checked) predict();
    }

    for (const name of ["pointerup", "pointercancel"]) canvas.addEventListener(name, strokeEnded);
    predictButton.addEventListener("click", predict);
    clearButton.addEventListener("click", () => {
        drawing.clear();
        resetOutput(session ? "Ready" : statusElement.textContent);
    });

    async function load() {
        resetOutput("Loading model…");
        try {
            const info = await (await fetch("model.json")).json();
            if (info.preprocessing_version !== P.PREPROCESSING_VERSION) {
                throw new Error("model and preprocessing versions do not match");
            }
            ort.env.wasm.wasmPaths = new URL("vendor/", document.baseURI).href;
            ort.env.wasm.numThreads = 1;
            session = await ort.InferenceSession.create("model.onnx", {
                executionProviders: ["wasm"],
            });
            inputName = session.inputNames[0];
            modelIdElement.textContent = `Model ${info.model_id} · SHA-256 ${info.source_sha256.slice(0, 12)}…`;
            predictButton.disabled = false;
            statusElement.textContent = "Ready";
            if (drawing.hasInk && liveToggle.checked) predict();
        } catch (error) {
            statusElement.textContent = `The model could not be loaded: ${error.message}`;
        }
    }

    load();
})();
