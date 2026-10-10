const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { randomUUID } = require("node:crypto");

function harness(script = "script.js", fetchImplementation = async () => ({
    ok: true, json: async () => ({ prediction: 7, confidence: 0.98,
        top3: [{ digit: 7, confidence: 0.98 }, { digit: 1, confidence: 0.01 },
            { digit: 9, confidence: 0.01 }], uncertain: null }),
})) {
    const elements = new Map();
    const operations = [];
    const context = new Proxy({}, { get: (_, key) => (...args) => operations.push([key, ...args]),
        set: () => true });
    function element(id) {
        if (!elements.has(id)) elements.set(id, {
            textContent: "", value: "", disabled: false, listeners: {}, children: [],
            addEventListener(name, fn) { this.listeners[name] = fn; },
            setAttribute() {}, replaceChildren(...children) { this.children = children; },
            getContext: () => context, getBoundingClientRect: () => ({ left: 0, top: 0,
                width: 280, height: 280 }),
            setPointerCapture() {}, hasPointerCapture: () => false,
            releasePointerCapture() {}, toDataURL: () => "data:image/png;base64,AAAA",
            click() { this.clicked = true; },
        });
        return elements.get(id);
    }
    const timers = [];
    const sandbox = vm.createContext({
        window: { devicePixelRatio: 10, addEventListener() {} },
        document: { getElementById: element, createElement: element },
        crypto: { randomUUID }, AbortController, Blob,
        URL: { createObjectURL: () => "blob:sample", revokeObjectURL() {} },
        fetch: fetchImplementation, setTimeout: (fn) => { timers.push(fn); return timers.length; },
        clearTimeout: (id) => { timers[id - 1] = null; }, console,
    });
    for (const file of ["drawing.js", script]) {
        vm.runInContext(fs.readFileSync(path.join(__dirname, "../mnist_web/static", file), "utf8"),
            sandbox, { filename: file });
    }
    const run = (code) => vm.runInContext(code, sandbox);
    const pointer = (pointerId = 1, x = 50, y = 50) => ({ pointerId, clientX: x, clientY: y,
        button: 0, isPrimary: true, preventDefault() {} });
    return { element, run, pointer, operations, timers };
}

test("single tap creates visible ink; only one pointer owns the stroke; DPR is bounded", () => {
    const h = harness();
    assert.equal(h.element("canvas").width, 1120);
    h.element("canvas").listeners.pointerdown(h.pointer());
    assert.equal(h.run("drawing.hasInk"), true);
    assert.ok(h.operations.some(([name]) => name === "arc"));
    h.element("canvas").listeners.pointerdown(h.pointer(2));
    h.element("canvas").listeners.pointerup(h.pointer(2));
    assert.equal(h.run("drawing.pointer"), 1);
    h.element("canvas").listeners.lostpointercapture(h.pointer(1));
    assert.equal(h.run("drawing.pointer"), null);
});

test("pending prediction freezes drawing and blocks duplicate requests", async () => {
    let resolve;
    let calls = 0;
    const h = harness("script.js", () => { calls++; return new Promise((r) => { resolve = r; }); });
    h.element("canvas").listeners.pointerdown(h.pointer());
    const version = h.run("drawing.version");
    const pending = h.run("predict()");
    assert.equal(h.element("predict-button").disabled, true);
    h.element("canvas").listeners.pointerdown(h.pointer());
    h.element("canvas").listeners.pointermove(h.pointer());
    assert.equal(h.run("drawing.version"), version);
    await h.run("predict()");
    assert.equal(calls, 1);
    resolve({ ok: true, json: async () => ({ prediction: 7, confidence: 0.99,
        top3: [{ digit: 7, confidence: 0.99 }, { digit: 1, confidence: 0.01 },
            { digit: 9, confidence: 0 }], uncertain: false }) });
    await pending;
    assert.equal(h.element("result").textContent, 7);
    assert.equal(h.element("predict-button").disabled, false);
    assert.equal(h.run("drawing.busy"), false);
    h.element("canvas").listeners.pointerdown(h.pointer());
    assert.equal(h.element("result").textContent, "—");
});

test("timeout aborts the request and restores controls", async () => {
    const h = harness("script.js", (_, options) => new Promise((resolve, reject) => {
        options.signal.addEventListener("abort", () => {
            const error = new Error("aborted"); error.name = "AbortError"; reject(error);
        });
    }));
    h.element("canvas").listeners.pointerdown(h.pointer());
    const pending = h.run("predict()");
    h.timers[0]();
    await pending;
    assert.equal(h.element("status").textContent, "Recognition timed out. Please try again.");
    assert.equal(h.run("drawing.busy"), false);
    assert.equal(h.element("clear-button").disabled, false);
});

test("unavailable service surfaces JSON error and stale responses are discarded", async () => {
    const h = harness("script.js", async () => ({ ok: false,
        json: async () => ({ error: "Model is unavailable." }) }));
    h.element("canvas").listeners.pointerdown(h.pointer());
    await h.run("predict()");
    assert.equal(h.element("status").textContent, "Model is unavailable.");
    const stale = harness("script.js", async () => {
        stale.run("drawing.version += 1");
        return { ok: true, json: async () => ({ prediction: 7, confidence: 0.98,
            top3: [{ digit: 7, confidence: 0.98 }, { digit: 1, confidence: 0.01 },
                { digit: 9, confidence: 0.01 }] }) };
    });
    stale.element("canvas").listeners.pointerdown(stale.pointer());
    await stale.run("predict()");
    assert.equal(stale.element("result").textContent, "—");
});

test("collection saves anonymous label, clears drawing, and supports undo and export", () => {
    const h = harness("collect.js");
    h.element("digit-label").value = "7";
    h.element("canvas").listeners.pointerdown(h.pointer());
    h.element("save-button").listeners.click();
    assert.equal(h.run("samples.length"), 1);
    assert.equal(h.run("samples[0].label"), 7);
    assert.ok(h.run("samples[0].writer_id").startsWith("writer-"));
    assert.equal(h.run("drawing.hasInk"), false);
    h.element("export-button").listeners.click();
    assert.ok(h.element("a").download.endsWith(".json"));
    h.element("undo-button").listeners.click();
    assert.equal(h.run("samples.length"), 0);
});

test("browser preprocessing reproduces the Python pipeline pixel for pixel", () => {
    const zlib = require("node:zlib");
    const preprocessing = require("../demo/preprocessing.js");
    const fixture = JSON.parse(fs.readFileSync(
        path.join(__dirname, "fixtures", "preprocessing_parity.json"), "utf8"));
    assert.equal(fixture.preprocessing_version, preprocessing.PREPROCESSING_VERSION);
    assert.ok(fixture.samples.length >= 300);
    for (const [index, sample] of fixture.samples.entries()) {
        const gray = new Uint8Array(zlib.inflateSync(Buffer.from(sample.gray, "base64")));
        const expected = new Uint8Array(zlib.inflateSync(Buffer.from(sample.expected, "base64")));
        const actual = preprocessing.preprocessGray(gray, sample.width, sample.height);
        assert.deepEqual(actual, expected, `sample ${index} (${sample.source})`);
    }
});

test("browser preprocessing rejects empty drawings and rounds like Python", () => {
    const preprocessing = require("../demo/preprocessing.js");
    assert.throws(() => preprocessing.preprocessGray(new Uint8Array(280 * 280), 280, 280),
        preprocessing.EmptyDrawingError);
    assert.deepEqual([0.5, 1.5, 2.5, -0.5, 2.6].map(preprocessing.roundHalfEven), [0, 2, 2, 0, 3]);
});
