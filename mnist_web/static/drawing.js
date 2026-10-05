"use strict";

// Shared canvas implementation for inference and labeled sample collection.
class MnistCanvas {
    constructor(canvas, onChange = () => {}) {
        this.canvas = canvas;
        this.context = canvas.getContext("2d");
        this.size = 280;
        this.scale = Math.min(4, Math.max(1, window.devicePixelRatio || 1));
        this.pointer = null;
        this.hasInk = false;
        this.busy = false;
        this.version = 0;
        this.onChange = onChange;
        canvas.width = this.size * this.scale;
        canvas.height = this.size * this.scale;
        this.context.setTransform(this.scale, 0, 0, this.scale, 0, 0);
        this.context.lineWidth = 20;
        this.context.lineCap = "round";
        this.context.lineJoin = "round";
        this.context.strokeStyle = "white";
        canvas.addEventListener("pointerdown", (event) => this.start(event));
        canvas.addEventListener("pointermove", (event) => this.move(event));
        for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
            canvas.addEventListener(name, (event) => this.stop(event));
        }
        this.clear();
    }

    point(event) {
        const bounds = this.canvas.getBoundingClientRect();
        return {
            x: (event.clientX - bounds.left) * this.size / bounds.width,
            y: (event.clientY - bounds.top) * this.size / bounds.height,
        };
    }

    changed() {
        this.version += 1;
        this.onChange();
    }

    start(event) {
        if (this.busy || this.pointer !== null || event.button !== 0 || event.isPrimary === false) return;
        event.preventDefault();
        this.pointer = event.pointerId;
        this.previous = this.point(event);
        this.canvas.setPointerCapture(event.pointerId);
        // Explicit circle ensures a single tap produces visible ink.
        this.context.beginPath();
        this.context.fillStyle = "white";
        this.context.arc(this.previous.x, this.previous.y, 10, 0, Math.PI * 2);
        this.context.fill();
        this.hasInk = true;
        this.changed();
    }

    move(event) {
        if (this.busy || event.pointerId !== this.pointer) return;
        const point = this.point(event);
        this.context.beginPath();
        this.context.moveTo(this.previous.x, this.previous.y);
        this.context.lineTo(point.x, point.y);
        this.context.stroke();
        this.previous = point;
        this.changed();
    }

    stop(event) {
        if (event.pointerId !== this.pointer) return;
        const pointer = this.pointer;
        this.pointer = null;
        if (this.canvas.hasPointerCapture(pointer)) this.canvas.releasePointerCapture(pointer);
    }

    setBusy(busy) {
        if (busy && this.pointer !== null) this.stop({ pointerId: this.pointer });
        this.busy = busy;
        this.canvas.setAttribute("aria-busy", String(busy));
    }

    clear() {
        if (this.busy) return;
        if (this.pointer !== null) this.stop({ pointerId: this.pointer });
        this.context.fillStyle = "black";
        this.context.fillRect(0, 0, this.size, this.size);
        this.context.beginPath();
        this.hasInk = false;
        this.changed();
    }

    exportPNG() {
        const exportCanvas = document.createElement("canvas");
        exportCanvas.width = exportCanvas.height = this.size;
        exportCanvas.getContext("2d").drawImage(this.canvas, 0, 0, this.size, this.size);
        return exportCanvas.toDataURL("image/png");
    }
}

window.MnistCanvas = MnistCanvas;
