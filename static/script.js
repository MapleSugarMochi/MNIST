const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let painting = false;

ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "white";

canvas.addEventListener("mousedown", () => painting = true);
canvas.addEventListener("mouseup", () => painting = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!painting) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

clearCanvas();

async function predict() {
    const dataURL = canvas.toDataURL("image/png");

    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ image: dataURL })
    });

    const result = await response.json();
    document.getElementById("result").innerText = result.prediction;
}
