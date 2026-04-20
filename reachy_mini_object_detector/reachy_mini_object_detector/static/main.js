const videoFeed = document.getElementById("video-feed");
const overlay = document.getElementById("video-overlay");
const fpsBadge = document.getElementById("fps-badge");
const detectionList = document.getElementById("detection-list");
const trackingCheckbox = document.getElementById("tracking-checkbox");

// Hide overlay once video loads
videoFeed.addEventListener("load", () => {
    overlay.classList.add("hidden");
});

// Poll detections every 500ms
async function fetchDetections() {
    try {
        const resp = await fetch("/detections");
        const data = await resp.json();

        // Update FPS badge
        fpsBadge.textContent = `${data.fps} FPS`;

        // Update tracking toggle
        trackingCheckbox.checked = data.tracking_enabled;

        // Update detection list
        if (data.detections.length === 0) {
            detectionList.innerHTML = '<li class="placeholder">No objects detected</li>';
        } else {
            detectionList.innerHTML = data.detections
                .map(d => `<li><span class="label">${d.label}</span><span class="score">${(d.score * 100).toFixed(0)}%</span></li>`)
                .join("");
        }
    } catch (e) {
        fpsBadge.textContent = "-- FPS";
    }
}

// Tracking toggle
trackingCheckbox.addEventListener("change", async (e) => {
    try {
        await fetch("/tracking", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled: e.target.checked }),
        });
    } catch (err) {
        console.error("Error toggling tracking:", err);
    }
});

// Start polling
setInterval(fetchDetections, 500);
fetchDetections();