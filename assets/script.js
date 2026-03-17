const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const gestureName = document.getElementById('gesture-name');
const confidenceText = document.getElementById('confidence-text');
const gestureMatchImage = document.getElementById('gesture-match-image');
const fpsCounter = document.getElementById('fps-counter');

// Sliders and Toggles
const qualitySlider = document.getElementById('quality-slider');
const qualityValue = document.getElementById('quality-value');
const fpsLimitSlider = document.getElementById('fps-limit-slider');
const fpsLimitValue = document.getElementById('fps-limit-value');
const landmarksToggle = document.getElementById('landmarks-toggle');

let quality = 0.5; // Lowered default quality for better performance
let fpsLimit = 30;
let isWaitingForResponse = false; // Sync flag to prevent frame queuing

// Setup Camera
navigator.mediaDevices.getUserMedia({ 
    video: { width: { ideal: 640 }, height: { ideal: 480 } } // Request lower resolution
}).then(stream => {
    video.srcObject = stream;
});

// Event Listeners
qualitySlider?.addEventListener('input', (e) => {
    quality = parseFloat(e.target.value);
    qualityValue.textContent = Math.round(quality * 100) + '%';
});

fpsLimitSlider?.addEventListener('input', (e) => {
    fpsLimit = parseInt(e.target.value);
    fpsLimitValue.textContent = fpsLimit + ' fps';
});

// WebSocket Connection
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

video.addEventListener('play', () => {
    const offCanvas = document.createElement('canvas');
    
    function sendFrame() {
        if (video.paused || video.ended) return;
        
        // Optimization: Only send if the socket is open and we aren't already waiting for a frame
        if (ws.readyState === WebSocket.OPEN && !isWaitingForResponse) {
            // Downscale for processing - 640x360 is plenty for gesture recognition
            const targetWidth = 640;
            const targetHeight = (video.videoHeight / video.videoWidth) * targetWidth;
            
            offCanvas.width = targetWidth;
            offCanvas.height = targetHeight;
            
            const octx = offCanvas.getContext('2d');
            octx.drawImage(video, 0, 0, targetWidth, targetHeight);
            
            const dataURL = offCanvas.toDataURL('image/jpeg', quality);
            const drawLandmarks = landmarksToggle?.checked ?? true;
            
            isWaitingForResponse = true; // Set flag
            ws.send(JSON.stringify({
                image: dataURL,
                draw_landmarks: drawLandmarks
            }));
        }
        
        // Dynamic interval based on FPS limit slider
        const interval = 1000 / fpsLimit;
        setTimeout(sendFrame, interval);
    }
    sendFrame();
});

ws.onmessage = function(event) {
    isWaitingForResponse = false; // Reset flag so next frame can be sent
    
    try {
        const data = JSON.parse(event.data);
        
        // Update FPS Badge
        if (fpsCounter && data.fps !== undefined) {
            fpsCounter.textContent = `${data.fps} fps`;
        }
        
        // Update Canvas
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = data.image;
        
        // Update Gesture Display
        if (data.labels && data.labels.length > 0) {
            // Get the gesture with highest confidence
            const topLabel = data.labels.reduce((prev, current) => 
                (prev.confidence > current.confidence) ? prev : current
            );
            
            gestureName.textContent = topLabel.gesture;
            confidenceText.textContent = `${(topLabel.confidence * 100).toFixed(0)}% confiança`;
            
            // Show match image if applicable
            if (data.image_to_show) {
                const newSrc = '/assets/images/gestures/' + data.image_to_show;
                if (!gestureMatchImage.src.endsWith(newSrc)) {
                    gestureMatchImage.src = newSrc;
                }
                gestureMatchImage.style.display = 'block';
            } else {
                gestureMatchImage.style.display = 'none';
            }
        } else {
            gestureName.textContent = "Nenhum Gesto";
            confidenceText.textContent = "0% confiança";
            gestureMatchImage.style.display = 'none';
        }
    } catch (e) {
        console.error("Error processing WS message:", e);
    }
};

ws.onerror = () => { isWaitingForResponse = false; };
ws.onclose = () => { isWaitingForResponse = false; };
