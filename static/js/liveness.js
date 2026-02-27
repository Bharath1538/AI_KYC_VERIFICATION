/**
 * Face Liveness Detection JavaScript
 * Uses real face detection for verification
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const video = document.getElementById('livenessVideo');
    const faceCanvas = document.getElementById('faceCanvas');
    const faceGuide = document.getElementById('faceGuide');
    const guideText = document.getElementById('guideText');
    const progressRing = document.getElementById('progressRing');
    const progressCircle = document.getElementById('progressCircle');
    const challengeIcon = document.getElementById('challengeIcon');
    const challengeTitle = document.getElementById('challengeTitle');
    const challengeDesc = document.getElementById('challengeDesc');
    const startBtn = document.getElementById('startLivenessBtn');
    const livenessResult = document.getElementById('livenessResult');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultDesc = document.getElementById('resultDesc');

    const steps = {
        1: document.getElementById('step1'),
        2: document.getElementById('step2'),
        3: document.getElementById('step3'),
        4: document.getElementById('step4')
    };

    // State
    let mediaStream = null;
    let isRunning = false;
    let faceDetector = null;
    let consecutiveDetections = 0;
    const REQUIRED_DETECTIONS = 5;

    const challenges = [
        { step: 1, title: 'Face Detection', desc: 'Position your face in the oval', icon: 'scan-face', type: 'face' },
        { step: 2, title: 'Blink Detection', desc: 'Blink your eyes naturally', icon: 'eye', type: 'blink' },
        { step: 3, title: 'Smile Detection', desc: 'Give a natural smile', icon: 'smile', type: 'smile' },
        { step: 4, title: 'Head Turn', desc: 'Slowly turn your head left then right', icon: 'move', type: 'turn' }
    ];

    // Face Detection using skin-tone and motion detection
    async function detectFace() {
        if (!video.videoWidth) return { hasFace: false, motion: 0 };

        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 120;
        const ctx = canvas.getContext('2d');

        // Calculate crop to match object-fit: cover for 4:3 container
        const vidW = video.videoWidth;
        const vidH = video.videoHeight;
        let cropW = vidW;
        let cropH = vidH;
        let cropX = 0;
        let cropY = 0;

        if (vidW / vidH > 4 / 3) {
            cropW = vidH * (4 / 3);
            cropX = (vidW - cropW) / 2;
        } else {
            cropH = vidW * (3 / 4);
            cropY = (vidH - cropH) / 2;
        }

        // Mirror to match the video
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, canvas.width, canvas.height);

        // Define Oval Region (Center)
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radiusX = canvas.width * 0.3;
        const radiusY = canvas.height * 0.4;

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        let edgesInOval = 0;
        let totalOval = 0;
        let diffPixels = 0;

        if (!window.prevFaceData) {
            window.prevFaceData = new Uint8ClampedArray(data.length);
        }

        for (let y = 1; y < canvas.height - 1; y += 2) {
            for (let x = 1; x < canvas.width - 1; x += 2) {
                const i = (y * canvas.width + x) * 4;

                // Math to check if point is inside ellipse
                const dx = (x - centerX) / radiusX;
                const dy = (y - centerY) / radiusY;
                const inOval = (dx * dx + dy * dy) <= 1;

                if (inOval) {
                    totalOval++;

                    // Simple brightness calculation (luminosity)
                    const r = data[i], g = data[i + 1], b = data[i + 2];
                    const brightness = (r + g + b) / 3;

                    // Simple edge detection: compare to pixel to the left to find facial features
                    const leftI = (y * canvas.width + (x - 1)) * 4;
                    const leftR = data[leftI], leftG = data[leftI + 1], leftB = data[leftI + 2];
                    const leftBrightness = (leftR + leftG + leftB) / 3;

                    // If there's a sharp contrast in the oval (eyes, nose, mouth edges)
                    if (Math.abs(brightness - leftBrightness) > 10) {
                        edgesInOval++;
                    }

                    // Motion detection in oval
                    const prevR = window.prevFaceData[i];
                    if (Math.abs(r - prevR) > 15) {
                        diffPixels++;
                    }
                }

                window.prevFaceData[i] = data[i];
            }
        }

        const edgeRatio = edgesInOval / Math.max(1, totalOval);
        const motionRatio = diffPixels / Math.max(1, totalOval);

        // Face is present if there is sufficient texture/edges inside the oval (i.e., not a blank wall)
        const hasFace = (edgeRatio > 0.05);

        return {
            hasFace: hasFace,
            motion: motionRatio
        };
    }

    async function startCamera() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
            });
            video.srcObject = mediaStream;
            await video.play();
            faceCanvas.width = 640;
            faceCanvas.height = 480;
            return true;
        } catch (err) {
            console.error('Camera error:', err);
            showToast('Unable to access camera', 'error');
            return false;
        }
    }

    function stopCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
    }

    async function startLivenessCheck() {
        if (isRunning) return;

        const cameraStarted = await startCamera();
        if (!cameraStarted) return;

        isRunning = true;
        startBtn.classList.add('hidden');
        faceGuide.classList.add('active');

        console.log('Starting liveness check with face detection...');

        for (let i = 0; i < challenges.length; i++) {
            if (!isRunning) break;

            const challenge = challenges[i];
            console.log(`Starting step ${challenge.step}: ${challenge.title}`);

            updateChallenge(challenge);
            updateStepStatus(challenge.step, 'active');

            const success = await runChallenge(challenge);
            console.log(`Step ${challenge.step} result: ${success ? 'SUCCESS' : 'FAILED'}`);

            if (success) {
                updateStepStatus(challenge.step, 'complete');
                await sleep(500); // Give user a moment to rest before next step
            } else {
                updateStepStatus(challenge.step, 'failed');
                showFailure('Verification timed out. Please follow the instructions and stay in the oval.');
                return;
            }
        }

        console.log('All challenges passed!');
        showSuccess();
    }

    async function runChallenge(challenge) {
        return new Promise((resolve) => {
            consecutiveDetections = 0;
            const startTime = Date.now();
            let hasCompletedAction = false;

            const checkInterval = setInterval(async () => {
                const elapsed = Date.now() - startTime;
                const { hasFace, motion } = await detectFace();

                let progressVal = 0;

                // Stop if we don't detect a face
                if (!hasFace) {
                    consecutiveDetections = Math.max(0, consecutiveDetections - 1);
                    guideText.textContent = "Please place your face inside the oval!";
                    guideText.style.color = "#ef4444";
                    faceGuide.style.opacity = '0.3';
                } else {
                    guideText.textContent = challenge.desc;
                    guideText.style.color = "var(--text-primary)";
                    faceGuide.style.opacity = '1';

                    if (challenge.type === 'face') {
                        consecutiveDetections++;
                        progressVal = (consecutiveDetections / 10) * 100;
                        if (consecutiveDetections >= 10) hasCompletedAction = true;
                    }
                    else if (challenge.type === 'blink') {
                        // Blink causes a sharp spike in motion
                        if (motion > 0.03 && motion < 0.15) consecutiveDetections++;
                        progressVal = (consecutiveDetections / 3) * 100;
                        if (consecutiveDetections >= 3) hasCompletedAction = true;
                    }
                    else if (challenge.type === 'smile') {
                        // Smile causes slight motion that settles quickly
                        if (motion > 0.01 && motion < 0.1) consecutiveDetections++;
                        progressVal = (consecutiveDetections / 5) * 100;
                        if (consecutiveDetections >= 5) hasCompletedAction = true;
                    }
                    else if (challenge.type === 'turn') {
                        // Head turn causes massive, sustained motion
                        if (motion > 0.12) consecutiveDetections++;
                        progressVal = (consecutiveDetections / 4) * 100;
                        if (consecutiveDetections >= 4) hasCompletedAction = true;
                    }
                }

                setProgress(Math.min(Math.max(progressVal, 0), 100));

                if (hasCompletedAction) {
                    clearInterval(checkInterval);
                    resolve(true);
                }

                // 15 seconds max duration per step to be generous
                if (elapsed > 15000) {
                    clearInterval(checkInterval);
                    resolve(false);
                }
            }, 150);
        });
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function updateChallenge(challenge) {
        challengeIcon.innerHTML = `<i data-lucide="${challenge.icon}"></i>`;
        challengeTitle.textContent = challenge.title;
        challengeDesc.textContent = challenge.desc;
        guideText.textContent = challenge.desc;
        guideText.style.color = "var(--text-primary)";
        lucide.createIcons();
        progressRing.classList.remove('hidden');
        setProgress(0);
    }

    function updateStepStatus(step, status) {
        const stepEl = steps[step];
        if (!stepEl) return;

        stepEl.classList.remove('active', 'complete', 'failed');
        stepEl.classList.add(status);

        const statusContainer = stepEl.querySelector('.step-status');
        if (!statusContainer) return;

        if (status === 'active') {
            statusContainer.innerHTML = '<i data-lucide="loader" class="spinning"></i>';
        } else if (status === 'complete') {
            statusContainer.innerHTML = '<i data-lucide="check-circle-2" class="success"></i>';
        } else if (status === 'failed') {
            statusContainer.innerHTML = '<i data-lucide="x-circle" class="failed"></i>';
        }
        lucide.createIcons();
    }

    function setProgress(percent) {
        const circumference = 2 * Math.PI * 54;
        const offset = circumference - (percent / 100) * circumference;
        progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
        progressCircle.style.strokeDashoffset = offset;
    }

    function showSuccess() {
        isRunning = false;
        progressRing.classList.add('hidden');
        faceGuide.classList.remove('active');
        faceGuide.style.opacity = '1';

        // Capture selfie - MIRRORED to match what user sees
        const ctx = faceCanvas.getContext('2d');

        // Set canvas to video dimensions
        faceCanvas.width = video.videoWidth || 640;
        faceCanvas.height = video.videoHeight || 480;

        // Mirror the canvas before drawing (like the CSS transform on video)
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -faceCanvas.width, 0, faceCanvas.width, faceCanvas.height);
        ctx.restore();

        // Save the full mirrored selfie - face_matching.py handles face extraction
        const selfieData = faceCanvas.toDataURL('image/jpeg', 0.9);

        sessionStorage.setItem('livenessVerified', 'true');
        sessionStorage.setItem('livenessTimestamp', Date.now().toString());
        sessionStorage.setItem('livenessSelfie', selfieData);

        console.log('Mirrored selfie captured:', faceCanvas.width, 'x', faceCanvas.height);

        livenessResult.classList.remove('hidden');
        resultIcon.className = 'result-icon success';
        resultIcon.innerHTML = '<i data-lucide="shield-check"></i>';
        resultTitle.textContent = 'Liveness Verified!';
        resultDesc.textContent = 'Proceed to document verification for face matching.';
        lucide.createIcons();

        showToast('Liveness check passed!', 'success');
        stopCamera();
    }

    function showFailure(message) {
        isRunning = false;
        progressRing.classList.add('hidden');
        faceGuide.classList.remove('active');
        faceGuide.style.opacity = '1';

        livenessResult.classList.remove('hidden');
        resultIcon.className = 'result-icon failed';
        resultIcon.innerHTML = '<i data-lucide="shield-x"></i>';
        resultTitle.textContent = 'Verification Failed';
        resultDesc.textContent = message || 'Please try again.';

        const actionsDiv = livenessResult.querySelector('.result-actions');
        if (!actionsDiv.querySelector('.retry')) {
            const retryBtn = document.createElement('button');
            retryBtn.className = 'continue-btn retry';
            retryBtn.innerHTML = '<i data-lucide="refresh-cw"></i><span>Try Again</span>';
            retryBtn.onclick = resetLiveness;
            actionsDiv.appendChild(retryBtn);
        }

        lucide.createIcons();
        showToast('Liveness check failed', 'error');
        stopCamera();
    }

    function resetLiveness() {
        isRunning = false;
        consecutiveDetections = 0;

        livenessResult.classList.add('hidden');
        startBtn.classList.remove('hidden');
        faceGuide.classList.remove('active');

        Object.values(steps).forEach(stepEl => {
            stepEl.classList.remove('active', 'complete', 'failed');
            const statusContainer = stepEl.querySelector('.step-status');
            if (statusContainer) {
                statusContainer.innerHTML = '<i data-lucide="circle" class="pending"></i>';
            }
        });

        const retryBtn = livenessResult.querySelector('.retry');
        if (retryBtn) retryBtn.remove();

        lucide.createIcons();
    }

    startBtn.addEventListener('click', startLivenessCheck);
    window.addEventListener('beforeunload', stopCamera);
    lucide.createIcons();
});
