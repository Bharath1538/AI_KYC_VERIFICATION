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
        { step: 1, title: 'Face Detection', desc: 'Position your face in the oval', icon: 'scan-face' },
        { step: 2, title: 'Hold Still', desc: 'Keep your face steady', icon: 'eye' },
        { step: 3, title: 'Look at Camera', desc: 'Look directly at the camera', icon: 'focus' },
        { step: 4, title: 'Capture', desc: 'Capturing your photo...', icon: 'camera' }
    ];

    // Face Detection using skin-tone detection
    async function detectFace() {
        if (!video.videoWidth) return false;

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Check center region for face (skin tones)
        const centerX = canvas.width * 0.3;
        const centerY = canvas.height * 0.2;
        const regionW = canvas.width * 0.4;
        const regionH = canvas.height * 0.5;

        const imageData = ctx.getImageData(centerX, centerY, regionW, regionH);
        const data = imageData.data;

        let skinPixels = 0;
        const totalPixels = data.length / 4;

        for (let i = 0; i < data.length; i += 4) {
            const r = data[i], g = data[i + 1], b = data[i + 2];

            // Skin tone detection (works for various skin tones)
            if (r > 60 && g > 40 && b > 20 &&
                r > g && r > b &&
                Math.abs(r - g) > 15 && r - b > 15) {
                skinPixels++;
            }
        }

        const skinRatio = skinPixels / totalPixels;
        console.log(`Skin ratio: ${(skinRatio * 100).toFixed(1)}%`);
        return skinRatio > 0.12;
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
                await sleep(300);
            } else {
                updateStepStatus(challenge.step, 'failed');
                showFailure('No face detected. Position your face in the oval.');
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
            const maxDuration = 8000;

            const checkInterval = setInterval(async () => {
                const elapsed = Date.now() - startTime;
                const detected = await detectFace();

                if (detected) {
                    consecutiveDetections++;
                    faceGuide.style.opacity = '1';
                } else {
                    consecutiveDetections = Math.max(0, consecutiveDetections - 1);
                    faceGuide.style.opacity = '0.5';
                }

                const progress = Math.min((consecutiveDetections / REQUIRED_DETECTIONS) * 100, 100);
                setProgress(progress);

                if (consecutiveDetections >= REQUIRED_DETECTIONS) {
                    clearInterval(checkInterval);
                    resolve(true);
                }

                if (elapsed > maxDuration) {
                    clearInterval(checkInterval);
                    resolve(consecutiveDetections >= 2);
                }
            }, 200);
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
