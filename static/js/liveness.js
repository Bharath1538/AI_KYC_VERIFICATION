/**
 * Face Liveness Detection JavaScript
 * Restored to Original UI but with Advanced MediaPipe Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const { FaceLandmarker, FilesetResolver } = mediapipe.tasks.vision;

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
    let faceLandmarker = null;
    let lastVideoTime = -1;
    let headTurnedState = { left: false, right: false };

    // Challenges Configuration
    const allChallenges = [
        { id: 'blink', title: 'Blink Detection', desc: 'Blink your eyes naturally', icon: 'eye', target: 2 },
        { id: 'smile', title: 'Smile Detection', desc: 'Give a natural smile', icon: 'smile', target: 1 },
        { id: 'turn', title: 'Head Turn', desc: 'Slowly turn your head left then right', icon: 'move', target: 2 }
    ];

    let sessionChallenges = [];

    async function initMediaPipe() {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
        );
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1
        });
        console.log("MediaPipe Loaded");
    }

    async function startCamera() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            video.srcObject = mediaStream;
            await video.play();
            return true;
        } catch (err) {
            console.error('Camera error:', err);
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
        
        startBtn.disabled = true;
        startBtn.innerHTML = '<i data-lucide="loader" class="spinning"></i><span>Initializing...</span>';
        lucide.createIcons();

        if (!faceLandmarker) await initMediaPipe();
        const cameraStarted = await startCamera();
        if (!cameraStarted) {
            startBtn.disabled = false;
            startBtn.innerHTML = '<i data-lucide="play"></i><span>Start Liveness Check</span>';
            lucide.createIcons();
            return;
        }

        isRunning = true;
        startBtn.classList.add('hidden');
        faceGuide.classList.add('active');

        // Randomized sequence
        sessionChallenges = [
            { id: 'face', title: 'Face Detection', desc: 'Position your face in the oval', icon: 'scan-face', target: 1 },
            ...shuffleArray([...allChallenges])
        ];

        // Update UI Step mapping dynamically
        updateStepListUI();

        for (let i = 0; i < sessionChallenges.length; i++) {
            if (!isRunning) break;
            const challenge = sessionChallenges[i];
            const stepNum = i + 1;
            
            updateChallengeUI(challenge);
            updateStepStatus(stepNum, 'active');

            const success = await runChallenge(challenge);
            if (success) {
                updateStepStatus(stepNum, 'complete');
                await sleep(500);
            } else {
                updateStepStatus(stepNum, 'failed');
                showFailure('Verification timed out. Please follow the instructions.');
                return;
            }
        }

        if (isRunning) showSuccess();
    }

    function updateStepListUI() {
        sessionChallenges.forEach((c, index) => {
            const stepNum = index + 1;
            const stepEl = steps[stepNum];
            if (stepEl) {
                stepEl.querySelector('h4').textContent = c.title;
                stepEl.querySelector('p').textContent = c.desc;
            }
        });
    }

    async function runChallenge(challenge) {
        return new Promise((resolve) => {
            let count = 0;
            const startTime = Date.now();
            if (challenge.id === 'turn') headTurnedState = { left: false, right: false };

            const check = async () => {
                if (!isRunning) return resolve(false);

                if (video.currentTime !== lastVideoTime) {
                    lastVideoTime = video.currentTime;
                    const results = faceLandmarker.detectForVideo(video, lastVideoTime);
                    
                    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                        const blendshapes = results.faceBlendshapes[0].categories;
                        const landmarks = results.faceLandmarks[0];

                        let detected = false;
                        if (challenge.id === 'face') detected = true;
                        else if (challenge.id === 'blink') {
                            const blinkLeft = findBlendshape(blendshapes, 'eyeBlinkLeft');
                            const blinkRight = findBlendshape(blendshapes, 'eyeBlinkRight');
                            if (blinkLeft > 0.5 && blinkRight > 0.5) detected = true;
                        }
                        else if (challenge.id === 'smile') {
                            const smile = findBlendshape(blendshapes, 'mouthSmileLeft');
                            if (smile > 0.5) detected = true;
                        }
                        else if (challenge.id === 'turn') {
                            const nose = landmarks[1];
                            const leftEye = landmarks[33];
                            const rightEye = landmarks[263];
                            const leftDist = Math.abs(nose.x - leftEye.x);
                            const rightDist = Math.abs(nose.x - rightEye.x);
                            
                            if (leftDist < (rightDist * 0.45)) headTurnedState.right = true;
                            if (rightDist < (leftDist * 0.45)) headTurnedState.left = true;
                            
                            if (headTurnedState.left && headTurnedState.right) {
                                detected = true;
                            } else {
                                if (headTurnedState.left) guideText.textContent = "Now turn Right";
                                else if (headTurnedState.right) guideText.textContent = "Now turn Left";
                            }
                        }

                        if (detected) {
                            count++;
                            setProgress((count / challenge.target) * 100);
                            if (count >= challenge.target) return resolve(true);
                        }
                    } else {
                        guideText.textContent = "Face not detected";
                        guideText.style.color = "#ef4444";
                    }
                }

                if (Date.now() - startTime > 15000) return resolve(false);
                requestAnimationFrame(check);
            };
            check();
        });
    }

    function findBlendshape(blendshapes, name) {
        const item = blendshapes.find(b => b.categoryName === name);
        return item ? item.score : 0;
    }

    function updateChallengeUI(challenge) {
        challengeIcon.innerHTML = `<i data-lucide="${challenge.icon}"></i>`;
        challengeTitle.textContent = challenge.title;
        challengeDesc.textContent = challenge.desc;
        guideText.textContent = challenge.desc;
        guideText.style.color = "var(--text-primary)";
        lucide.createIcons();
        setProgress(0);
    }

    function updateStepStatus(stepNum, status) {
        const stepEl = steps[stepNum];
        if (!stepEl) return;
        stepEl.classList.remove('active', 'complete', 'failed');
        stepEl.classList.add(status);
        const statusIcon = stepEl.querySelector('.step-status');
        if (status === 'active') statusIcon.innerHTML = '<i data-lucide="loader" class="spinning"></i>';
        else if (status === 'complete') statusIcon.innerHTML = '<i data-lucide="check-circle-2" class="success"></i>';
        else if (status === 'failed') statusIcon.innerHTML = '<i data-lucide="x-circle" class="failed"></i>';
        lucide.createIcons();
    }

    function setProgress(percent) {
        const circumference = 2 * Math.PI * 54;
        progressCircle.style.strokeDashoffset = circumference - (percent / 100) * circumference;
    }

    function showSuccess() {
        isRunning = false;
        faceGuide.classList.remove('active');
        livenessResult.classList.remove('hidden');
        
        // Capture selfie
        const ctx = faceCanvas.getContext('2d');
        faceCanvas.width = video.videoWidth;
        faceCanvas.height = video.videoHeight;
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -faceCanvas.width, 0, faceCanvas.width, faceCanvas.height);
        ctx.restore();
        
        sessionStorage.setItem('livenessVerified', 'true');
        sessionStorage.setItem('livenessSelfie', faceCanvas.toDataURL('image/jpeg', 0.9));
        
        stopCamera();
    }

    function showFailure(msg) {
        isRunning = false;
        resultTitle.textContent = "Verification Failed";
        resultDesc.textContent = msg;
        livenessResult.classList.remove('hidden');
        stopCamera();
    }

    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

    startBtn.addEventListener('click', startLivenessCheck);
});
