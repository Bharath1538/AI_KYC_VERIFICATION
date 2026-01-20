/**
 * AI KYC Verification - Verification Page JavaScript
 * Handles file upload, camera capture, and verification API calls
 */

document.addEventListener('DOMContentLoaded', () => {
    // =====================================================
    // DOM Elements
    // =====================================================
    const modeTabs = document.querySelectorAll('.mode-tab');
    const uploadPanel = document.getElementById('uploadPanel');
    const cameraPanel = document.getElementById('cameraPanel');
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImage = document.getElementById('previewImage');
    const removeImageBtn = document.getElementById('removeImage');
    const cameraFeed = document.getElementById('cameraFeed');
    const captureBtn = document.getElementById('captureBtn');
    const verifyBtn = document.getElementById('verifyBtn');
    const resultsPlaceholder = document.querySelector('.results-placeholder');
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');
    const resultHeader = document.getElementById('resultHeader');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const resultType = document.getElementById('resultType');
    const extractedData = document.getElementById('extractedData');
    const resetBtn = document.getElementById('resetBtn');

    let currentMode = 'upload';
    let selectedFile = null;
    let mediaStream = null;

    // =====================================================
    // Mode Switching
    // =====================================================
    modeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const mode = tab.dataset.mode;
            if (mode === currentMode) return;

            // Update tabs
            modeTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Switch panels
            if (mode === 'upload') {
                uploadPanel.classList.remove('hidden');
                cameraPanel.classList.add('hidden');
                stopCamera();
            } else {
                uploadPanel.classList.add('hidden');
                cameraPanel.classList.remove('hidden');
                startCamera();
            }

            currentMode = mode;
            updateVerifyButton();
        });
    });

    // =====================================================
    // File Upload
    // =====================================================
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            showToast('Please select a valid image file (JPG, PNG, WebP, GIF)', 'error');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            showToast('File size must be less than 16MB', 'error');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadZone.classList.add('hidden');
            imagePreview.classList.remove('hidden');
            updateVerifyButton();
        };
        reader.readAsDataURL(file);
    }

    removeImageBtn.addEventListener('click', () => {
        selectedFile = null;
        previewImage.src = '';
        imagePreview.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        fileInput.value = '';
        updateVerifyButton();
    });

    // =====================================================
    // Camera
    // =====================================================
    async function startCamera() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                }
            });
            cameraFeed.srcObject = mediaStream;
            updateVerifyButton();
        } catch (err) {
            console.error('Camera error:', err);
            showToast('Unable to access camera. Please check permissions.', 'error');

            // Switch back to upload mode
            modeTabs[0].click();
        }
    }

    function stopCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
            cameraFeed.srcObject = null;
        }
    }

    captureBtn.addEventListener('click', () => {
        if (!mediaStream) return;

        // Create canvas and capture frame
        const canvas = document.createElement('canvas');
        canvas.width = cameraFeed.videoWidth;
        canvas.height = cameraFeed.videoHeight;
        const ctx = canvas.getContext('2d');

        // Draw image WITHOUT flipping - OCR needs correct orientation
        // (The video preview is mirrored in CSS for user comfort, but capture must be normal)
        ctx.drawImage(cameraFeed, 0, 0);

        // Convert to blob
        canvas.toBlob((blob) => {
            selectedFile = new File([blob], 'capture.jpg', { type: 'image/jpeg' });

            // Switch to upload view with captured image
            previewImage.src = canvas.toDataURL('image/jpeg');
            uploadPanel.classList.remove('hidden');
            cameraPanel.classList.add('hidden');
            uploadZone.classList.add('hidden');
            imagePreview.classList.remove('hidden');

            stopCamera();
            currentMode = 'upload';
            modeTabs.forEach(t => t.classList.remove('active'));
            modeTabs[0].classList.add('active');

            updateVerifyButton();
            showToast('Image captured successfully!', 'success');
        }, 'image/jpeg', 0.9);
    });

    // =====================================================
    // Verification
    // =====================================================
    function updateVerifyButton() {
        verifyBtn.disabled = !selectedFile;
    }

    verifyBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading
        resultsPlaceholder.classList.add('hidden');
        resultsContent.classList.add('hidden');
        loadingState.classList.remove('hidden');
        verifyBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/api/verify', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Check for liveness selfie and do face matching
                const selfie = sessionStorage.getItem('livenessSelfie');
                if (selfie && selectedFile) {
                    console.log('Performing face matching with liveness selfie...');
                    try {
                        const matchFormData = new FormData();
                        matchFormData.append('file', selectedFile);
                        matchFormData.append('selfie', selfie);

                        const matchResponse = await fetch('/api/face-match', {
                            method: 'POST',
                            body: matchFormData
                        });
                        const matchData = await matchResponse.json();

                        if (matchData.success && matchData.result) {
                            data.result.face_match = matchData.result;
                            console.log('Face match result:', matchData.result);
                        }
                    } catch (e) {
                        console.log('Face matching skipped:', e);
                    }
                }

                displayResults(data.result, data.demo_mode);
            } else {
                showError(data.error || 'Verification failed');
            }
        } catch (err) {
            console.error('Verification error:', err);
            showError('Network error. Please try again.');
        }
    });

    function displayResults(result, isDemoMode) {
        loadingState.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        // Update status based on database verification
        const statusEl = resultHeader.querySelector('.result-status');
        const verification = result.verification;
        const confidence = result.confidence || 0;

        if (verification) {
            // Use database verification status
            const status = verification.status;
            if (status === 'verified') {
                statusEl.innerHTML = '<i data-lucide="shield-check"></i><span>Verified</span>';
                statusEl.classList.remove('failed');
                statusEl.style.color = 'var(--accent-green)';
            } else if (status === 'partial_match') {
                statusEl.innerHTML = '<i data-lucide="alert-triangle"></i><span>Partial Match</span>';
                statusEl.classList.remove('failed');
                statusEl.style.color = 'var(--accent-orange)';
            } else if (status === 'not_found') {
                statusEl.innerHTML = '<i data-lucide="search-x"></i><span>Not in Database</span>';
                statusEl.classList.add('failed');
                statusEl.style.color = 'var(--accent-orange)';
            } else {
                statusEl.innerHTML = '<i data-lucide="x-circle"></i><span>Mismatch</span>';
                statusEl.classList.add('failed');
                statusEl.style.color = 'var(--accent-red)';
            }
        } else if (confidence >= 0.7) {
            statusEl.innerHTML = '<i data-lucide="check-circle-2"></i><span>Detected</span>';
            statusEl.classList.remove('failed');
        } else {
            statusEl.innerHTML = '<i data-lucide="x-circle"></i><span>Low Confidence</span>';
            statusEl.classList.add('failed');
        }

        // Update confidence badge
        if (verification && verification.match_score !== undefined) {
            confidenceBadge.querySelector('span').textContent = `${verification.match_score}% Match`;
        } else {
            confidenceBadge.querySelector('span').textContent = `${Math.round(confidence * 100)}%`;
        }

        // Update document type
        resultType.querySelector('.type-value').textContent = result.doc_type || 'Unknown';

        // Display extracted data
        extractedData.innerHTML = '';
        const extractedFields = result.extracted_data || {};

        for (const [key, value] of Object.entries(extractedFields)) {
            if (key === 'raw_text' || key === 'validation') continue;

            const fieldEl = document.createElement('div');
            fieldEl.className = 'data-field';
            fieldEl.innerHTML = `
                <span class="field-label">${key}</span>
                <span class="field-value">${value || 'N/A'}</span>
            `;
            extractedData.appendChild(fieldEl);
        }

        // Show verification details
        if (verification) {
            // Add verification status message
            const statusEl = document.createElement('div');
            statusEl.className = 'data-field verification-status';
            const statusColor = verification.status === 'verified' ? 'var(--accent-green)' :
                verification.status === 'partial_match' ? 'var(--accent-orange)' : 'var(--accent-red)';
            statusEl.style.borderLeft = `3px solid ${statusColor}`;
            statusEl.innerHTML = `
                <span class="field-label" style="color: ${statusColor};">Database Status</span>
                <span class="field-value" style="color: ${statusColor};">${verification.message || verification.status}</span>
            `;
            extractedData.appendChild(statusEl);

            // Show matched fields
            if (verification.matched_fields && verification.matched_fields.length > 0) {
                const matchedEl = document.createElement('div');
                matchedEl.className = 'data-field';
                matchedEl.style.borderLeft = '3px solid var(--accent-green)';
                matchedEl.innerHTML = `
                    <span class="field-label" style="color: var(--accent-green);">✓ Matched</span>
                    <span class="field-value" style="color: var(--accent-green);">${verification.matched_fields.join(', ')}</span>
                `;
                extractedData.appendChild(matchedEl);
            }

            // Show mismatched fields
            if (verification.mismatched_fields && verification.mismatched_fields.length > 0) {
                for (const mismatch of verification.mismatched_fields) {
                    const mismatchEl = document.createElement('div');
                    mismatchEl.className = 'data-field';
                    mismatchEl.style.borderLeft = '3px solid var(--accent-red)';
                    mismatchEl.innerHTML = `
                        <span class="field-label" style="color: var(--accent-red);">✗ ${mismatch.field}</span>
                        <span class="field-value" style="color: var(--accent-red);">Got: ${mismatch.extracted}<br>Expected: ${mismatch.expected}</span>
                    `;
                    extractedData.appendChild(mismatchEl);
                }
            }
        }

        // Add demo mode indicator
        if (isDemoMode) {
            const demoEl = document.createElement('div');
            demoEl.className = 'data-field';
            demoEl.style.borderLeft = '3px solid var(--accent-orange)';
            demoEl.innerHTML = `
                <span class="field-label" style="color: var(--accent-orange);">Demo Mode</span>
                <span class="field-value" style="color: var(--accent-orange);">Sample data</span>
            `;
            extractedData.appendChild(demoEl);
        }

        // Add face match result if available
        if (result.face_match) {
            const faceMatch = result.face_match;
            const isMatch = faceMatch.match;
            const score = faceMatch.score || 0;
            const color = isMatch ? 'var(--accent-green)' : 'var(--accent-red)';
            const icon = isMatch ? 'check-circle-2' : 'x-circle';
            const statusText = isMatch ? 'Face Match Confirmed' : 'Face Mismatch';

            const faceMatchEl = document.createElement('div');
            faceMatchEl.className = 'data-field';
            faceMatchEl.style.borderLeft = `3px solid ${color}`;
            faceMatchEl.style.marginTop = '1.5rem';
            faceMatchEl.innerHTML = `
                <span class="field-label" style="color: ${color};">
                    <i data-lucide="${icon}" style="width: 16px; height: 16px; margin-right: 4px;"></i>
                    Face Matching
                </span>
                <span class="field-value" style="color: ${color}; font-weight: 600;">
                    ${statusText} (${score}% similarity)
                </span>
            `;
            extractedData.appendChild(faceMatchEl);

            if (!isMatch) {
                showToast('Face does not match document photo!', 'error');
            }
        }

        // Re-initialize icons
        lucide.createIcons();

        // Save to history
        saveToHistory(result, verification);

        // Show appropriate toast
        if (verification && verification.status === 'verified') {
            showToast('Identity verified successfully!', 'success');
        } else if (verification && verification.status === 'partial_match') {
            showToast('Partial match found', 'warning');
        } else {
            showToast('Verification complete', 'success');
        }
    }

    function saveToHistory(result, verification) {
        try {
            const STORAGE_KEY = 'kyc_verification_history';
            const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');

            const entry = {
                id: Date.now(),
                timestamp: new Date().toISOString(),
                docType: result.doc_type || 'Unknown',
                name: result.extracted_data?.Name || 'N/A',
                aadhaarNumber: result.extracted_data?.['Aadhaar Number'] || 'N/A',
                dob: result.extracted_data?.DOB || 'N/A',
                gender: result.extracted_data?.Gender || 'N/A',
                confidence: result.confidence || 0,
                status: verification?.status || (result.confidence > 0.7 ? 'verified' : 'failed'),
                matchScore: verification?.match_score || null
            };

            history.unshift(entry);

            // Keep only last 100 entries
            if (history.length > 100) {
                history.pop();
            }

            localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
            console.log('Saved to history:', entry);
        } catch (e) {
            console.error('Error saving to history:', e);
        }
    }

    function showError(message) {
        loadingState.classList.add('hidden');
        resultsPlaceholder.classList.remove('hidden');
        resultsPlaceholder.innerHTML = `
            <div class="placeholder-icon" style="background: rgba(239, 68, 68, 0.1); color: var(--accent-red);">
                <i data-lucide="alert-triangle"></i>
            </div>
            <h3 style="color: var(--accent-red);">Verification Failed</h3>
            <p>${message}</p>
        `;
        lucide.createIcons();
        verifyBtn.disabled = false;
        showToast(message, 'error');
    }

    // =====================================================
    // Reset
    // =====================================================
    resetBtn.addEventListener('click', () => {
        // Reset file
        selectedFile = null;
        previewImage.src = '';
        imagePreview.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        fileInput.value = '';

        // Reset results
        resultsContent.classList.add('hidden');
        resultsPlaceholder.classList.remove('hidden');
        resultsPlaceholder.innerHTML = `
            <div class="placeholder-icon">
                <i data-lucide="scan-search"></i>
            </div>
            <h3>Ready to Verify</h3>
            <p>Upload or capture a document to see the verification results</p>
        `;
        lucide.createIcons();

        updateVerifyButton();
    });

    // =====================================================
    // Cleanup on page unload
    // =====================================================
    window.addEventListener('beforeunload', () => {
        stopCamera();
    });
});
