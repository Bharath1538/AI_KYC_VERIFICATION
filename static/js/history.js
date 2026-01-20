/**
 * Verification History JavaScript
 * Manages localStorage history and table display
 */

document.addEventListener('DOMContentLoaded', () => {
    // =====================================================
    // Configuration
    // =====================================================
    const STORAGE_KEY = 'kyc_verification_history';
    const MAX_HISTORY_ITEMS = 100;

    // =====================================================
    // DOM Elements
    // =====================================================
    const tableBody = document.getElementById('historyTableBody');
    const emptyState = document.getElementById('historyEmpty');
    const clearBtn = document.getElementById('clearHistoryBtn');
    const filterBtns = document.querySelectorAll('.filter-btn');

    // Stats elements
    const totalVerifiedEl = document.getElementById('totalVerified');
    const totalFailedEl = document.getElementById('totalFailed');
    const totalScansEl = document.getElementById('totalScans');
    const successRateEl = document.getElementById('successRate');

    let currentFilter = 'all';

    // =====================================================
    // History Management
    // =====================================================
    function getHistory() {
        try {
            const data = localStorage.getItem(STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch (e) {
            console.error('Error reading history:', e);
            return [];
        }
    }

    function saveHistory(history) {
        try {
            // Limit history size
            if (history.length > MAX_HISTORY_ITEMS) {
                history = history.slice(0, MAX_HISTORY_ITEMS);
            }
            localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
        } catch (e) {
            console.error('Error saving history:', e);
        }
    }

    function clearHistory() {
        if (confirm('Are you sure you want to clear all verification history?')) {
            localStorage.removeItem(STORAGE_KEY);
            renderHistory();
            showToast('History cleared', 'success');
        }
    }

    function addToHistory(verification) {
        const history = getHistory();
        const entry = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            ...verification
        };
        history.unshift(entry);
        saveHistory(history);
        return entry;
    }

    // =====================================================
    // Rendering
    // =====================================================
    function renderHistory() {
        const history = getHistory();
        const filtered = filterHistory(history, currentFilter);

        // Update stats
        updateStats(history);

        // Clear table
        tableBody.innerHTML = '';

        if (filtered.length === 0) {
            emptyState.classList.remove('hidden');
            return;
        }

        emptyState.classList.add('hidden');

        // Render rows
        filtered.forEach(item => {
            const row = createHistoryRow(item);
            tableBody.appendChild(row);
        });

        lucide.createIcons();
    }

    function filterHistory(history, filter) {
        if (filter === 'all') return history;
        if (filter === 'verified') {
            return history.filter(h => h.status === 'verified');
        }
        if (filter === 'failed') {
            return history.filter(h => h.status !== 'verified');
        }
        return history;
    }

    function createHistoryRow(item) {
        const row = document.createElement('tr');

        // Format date
        const date = new Date(item.timestamp);
        const dateStr = date.toLocaleDateString('en-IN', {
            day: '2-digit',
            month: 'short',
            year: 'numeric'
        });
        const timeStr = date.toLocaleTimeString('en-IN', {
            hour: '2-digit',
            minute: '2-digit'
        });

        // Status badge
        const isVerified = item.status === 'verified';
        const statusClass = isVerified ? 'status-verified' : 'status-failed';
        const statusText = isVerified ? 'Verified' : 'Failed';
        const statusIcon = isVerified ? 'check-circle-2' : 'x-circle';

        // Confidence display
        const confidence = item.confidence ? `${Math.round(item.confidence * 100)}%` : 'N/A';

        // Mask Aadhaar number
        const aadhaar = item.aadhaarNumber || 'N/A';
        const maskedAadhaar = aadhaar.length >= 12 ?
            `XXXX XXXX ${aadhaar.slice(-4)}` : aadhaar;

        row.innerHTML = `
            <td>
                <div class="date-cell">
                    <span class="date">${dateStr}</span>
                    <span class="time">${timeStr}</span>
                </div>
            </td>
            <td>
                <span class="doc-type">${item.docType || 'Unknown'}</span>
            </td>
            <td>${item.name || 'N/A'}</td>
            <td class="aadhaar-number">${maskedAadhaar}</td>
            <td>
                <span class="status-badge ${statusClass}">
                    <i data-lucide="${statusIcon}"></i>
                    <span>${statusText}</span>
                </span>
            </td>
            <td>
                <span class="confidence-badge">${confidence}</span>
            </td>
        `;

        return row;
    }

    function updateStats(history) {
        const verified = history.filter(h => h.status === 'verified').length;
        const failed = history.filter(h => h.status !== 'verified').length;
        const total = history.length;
        const rate = total > 0 ? Math.round((verified / total) * 100) : 0;

        totalVerifiedEl.textContent = verified;
        totalFailedEl.textContent = failed;
        totalScansEl.textContent = total;
        successRateEl.textContent = `${rate}%`;
    }

    // =====================================================
    // Event Listeners
    // =====================================================
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.filter;
            renderHistory();
        });
    });

    clearBtn.addEventListener('click', clearHistory);

    // =====================================================
    // Initialize
    // =====================================================
    renderHistory();

    // Expose function to add history from verification page
    window.addVerificationToHistory = addToHistory;
});
