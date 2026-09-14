// ==========================================
// NAVIGATION & DATA LOADING
// ==========================================

function switchTab(tabId) {
    // Hide all tab panes
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.add('hidden'));

    // Deactivate all tab buttons
    document.querySelectorAll('.nav-tab').forEach(btn => {
        btn.classList.remove('bg-indigo-600', 'text-white', 'shadow-sm');
        btn.classList.add('text-slate-600');
    });

    // Show selected tab pane
    const targetPane = document.getElementById(`tab-content-${tabId}`);
    if (targetPane) targetPane.classList.remove('hidden');

    // Activate selected button
    const targetBtn = document.getElementById(`tab-btn-${tabId}`);
    if (targetBtn) {
        targetBtn.classList.remove('text-slate-600');
        targetBtn.classList.add('bg-indigo-600', 'text-white', 'shadow-sm');
    }

    // Tab-specific refreshes
    if (tabId === 'leaderboard') renderLeaderboard();
    if (tabId === 'synergies') loadSynergies();
    if (tabId === 'players') {
        if (typeof currentAdminSubTab !== 'undefined' && currentAdminSubTab === 'matches') {
            loadAdminMatches();
        } else {
            renderAdminPlayers();
        }
    }
    if (tabId === 'captains') renderCaptainPlayers();
    if (tabId === 'matchmaker') renderMatchmakerPlayers();
    if (tabId === 'simulation') renderSimulationBoard();
}

// ==========================================
// DATA LOADING
// ==========================================
async function loadAllPlayers() {
    try {
        const res = await fetch('/api/players');
        const data = await res.json();
        if (data.success) {
            allPlayers = data.players;
            renderMatchmakerPlayers();
            renderAdminPlayers();
            renderLeaderboard();
            renderCaptainPlayers();
            renderSimulationBoard();
            document.getElementById('total-players-badge').innerText = `${allPlayers.length} Tuyển Thủ`;
        }
    } catch (err) {
        console.error("Lỗi khi tải danh sách người chơi:", err);
    }
}

async function loadStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.success) {
            document.getElementById('storage-mode-text').innerText = (data.storage_mode || 'LOCAL').toUpperCase();
            document.getElementById('storage-matches-text').innerText = `${data.total_matches} trận`;
            document.getElementById('storage-players-text').innerText = `${data.total_players} người`;
            const badge = document.getElementById('gemini-status-badge');
            if (badge && data.has_gemini) {
                badge.innerText = "Đã cấu hình Key (.env)";
                badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-bold";
            }
        }
        checkSupabaseStatus();
    } catch (err) {
        console.error("Lỗi khi tải trạng thái hệ thống:", err);
    }
}


function formatAiMarkdown(text) {
    return text
        .replace(/### (.*)/g, '<h4 class="font-bold text-indigo-700 text-sm mt-2 mb-1">$1</h4>')
        .replace(/\*\*(.*?)\*\*/g, '<b class="text-slate-900 font-bold">$1</b>')
        .replace(/- (.*)/g, '<li class="ml-4 list-disc text-slate-700">$1</li>')
        .replace(/\n\n/g, '<br>');
}

