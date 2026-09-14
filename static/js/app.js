// FBCS AI 3.0 Client Application Logic (Clean Modern Light Theme)
let allPlayers = [];
let selectedMatchmaker = [];
let selectedCaptains = [];
let selectedCaptainMembers = [];
let currentTeamsResult = null;

// Trạng thái cho tab Mô Phỏng 5vs5 (Simulation)
let simTeam1 = [null, null, null, null, null];
let simTeam2 = [null, null, null, null, null];
let simUnmatchedSlotInfo = { team1: {}, team2: {} };
let pendingSlotAssignment = null;
let lastOcrResult = null;

const SWAL_THEME = {
    background: '#ffffff',
    color: '#0f172a',
    confirmButtonColor: '#4f46e5',
    cancelButtonColor: '#94a3b8'
};

// ==========================================
// INITIALIZATION & TAB SWITCHING
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    loadAllPlayers();
    loadStatus();
});

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
    if (tabId === 'players') renderAdminPlayers();
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

// ==========================================
// TAB 1: MATCHMAKER (CHIA ĐỘI)
// ==========================================
function renderMatchmakerPlayers() {
    const grid = document.getElementById('matchmaker-players-grid');
    if (!grid) return;
    grid.innerHTML = '';

    const query = (document.getElementById('matchmaker-search')?.value || '').toLowerCase().trim();

    allPlayers.forEach(p => {
        if (query && !p.id.toLowerCase().includes(query) && !p.nickname.toLowerCase().includes(query)) {
            return;
        }

        const isSelected = selectedMatchmaker.includes(p.id);

        const card = document.createElement('div');
        card.className = `p-3 rounded-2xl border cursor-pointer transition-all duration-150 flex flex-col items-center text-center relative select-none ${
            isSelected 
                ? 'bg-indigo-50/90 border-indigo-600 ring-2 ring-indigo-500/20 shadow-sm scale-[1.02]' 
                : 'bg-white border-slate-200 hover:border-indigo-300 hover:shadow-xs'
        }`;

        card.onclick = () => toggleMatchmakerPlayer(p.id);

        // Form icon badge
        const formIcon = p.form?.icon || '🌱';
        const formStatus = p.form?.status || 'neutral';
        const formBadgeColor = formStatus === 'on_fire' ? 'text-amber-600' : (formStatus === 'cold' ? 'text-blue-600' : 'text-slate-600');

        card.innerHTML = `
            ${isSelected ? `
                <div class="absolute top-2 right-2 w-5 h-5 rounded-full bg-indigo-600 text-white flex items-center justify-center text-[10px] font-black shadow-xs">
                    <i class="fa-solid fa-check"></i>
                </div>
            ` : ''}
            <div class="relative mb-2">
                <img src="${p.avatar}" alt="${p.nickname}" class="w-12 h-12 rounded-xl object-cover bg-slate-100 border border-slate-200">
                <span class="absolute -bottom-1 -right-1 text-xs" title="${p.form?.label || ''}">${formIcon}</span>
            </div>
            <h4 class="font-bold font-heading text-xs text-slate-900 truncate max-w-[100px]">${p.nickname}</h4>
            <div class="flex items-center gap-1.5 mt-1.5">
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 font-bold border border-indigo-100">
                    Elo ${Math.round(p.hidden_elo)}
                </span>
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 ${formBadgeColor} font-bold border border-slate-200">
                    ${p.effective_power}
                </span>
            </div>
        `;
        grid.appendChild(card);
    });

    updateMatchmakerCounter();
}

function filterMatchmakerPlayers() {
    renderMatchmakerPlayers();
}

function toggleMatchmakerPlayer(id) {
    const idx = selectedMatchmaker.indexOf(id);
    if (idx > -1) {
        selectedMatchmaker.splice(idx, 1);
    } else {
        if (selectedMatchmaker.length >= 10) {
            Swal.fire({
                icon: 'warning',
                title: 'Đã đủ 10 người',
                text: 'Chỉ được chọn tối đa 10 tuyển thủ để chia thành 2 đội 5-5.',
                ...SWAL_THEME
            });
            return;
        }
        selectedMatchmaker.push(id);
    }
    renderMatchmakerPlayers();
}

function updateMatchmakerCounter() {
    const count = selectedMatchmaker.length;
    document.getElementById('selected-count-text').innerText = count;
    const badge = document.getElementById('selected-counter-badge');
    if (count === 10) {
        badge.className = "text-sm px-3 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 font-bold";
    } else {
        badge.className = "text-sm px-3 py-1 rounded-full bg-slate-100 text-indigo-700 border border-slate-200 font-bold";
    }
}

function clearSelectedPlayers() {
    selectedMatchmaker = [];
    currentTeamsResult = null;
    document.getElementById('teams-result-section').classList.add('hidden');
    renderMatchmakerPlayers();
}

async function handleCreateTeams() {
    if (selectedMatchmaker.length !== 10) {
        Swal.fire({
            icon: 'info',
            title: 'Chưa đủ người',
            text: `Bạn hiện mới chọn ${selectedMatchmaker.length}/10 tuyển thủ. Vui lòng chọn đủ 10 người để chia đội.`,
            ...SWAL_THEME
        });
        return;
    }

    const balanceMode = document.getElementById('matchmaking-mode')?.value || 'composite';

    try {
        const res = await fetch('/api/create_teams', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                players: selectedMatchmaker,
                balance_mode: balanceMode
            })
        });
        const data = await res.json();
        if (data.success) {
            currentTeamsResult = data;
            displayTeamsResult(data);
        } else {
            Swal.fire({
                icon: 'error',
                title: 'Lỗi chia đội',
                text: data.error || 'Không thể tính toán phân chia đội hình.',
                ...SWAL_THEME
            });
        }
    } catch (err) {
        console.error("Lỗi:", err);
    }
}

function displayTeamsResult(data) {
    const section = document.getElementById('teams-result-section');
    section.classList.remove('hidden');

    const isPureElo = data.balance_mode === 'pure_elo';

    document.getElementById('res-power-diff').innerText = isPureElo ? `${data.power_difference} Elo` : data.power_difference;
    document.getElementById('res-win-prob-label').innerText = `${data.team1_win_prob}% - ${data.team2_win_prob}%`;

    // Mode badge
    const modeBadge = document.getElementById('res-mode-badge');
    if (modeBadge) {
        if (isPureElo) {
            modeBadge.innerHTML = `<i class="fa-solid fa-crosshairs"></i> <span>Chế độ: Thuần Elo Ẩn</span>`;
            modeBadge.className = "px-3 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-bold font-heading flex items-center gap-1.5";
        } else {
            modeBadge.innerHTML = `<i class="fa-solid fa-bolt"></i> <span>Chế độ: Toàn Diện (Elo + Stats)</span>`;
            modeBadge.className = "px-3 py-1 rounded-full bg-purple-50 text-purple-700 border border-purple-200 text-xs font-bold font-heading flex items-center gap-1.5";
        }
    }

    // RNG info badge
    const rngBadge = document.getElementById('res-rng-badge');
    if (rngBadge) {
        if (data.rng_applied) {
            const diffText = isPureElo ? `Sai số ±${data.power_difference} Elo` : `Sai số ±${data.power_difference}`;
            rngBadge.innerHTML = `<i class="fa-solid fa-dice"></i> <span>RNG Cân Bằng: ${diffText}</span>`;
            rngBadge.className = "px-3 py-1 rounded-full bg-amber-50 text-amber-800 border border-amber-200 text-xs font-bold font-heading flex items-center gap-1.5";
        } else {
            rngBadge.innerHTML = `<i class="fa-solid fa-scale-balanced"></i> <span>Cân bằng tuyệt đối</span>`;
            rngBadge.className = "px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200 text-xs font-bold font-heading flex items-center gap-1.5";
        }
    }

    const poolLabel = document.getElementById('res-pool-count-label');
    if (poolLabel) {
        poolLabel.innerText = `Đã chọn từ nhóm ${data.pool_candidates_count || 1} phương án tối ưu`;
    }

    document.getElementById('team1-power-text').innerText = isPureElo ? `Elo ${data.team1_power}` : data.team1_power;
    document.getElementById('team2-power-text').innerText = isPureElo ? `Elo ${data.team2_power}` : data.team2_power;

    document.getElementById('team1-prob-badge').innerText = `${data.team1_win_prob}% Thắng`;
    document.getElementById('team2-prob-badge').innerText = `${data.team2_win_prob}% Thắng`;

    // Render Team 1
    const t1List = document.getElementById('team1-players-list');
    t1List.innerHTML = '';
    data.team1.forEach(p => {
        t1List.appendChild(createTeamPlayerCard(p, 'blue', isPureElo));
    });

    // Render Team 1 Synergies & Chemistry
    const t1SynContainer = document.getElementById('team1-synergies-container');
    const t1SynList = document.getElementById('team1-synergies-list');
    if (t1SynContainer && t1SynList) {
        if (data.team1_synergies && data.team1_synergies.length > 0) {
            t1SynContainer.classList.remove('hidden');
            t1SynList.innerHTML = '';
            data.team1_synergies.forEach(s => {
                const span = document.createElement('span');
                const isPositive = (s.bonus || 0) >= 0;
                span.className = `text-[10px] font-semibold px-2 py-1 rounded-lg border flex items-center gap-1 ${
                    isPositive ? 'bg-blue-100/70 border-blue-200 text-blue-800' : 'bg-rose-100/70 border-rose-200 text-rose-800'
                }`;
                span.title = s.label || '';
                span.innerHTML = `<span>${s.icon || '🤝'}</span> <span><b class="font-heading">${s.names}</b> (${isPositive ? '+' : ''}${s.bonus})</span>`;
                t1SynList.appendChild(span);
            });
        } else {
            t1SynContainer.classList.add('hidden');
        }
    }

    // Render Team 2
    const t2List = document.getElementById('team2-players-list');
    t2List.innerHTML = '';
    data.team2.forEach(p => {
        t2List.appendChild(createTeamPlayerCard(p, 'rose', isPureElo));
    });

    // Render Team 2 Synergies & Chemistry
    const t2SynContainer = document.getElementById('team2-synergies-container');
    const t2SynList = document.getElementById('team2-synergies-list');
    if (t2SynContainer && t2SynList) {
        if (data.team2_synergies && data.team2_synergies.length > 0) {
            t2SynContainer.classList.remove('hidden');
            t2SynList.innerHTML = '';
            data.team2_synergies.forEach(s => {
                const span = document.createElement('span');
                const isPositive = (s.bonus || 0) >= 0;
                span.className = `text-[10px] font-semibold px-2 py-1 rounded-lg border flex items-center gap-1 ${
                    isPositive ? 'bg-rose-100/70 border-rose-200 text-rose-800' : 'bg-slate-100 border-slate-200 text-slate-700'
                }`;
                span.title = s.label || '';
                span.innerHTML = `<span>${s.icon || '🤝'}</span> <span><b class="font-heading">${s.names}</b> (${isPositive ? '+' : ''}${s.bonus})</span>`;
                t2SynList.appendChild(span);
            });
        } else {
            t2SynContainer.classList.add('hidden');
        }
    }

    // Auto scroll to results
    section.scrollIntoView({ behavior: 'smooth' });

    // Auto trigger initial AI analysis preview
    requestAiAnalysis();
}

async function rerollTeams() {
    const btn = document.getElementById('btn-reroll-teams');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-sm"></i> <span>Đang xếp...</span>';
    }

    try {
        if (currentTeamsResult && currentTeamsResult.captain1 && selectedCaptains.length === 2 && selectedCaptainMembers.length === 8) {
            await handleCreateTeamsWithCaptains();
        } else {
            await handleCreateTeams();
        }
    } catch (err) {
        console.error("Lỗi khi xếp lại đội:", err);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-dice text-sm"></i> <span>Xếp Lại (RNG)</span>';
        }
    }
}

function createTeamPlayerCard(p, teamColor, isPureElo = false) {
    const div = document.createElement('div');
    div.className = "flex items-center justify-between p-2.5 rounded-xl bg-white border border-slate-200/90 shadow-xs";
    
    if (isPureElo) {
        div.innerHTML = `
            <div class="flex items-center gap-3">
                <div class="relative">
                    <img src="${p.avatar}" class="w-9 h-9 rounded-lg bg-slate-100 border border-slate-200 object-cover" alt="${p.nickname}">
                    <span class="absolute -bottom-1 -right-1 text-[10px]">${p.form?.icon || '🌱'}</span>
                </div>
                <div>
                    <h5 class="font-bold font-heading text-xs text-slate-900">${p.nickname}</h5>
                    <span class="text-[10px] text-slate-500">Elo Ẩn: <b class="text-indigo-600 font-bold">${Math.round(p.hidden_elo)}</b></span>
                </div>
            </div>
            <div class="text-right">
                <span class="text-xs font-black ${teamColor === 'blue' ? 'text-blue-700' : 'text-rose-700'}">
                    Elo ${Math.round(p.hidden_elo)}
                </span>
                <div class="text-[10px] text-slate-400">${p.form?.label?.split(' ')[0] || ''}</div>
            </div>
        `;
    } else {
        div.innerHTML = `
            <div class="flex items-center gap-3">
                <div class="relative">
                    <img src="${p.avatar}" class="w-9 h-9 rounded-lg bg-slate-100 border border-slate-200 object-cover" alt="${p.nickname}">
                    <span class="absolute -bottom-1 -right-1 text-[10px]">${p.form?.icon || '🌱'}</span>
                </div>
                <div>
                    <h5 class="font-bold font-heading text-xs text-slate-900">${p.nickname}</h5>
                    <span class="text-[10px] text-slate-500">Elo: <b class="text-slate-700">${Math.round(p.hidden_elo)}</b> • Kỹ năng: <b class="text-amber-600">${p.skill}/10</b></span>
                </div>
            </div>
            <div class="text-right">
                <span class="text-xs font-black ${teamColor === 'blue' ? 'text-blue-700' : 'text-rose-700'}">
                    ${p.effective_power}
                </span>
                <div class="text-[10px] text-slate-400">${p.form?.label?.split(' ')[0] || ''}</div>
            </div>
        `;
    }
    return div;
}

let isSubmittingMatch = false;

async function submitMatchWinner(winningTeam) {
    if (!currentTeamsResult || isSubmittingMatch) return;

    const winnerLabel = winningTeam === 'team1' ? 'Đội Xanh' : 'Đội Đỏ';

    const confirm = await Swal.fire({
        title: `Xác nhận ${winnerLabel} Thắng?`,
        text: 'Hệ thống sẽ lưu kết quả trận đấu, tự động cập nhật Elo ẩn và tính lại chuỗi Phong độ cho 10 tuyển thủ.',
        icon: 'question',
        showCancelButton: true,
        confirmButtonText: 'Đồng Ý Lưu',
        cancelButtonText: 'Hủy',
        showLoaderOnConfirm: true,
        preConfirm: async () => {
            isSubmittingMatch = true;
            try {
                const res = await fetch('/api/update_match_result', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        team1: currentTeamsResult.team1_names,
                        team2: currentTeamsResult.team2_names,
                        winner: winningTeam,
                        team1_power: currentTeamsResult.team1_power || 0,
                        team2_power: currentTeamsResult.team2_power || 0,
                        synergies: {
                            team1: currentTeamsResult.team1_synergies || [],
                            team2: currentTeamsResult.team2_synergies || []
                        }
                    })
                });
                const data = await res.json();
                if (!data.success) {
                    throw new Error(data.error || 'Lỗi khi lưu kết quả trận đấu');
                }
                return data;
            } catch (err) {
                Swal.showValidationMessage(err.message || 'Lỗi kết nối khi lưu kết quả');
            } finally {
                isSubmittingMatch = false;
            }
        },
        allowOutsideClick: () => !Swal.isLoading(),
        ...SWAL_THEME
    });

    if (confirm.isConfirmed && confirm.value) {
        Swal.fire({
            icon: 'success',
            title: 'Đã cập nhật trận đấu!',
            text: 'Elo ẩn và Phong độ đã được cập nhật thành công.',
            timer: 2000,
            showConfirmButton: false,
            ...SWAL_THEME
        });
        await loadAllPlayers();
        await loadStatus();
        clearSelectedPlayers();
    }
}

async function requestAiAnalysis() {
    if (!currentTeamsResult) return;
    const box = document.getElementById('ai-analysis-content');
    box.innerHTML = '<span class="text-indigo-600 animate-pulse font-semibold"><i class="fa-solid fa-spinner fa-spin mr-2"></i> AI đang phân tích chiến thuật đội hình...</span>';

    try {
        const res = await fetch('/api/ai_analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                team1: currentTeamsResult.team1,
                team2: currentTeamsResult.team2
            })
        });
        const data = await res.json();
        if (data.success) {
            box.innerHTML = formatAiMarkdown(data.analysis);
        } else {
            box.innerText = data.error || 'Không thể phân tích lúc này.';
        }
    } catch (err) {
        box.innerText = 'Lỗi kết nối khi gửi yêu cầu phân tích AI.';
    }
}

function formatAiMarkdown(text) {
    return text
        .replace(/### (.*)/g, '<h4 class="font-bold text-indigo-700 text-sm mt-2 mb-1">$1</h4>')
        .replace(/\*\*(.*?)\*\*/g, '<b class="text-slate-900 font-bold">$1</b>')
        .replace(/- (.*)/g, '<li class="ml-4 list-disc text-slate-700">$1</li>')
        .replace(/\n\n/g, '<br>');
}

// ==========================================
// TAB 2: CAPTAINS MODE (CHIA ĐỘI ĐỘI TRƯỞNG)
// ==========================================
function renderCaptainPlayers() {
    const grid = document.getElementById('captains-players-grid');
    if (!grid) return;
    grid.innerHTML = '';

    allPlayers.forEach(p => {
        const isCap1 = selectedCaptains[0] === p.id;
        const isCap2 = selectedCaptains[1] === p.id;
        const isMember = selectedCaptainMembers.includes(p.id);

        const card = document.createElement('div');
        let cardStyle = 'bg-white border-slate-200 hover:border-slate-300';
        let badgeText = '';

        if (isCap1) {
            cardStyle = 'bg-amber-50 border-amber-400 ring-2 ring-amber-400/20 shadow-sm scale-[1.02]';
            badgeText = '👑 Cap 1';
        } else if (isCap2) {
            cardStyle = 'bg-amber-50 border-amber-400 ring-2 ring-amber-400/20 shadow-sm scale-[1.02]';
            badgeText = '👑 Cap 2';
        } else if (isMember) {
            cardStyle = 'bg-indigo-50 border-indigo-400 shadow-xs';
            badgeText = 'Member';
        }

        card.className = `p-3 rounded-2xl border cursor-pointer transition-all duration-150 flex flex-col items-center text-center relative select-none ${cardStyle}`;
        card.onclick = () => toggleCaptainSelection(p.id);

        card.innerHTML = `
            ${badgeText ? `
                <span class="absolute top-1.5 right-1.5 text-[9px] font-extrabold px-1.5 py-0.5 rounded-full ${isCap1 || isCap2 ? 'bg-amber-500 text-white' : 'bg-indigo-600 text-white'}">
                    ${badgeText}
                </span>
            ` : ''}
            <img src="${p.avatar}" class="w-11 h-11 rounded-xl bg-slate-100 border border-slate-200 object-cover mb-1.5">
            <h4 class="font-bold font-heading text-xs text-slate-900 truncate max-w-[100px]">${p.nickname}</h4>
            <span class="text-[10px] text-slate-500 mt-0.5">Elo: ${Math.round(p.hidden_elo)}</span>
        `;
        grid.appendChild(card);
    });

    const cap1Obj = allPlayers.find(p => p.id === selectedCaptains[0]);
    const cap2Obj = allPlayers.find(p => p.id === selectedCaptains[1]);
    document.getElementById('cap1-display-name').innerText = cap1Obj ? cap1Obj.nickname : 'Chưa chọn';
    document.getElementById('cap2-display-name').innerText = cap2Obj ? cap2Obj.nickname : 'Chưa chọn';

    const totalSelected = selectedCaptains.length + selectedCaptainMembers.length;
    document.getElementById('captain-selection-text').innerText = `Đã chọn: ${totalSelected}/10 (${selectedCaptains.length} Cap, ${selectedCaptainMembers.length} Men)`;
}

function toggleCaptainSelection(id) {
    if (selectedCaptains[0] === id) {
        selectedCaptains.splice(0, 1);
        renderCaptainPlayers();
        return;
    }
    if (selectedCaptains[1] === id) {
        selectedCaptains.splice(1, 1);
        renderCaptainPlayers();
        return;
    }
    const mIdx = selectedCaptainMembers.indexOf(id);
    if (mIdx > -1) {
        selectedCaptainMembers.splice(mIdx, 1);
        renderCaptainPlayers();
        return;
    }

    if (selectedCaptains.length < 2) {
        selectedCaptains.push(id);
    } else {
        if (selectedCaptainMembers.length >= 8) {
            Swal.fire({
                icon: 'warning',
                title: 'Đã đủ 8 thành viên',
                text: 'Đã chọn đủ 2 Đội trưởng và 8 tuyển thủ.',
                ...SWAL_THEME
            });
            return;
        }
        selectedCaptainMembers.push(id);
    }
    renderCaptainPlayers();
}

function clearCaptainSelection() {
    selectedCaptains = [];
    selectedCaptainMembers = [];
    renderCaptainPlayers();
}

async function handleCreateTeamsWithCaptains() {
    if (selectedCaptains.length !== 2) {
        Swal.fire({
            icon: 'info',
            title: 'Chọn 2 Đội Trưởng',
            text: 'Vui lòng chọn chính xác 2 Đội trưởng trước.',
            ...SWAL_THEME
        });
        return;
    }
    if (selectedCaptainMembers.length !== 8) {
        Swal.fire({
            icon: 'info',
            title: 'Chọn đủ 8 Thành Viên',
            text: `Bạn mới chọn ${selectedCaptainMembers.length}/8 thành viên. Cần đủ 8 người.`,
            ...SWAL_THEME
        });
        return;
    }

    const balanceMode = document.getElementById('captain-matchmaking-mode')?.value || document.getElementById('matchmaking-mode')?.value || 'composite';

    try {
        const res = await fetch('/api/create_teams_with_captains', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                captain1: selectedCaptains[0],
                captain2: selectedCaptains[1],
                remaining_players: selectedCaptainMembers,
                balance_mode: balanceMode
            })
        });
        const data = await res.json();
        if (data.success) {
            switchTab('matchmaker');
            selectedMatchmaker = [...selectedCaptains, ...selectedCaptainMembers];
            currentTeamsResult = data;
            displayTeamsResult(data);
        } else {
            Swal.fire({
                icon: 'error',
                title: 'Lỗi',
                text: data.error,
                ...SWAL_THEME
            });
        }
    } catch (err) {
        console.error("Lỗi chia đội captains:", err);
    }
}

// ==========================================
// TAB 3: PLAYER MANAGEMENT (CRUD)
// ==========================================
function renderAdminPlayers() {
    const container = document.getElementById('admin-players-container');
    if (!container) return;
    container.innerHTML = '';

    const query = (document.getElementById('admin-player-search')?.value || '').toLowerCase().trim();
    const sortBy = document.getElementById('admin-sort-by')?.value || 'skill';

    let list = [...allPlayers];

    if (sortBy === 'elo') list.sort((a, b) => b.hidden_elo - a.hidden_elo);
    if (sortBy === 'skill') list.sort((a, b) => b.skill - a.skill);
    if (sortBy === 'form') list.sort((a, b) => (b.form?.score || 0) - (a.form?.score || 0));
    if (sortBy === 'winrate') list.sort((a, b) => b.winrate - a.winrate);
    if (sortBy === 'name') list.sort((a, b) => a.nickname.localeCompare(b.nickname));

    list.forEach(p => {
        if (query && !p.id.toLowerCase().includes(query) && !p.nickname.toLowerCase().includes(query)) {
            return;
        }

        const card = document.createElement('div');
        card.className = "bg-white border border-slate-200 hover:border-slate-300 p-4 rounded-2xl flex flex-col justify-between space-y-4 transition shadow-xs hover:shadow-sm";

        const formLabel = p.form?.label || 'Tân binh';
        const formIcon = p.form?.icon || '🌱';
        const formScore = p.form?.score || 5.0;

        card.innerHTML = `
            <div>
                <div class="flex items-start justify-between gap-3">
                    <div class="flex items-center gap-3">
                        <img src="${p.avatar}" class="w-12 h-12 rounded-xl object-cover bg-slate-100 border border-slate-200" alt="${p.nickname}">
                        <div>
                            <h4 class="font-bold font-heading text-sm text-slate-900">${p.nickname}</h4>
                            <span class="text-xs text-slate-400">@${p.id}</span>
                        </div>
                    </div>
                    <div class="text-right">
                        <span class="px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 font-black text-xs border border-indigo-200">
                            Elo ${Math.round(p.hidden_elo)}
                        </span>
                    </div>
                </div>

                <!-- 4 Stats Sliders Display (1 - 10) -->
                <div class="grid grid-cols-2 gap-2 mt-4 text-[11px]">
                    <div class="bg-slate-50 p-2 rounded-lg border border-slate-100">
                        <span class="text-slate-500">Kỹ năng:</span>
                        <b class="text-amber-600 ml-1 font-bold">${p.skill}/10</b>
                    </div>
                    <div class="bg-slate-50 p-2 rounded-lg border border-slate-100">
                        <span class="text-slate-500">Bể tướng:</span>
                        <b class="text-blue-600 ml-1 font-bold">${p.champion_pool}/10</b>
                    </div>
                    <div class="bg-slate-50 p-2 rounded-lg border border-slate-100">
                        <span class="text-slate-500">Flex lane:</span>
                        <b class="text-purple-600 ml-1 font-bold">${p.flex_lane}/10</b>
                    </div>
                    <div class="bg-slate-50 p-2 rounded-lg border border-slate-100">
                        <span class="text-slate-500">Ổn định:</span>
                        <b class="text-emerald-600 ml-1 font-bold">${p.consistency}/10</b>
                    </div>
                </div>

                <!-- Form Display (Auto) -->
                <div class="mt-3 p-2 rounded-xl bg-slate-50 border border-slate-100 flex items-center justify-between text-xs">
                    <span class="text-slate-500">Phong độ:</span>
                    <span class="font-bold text-indigo-700 flex items-center gap-1">
                        <span>${formIcon}</span>
                        <span>${formScore}/10 (${formLabel})</span>
                    </span>
                </div>

                <!-- Match Stats -->
                <div class="flex items-center justify-between text-[11px] text-slate-500 mt-2 px-1">
                    <span>Số trận: <b class="text-slate-800">${p.matches}</b></span>
                    <span>Tỷ lệ thắng: <b class="text-emerald-600">${p.winrate}%</b></span>
                </div>
            </div>

            <!-- Card Actions -->
            <div class="flex items-center gap-2 pt-2 border-t border-slate-100">
                <button onclick="openPlayerModal('edit', '${p.id}')" class="flex-1 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition flex items-center justify-center gap-1.5">
                    <i class="fa-solid fa-pen-to-square"></i>
                    <span>Sửa</span>
                </button>
                <button onclick="confirmDeletePlayer('${p.id}')" class="py-1.5 px-3 rounded-xl bg-rose-50 hover:bg-rose-100 border border-rose-200 text-rose-600 text-xs font-semibold transition">
                    <i class="fa-solid fa-trash-can"></i>
                </button>
            </div>
        `;
        container.appendChild(card);
    });
}

function filterAdminPlayers() {
    renderAdminPlayers();
}

function openPlayerModal(mode, playerId = null) {
    const modal = document.getElementById('player-modal');
    modal.classList.remove('hidden');

    const modeInput = document.getElementById('form-mode');
    const title = document.getElementById('modal-player-title');
    const idInput = document.getElementById('form-id');

    modeInput.value = mode;

    if (mode === 'add') {
        title.innerHTML = '<i class="fa-solid fa-user-plus text-indigo-600"></i><span>Thêm Tuyển Thủ Mới</span>';
        idInput.disabled = false;
        idInput.value = '';
        document.getElementById('form-nickname').value = '';
        document.getElementById('form-avatar').value = '';
        document.getElementById('form-avatar-preview').src = 'https://api.dicebear.com/7.x/bottts/svg?seed=new';
        document.getElementById('form-skill').value = 5.0;
        document.getElementById('val-skill').innerText = '5.0';
        document.getElementById('form-pool').value = 5.0;
        document.getElementById('val-pool').innerText = '5.0';
        document.getElementById('form-flex').value = 5.0;
        document.getElementById('val-flex').innerText = '5.0';
        document.getElementById('form-consist').value = 5.0;
        document.getElementById('val-consist').innerText = '5.0';
        document.getElementById('form-display-badge').innerText = '🌱 Tân Binh (Auto)';
    } else {
        title.innerHTML = '<i class="fa-solid fa-user-pen text-indigo-600"></i><span>Chỉnh Sửa Tuyển Thủ</span>';
        idInput.disabled = true;
        const p = allPlayers.find(x => x.id === playerId);
        if (!p) return;

        idInput.value = p.id;
        document.getElementById('form-nickname').value = p.nickname;
        document.getElementById('form-avatar').value = p.avatar;
        document.getElementById('form-avatar-preview').src = p.avatar;

        document.getElementById('form-skill').value = p.skill;
        document.getElementById('val-skill').innerText = p.skill;

        document.getElementById('form-pool').value = p.champion_pool;
        document.getElementById('val-pool').innerText = p.champion_pool;

        document.getElementById('form-flex').value = p.flex_lane;
        document.getElementById('val-flex').innerText = p.flex_lane;

        document.getElementById('form-consist').value = p.consistency;
        document.getElementById('val-consist').innerText = p.consistency;

        const formLabel = p.form?.label || 'Tân binh';
        const formIcon = p.form?.icon || '🌱';
        const formScore = p.form?.score || 5.0;
        document.getElementById('form-display-badge').innerText = `${formIcon} ${formScore}/10 (${formLabel})`;
    }
}

function closePlayerModal() {
    document.getElementById('player-modal').classList.add('hidden');
}

function updateAvatarPreview(url) {
    if (url) {
        document.getElementById('form-avatar-preview').src = url;
    }
}

function randomizeAvatar() {
    const randomSeed = Math.random().toString(36).substring(2, 9);
    const newAvatar = `https://api.dicebear.com/7.x/bottts/svg?seed=${randomSeed}`;
    document.getElementById('form-avatar').value = newAvatar;
    document.getElementById('form-avatar-preview').src = newAvatar;
}

let isSavingPlayer = false;

async function handleSavePlayer(e) {
    e.preventDefault();
    if (isSavingPlayer) return;

    const mode = document.getElementById('form-mode').value;
    const id = document.getElementById('form-id').value.trim().toLowerCase();
    const nickname = document.getElementById('form-nickname').value.trim();
    const avatar = document.getElementById('form-avatar').value.trim() || `https://api.dicebear.com/7.x/bottts/svg?seed=${id}`;
    const skill = parseFloat(document.getElementById('form-skill').value);
    const champion_pool = parseFloat(document.getElementById('form-pool').value);
    const flex_lane = parseFloat(document.getElementById('form-flex').value);
    const consistency = parseFloat(document.getElementById('form-consist').value);

    if (!id) {
        Swal.fire({
            icon: 'warning',
            title: 'Thiếu ID',
            text: 'Vui lòng nhập ID định danh cho tuyển thủ.',
            ...SWAL_THEME
        });
        return;
    }

    const payload = {
        id,
        nickname: nickname || id.toUpperCase(),
        avatar,
        skill,
        champion_pool,
        flex_lane,
        consistency
    };

    const saveBtn = document.getElementById('btn-save-player') || document.querySelector('#player-form button[type="submit"]');
    const origBtnHtml = saveBtn ? saveBtn.innerHTML : 'Lưu Thông Tin';
    if (saveBtn) {
        saveBtn.disabled = true;
        saveBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Đang lưu...';
        saveBtn.classList.add('opacity-70', 'cursor-not-allowed');
    }

    const formInputs = document.querySelectorAll('#player-form input');
    formInputs.forEach(input => input.disabled = true);

    isSavingPlayer = true;

    try {
        let res;
        if (mode === 'add') {
            res = await fetch('/api/players', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        } else {
            res = await fetch(`/api/players/${encodeURIComponent(id)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        }

        const data = await res.json();
        if (data.success) {
            closePlayerModal();
            Swal.fire({
                icon: 'success',
                title: 'Thành công',
                text: data.message,
                timer: 1500,
                showConfirmButton: false,
                ...SWAL_THEME
            });
            await loadAllPlayers();
            if (pendingSlotAssignment) {
                const { team, slotIdx } = pendingSlotAssignment;
                if (team === 'team1') simTeam1[slotIdx] = id;
                if (team === 'team2') simTeam2[slotIdx] = id;
                if (simUnmatchedSlotInfo[team]) delete simUnmatchedSlotInfo[team][slotIdx];
                pendingSlotAssignment = null;
                renderSimulationBoard();
            }
        } else {
            Swal.fire({
                icon: 'error',
                title: 'Lỗi',
                text: data.error,
                ...SWAL_THEME
            });
        }
    } catch (err) {
        console.error("Lỗi lưu người chơi:", err);
        Swal.fire({
            icon: 'error',
            title: 'Lỗi kết nối',
            text: 'Không thể kết nối đến máy chủ. Vui lòng thử lại.',
            ...SWAL_THEME
        });
    } finally {
        isSavingPlayer = false;
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.innerHTML = origBtnHtml;
            saveBtn.classList.remove('opacity-70', 'cursor-not-allowed');
        }
        formInputs.forEach(input => {
            if (input.id === 'form-id' && mode === 'edit') {
                input.disabled = true;
            } else {
                input.disabled = false;
            }
        });
    }
}

async function confirmDeletePlayer(id) {
    const confirm = await Swal.fire({
        title: `Xóa tuyển thủ @${id}?`,
        text: 'Hồ sơ tuyển thủ sẽ bị xóa khỏi danh sách quản lý.',
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: 'Đồng ý xóa',
        cancelButtonText: 'Hủy',
        confirmButtonColor: '#ef4444',
        cancelButtonColor: '#94a3b8',
        background: '#ffffff',
        color: '#0f172a'
    });

    if (!confirm.isConfirmed) return;

    try {
        const res = await fetch(`/api/players/${id}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            Swal.fire({
                icon: 'success',
                title: 'Đã xóa',
                text: data.message,
                timer: 1500,
                showConfirmButton: false,
                ...SWAL_THEME
            });
            await loadAllPlayers();
        } else {
            Swal.fire({
                icon: 'error',
                title: 'Lỗi',
                text: data.error,
                ...SWAL_THEME
            });
        }
    } catch (err) {
        console.error("Lỗi xóa:", err);
    }
}

// ==========================================
// TAB 4: LEADERBOARD (BẢNG XẾP HẠNG)
// ==========================================
function renderLeaderboard() {
    const tbody = document.getElementById('leaderboard-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    // Chỉ hiển thị tuyển thủ đã thi đấu ít nhất 1 trận (loại bỏ người chơi chưa đánh trận nào)
    const rankedPlayers = allPlayers
        .filter(p => (p.matches || 0) > 0)
        .sort((a, b) => b.hidden_elo - a.hidden_elo);

    if (rankedPlayers.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center py-12 text-slate-400">
                    <div class="flex flex-col items-center justify-center gap-2">
                        <i class="fa-solid fa-trophy text-slate-300 text-3xl"></i>
                        <span class="font-bold text-slate-600 text-sm">Chưa có tuyển thủ nào đủ điều kiện xếp hạng Elo</span>
                        <p class="text-xs text-slate-400 max-w-md">
                            Bảng xếp hạng chỉ hiển thị tuyển thủ đã tham gia ít nhất 1 trận đấu thực tế. Hãy ghi nhận kết quả trận đấu trong tab <b>Mô Phỏng 5vs5</b> để xuất hiện trên BXH!
                        </p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    rankedPlayers.forEach((p, idx) => {
        const rank = idx + 1;
        let rankBadge = `<span class="font-bold text-slate-500">#${rank}</span>`;
        if (rank === 1) rankBadge = `<span class="w-7 h-7 rounded-full bg-amber-400 text-slate-900 font-black flex items-center justify-center mx-auto shadow-xs">1</span>`;
        if (rank === 2) rankBadge = `<span class="w-7 h-7 rounded-full bg-slate-300 text-slate-900 font-black flex items-center justify-center mx-auto shadow-xs">2</span>`;
        if (rank === 3) rankBadge = `<span class="w-7 h-7 rounded-full bg-amber-700 text-white font-black flex items-center justify-center mx-auto shadow-xs">3</span>`;

        const recentBadges = (p.form?.recent_5 || []).map(r => {
            if (r === 'W') return `<span class="w-5 h-5 rounded-md bg-emerald-100 text-emerald-800 border border-emerald-200 text-[10px] font-bold inline-flex items-center justify-center">W</span>`;
            return `<span class="w-5 h-5 rounded-md bg-rose-100 text-rose-800 border border-rose-200 text-[10px] font-bold inline-flex items-center justify-center">L</span>`;
        }).join('');

        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-50 transition";
        tr.innerHTML = `
            <td class="py-3 px-4 text-center">${rankBadge}</td>
            <td class="py-3 px-4">
                <div class="flex items-center gap-3">
                    <img src="${p.avatar}" class="w-9 h-9 rounded-xl bg-slate-100 border border-slate-200 object-cover" alt="${p.nickname}">
                    <div>
                        <span class="font-bold text-slate-900 block">${p.nickname}</span>
                        <span class="text-xs text-slate-400">@${p.id}</span>
                    </div>
                </div>
            </td>
            <td class="py-3 px-4 text-center font-extrabold text-indigo-600">
                ${Math.round(p.hidden_elo)}
            </td>
            <td class="py-3 px-4 text-center">
                <span class="px-2.5 py-0.5 rounded-lg bg-slate-100 font-bold text-amber-700 border border-slate-200">
                    ${p.stats_ovr || p.skill}
                </span>
            </td>
            <td class="py-3 px-4 text-center">
                <span class="font-bold text-slate-800">
                    ${p.form?.icon || '🌱'} ${p.form?.score || 5.0}
                </span>
            </td>
            <td class="py-3 px-4 text-center font-bold ${p.winrate >= 50 ? 'text-emerald-600' : 'text-slate-500'}">
                ${p.winrate}%
            </td>
            <td class="py-3 px-4 text-center text-xs text-slate-500">
                ${p.matches} (<span class="text-emerald-600">${p.wins}</span> / <span class="text-rose-600">${p.losses}</span>)
            </td>
            <td class="py-3 px-4 text-center">
                <div class="flex items-center justify-center gap-1">
                    ${recentBadges || '<span class="text-slate-400 text-xs">-</span>'}
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function loadLeaderboard() {
    loadAllPlayers();
}

// ==========================================
// TAB 5: SYNERGIES (CẶP BÀI TRÙNG & TAM TẤU)
// ==========================================
let currentSynergyView = 'duo';

function switchSynergyView(view) {
    currentSynergyView = view;
    const btnDuo = document.getElementById('btn-synergy-duo');
    const btnTrio = document.getElementById('btn-synergy-trio');
    const containerDuo = document.getElementById('synergies-duo-container');
    const containerTrio = document.getElementById('synergies-trio-container');

    if (!btnDuo || !btnTrio || !containerDuo || !containerTrio) return;

    if (view === 'duo') {
        btnDuo.className = "px-3.5 py-1.5 rounded-xl bg-white text-indigo-700 shadow-xs transition";
        btnTrio.className = "px-3.5 py-1.5 rounded-xl text-slate-600 hover:text-slate-900 transition";
        containerDuo.classList.remove('hidden');
        containerTrio.classList.add('hidden');
    } else {
        btnTrio.className = "px-3.5 py-1.5 rounded-xl bg-white text-indigo-700 shadow-xs transition";
        btnDuo.className = "px-3.5 py-1.5 rounded-xl text-slate-600 hover:text-slate-900 transition";
        containerTrio.classList.remove('hidden');
        containerDuo.classList.add('hidden');
    }
}

async function loadSynergies() {
    const duoContainer = document.getElementById('synergies-duo-container');
    const trioContainer = document.getElementById('synergies-trio-container');
    if (!duoContainer || !trioContainer) return;

    duoContainer.innerHTML = '<div class="col-span-3 text-center text-slate-400 py-8"><i class="fa-solid fa-spinner fa-spin mr-2"></i> Đang tải thống kê cặp đôi...</div>';
    trioContainer.innerHTML = '<div class="col-span-3 text-center text-slate-400 py-8"><i class="fa-solid fa-spinner fa-spin mr-2"></i> Đang tải thống kê tam tấu...</div>';

    try {
        const res = await fetch('/api/synergies');
        const data = await res.json();
        if (data.success) {
            // Render Duo
            duoContainer.innerHTML = '';
            const pairs = data.pairs || data.synergies || [];
            if (pairs.length === 0) {
                duoContainer.innerHTML = '<div class="col-span-3 text-center text-slate-500 py-8">Chưa có dữ liệu cặp đôi nào trong các trận mới (cần từ 2 trận chung đội). Hãy ghi nhận kết quả trận đấu để xem thống kê ăn ý!</div>';
            } else {
                pairs.slice(0, 30).forEach(pair => {
                    const p1 = allPlayers.find(p => p.id === pair.p1) || { nickname: pair.p1, avatar: `https://api.dicebear.com/7.x/bottts/svg?seed=${pair.p1}` };
                    const p2 = allPlayers.find(p => p.id === pair.p2) || { nickname: pair.p2, avatar: `https://api.dicebear.com/7.x/bottts/svg?seed=${pair.p2}` };

                    const card = document.createElement('div');
                    card.className = "bg-white border border-slate-200 p-4 rounded-2xl flex items-center justify-between shadow-xs hover:border-indigo-300 transition";
                    card.innerHTML = `
                        <div class="flex items-center gap-2">
                            <img src="${p1.avatar}" class="w-10 h-10 rounded-xl bg-slate-100 border border-slate-200 object-cover">
                            <span class="text-slate-400 font-bold text-xs">+</span>
                            <img src="${p2.avatar}" class="w-10 h-10 rounded-xl bg-slate-100 border border-slate-200 object-cover">
                            <div class="ml-2">
                                <h5 class="font-bold font-heading text-xs text-slate-900">${p1.nickname} & ${p2.nickname}</h5>
                                <span class="text-[10px] text-slate-500">${pair.matches} trận cùng team</span>
                            </div>
                        </div>
                        <div class="text-right">
                            <span class="text-sm font-black text-indigo-600 font-heading">${pair.winrate}%</span>
                            <div class="text-[10px] text-slate-400">${pair.wins} thắng</div>
                        </div>
                    `;
                    duoContainer.appendChild(card);
                });
            }

            // Render Trio
            trioContainer.innerHTML = '';
            const trios = data.trios || [];
            if (trios.length === 0) {
                trioContainer.innerHTML = '<div class="col-span-3 text-center text-slate-500 py-8">Chưa có dữ liệu bộ ba nào trong các trận mới (cần từ 2 trận cùng 3 người). Hãy ghi nhận kết quả trận đấu để xem thống kê tam tấu!</div>';
            } else {
                trios.slice(0, 30).forEach(trio => {
                    const p1 = allPlayers.find(p => p.id === trio.p1) || { nickname: trio.p1, avatar: `https://api.dicebear.com/7.x/bottts/svg?seed=${trio.p1}` };
                    const p2 = allPlayers.find(p => p.id === trio.p2) || { nickname: trio.p2, avatar: `https://api.dicebear.com/7.x/bottts/svg?seed=${trio.p2}` };
                    const p3 = allPlayers.find(p => p.id === trio.p3) || { nickname: trio.p3, avatar: `https://api.dicebear.com/7.x/bottts/svg?seed=${trio.p3}` };

                    const card = document.createElement('div');
                    card.className = "bg-white border border-slate-200 p-4 rounded-2xl flex items-center justify-between shadow-xs hover:border-amber-300 transition";
                    card.innerHTML = `
                        <div class="flex items-center gap-1.5">
                            <img src="${p1.avatar}" class="w-9 h-9 rounded-xl bg-slate-100 border border-slate-200 object-cover">
                            <img src="${p2.avatar}" class="w-9 h-9 rounded-xl bg-slate-100 border border-slate-200 object-cover">
                            <img src="${p3.avatar}" class="w-9 h-9 rounded-xl bg-slate-100 border border-slate-200 object-cover">
                            <div class="ml-2">
                                <h5 class="font-bold font-heading text-xs text-slate-900 truncate max-w-[120px] sm:max-w-[150px]">${p1.nickname}, ${p2.nickname}, ${p3.nickname}</h5>
                                <span class="text-[10px] text-slate-500">${trio.matches} trận cùng team</span>
                            </div>
                        </div>
                        <div class="text-right">
                            <span class="text-sm font-black text-amber-600 font-heading">${trio.winrate}%</span>
                            <div class="text-[10px] text-slate-400">${trio.wins} thắng</div>
                        </div>
                    `;
                    trioContainer.appendChild(card);
                });
            }
        }
    } catch (err) {
        console.error("Lỗi synergies:", err);
    }
}

// ==========================================
// TAB 6: SETTINGS & SUPABASE (CÀI ĐẶT)
// ==========================================
async function checkSupabaseStatus() {
    const badge = document.getElementById('supabase-status-badge');
    const pCount = document.getElementById('supabase-players-count');
    const mCount = document.getElementById('supabase-matches-count');
    const modeText = document.getElementById('supabase-mode-text');

    if (!badge) return;

    try {
        const res = await fetch('/api/supabase/status');
        const data = await res.json();

        if (data.success && data.connected) {
            badge.innerHTML = '<i class="fa-solid fa-circle-check text-emerald-600"></i> <span class="text-emerald-800">Online • Sẵn sàng</span>';
            badge.className = "px-3 py-1 rounded-full bg-emerald-100 text-xs font-bold font-heading flex items-center gap-1.5";
            if (pCount) pCount.innerText = `${data.players_count} tuyển thủ`;
            if (mCount) mCount.innerText = `${data.matches_count} trận đấu`;
            if (modeText) modeText.innerText = (data.storage_mode || 'SUPABASE').toUpperCase();
        } else {
            badge.innerHTML = '<i class="fa-solid fa-circle-exclamation text-amber-600"></i> <span class="text-amber-800">Chưa kết nối bảng</span>';
            badge.className = "px-3 py-1 rounded-full bg-amber-100 text-xs font-bold font-heading flex items-center gap-1.5";
            if (pCount) pCount.innerText = "0 tuyển thủ";
            if (mCount) mCount.innerText = "0 trận";
            if (modeText) modeText.innerText = "FALLBACK LOCAL";
        }
    } catch (err) {
        console.error("Lỗi kiểm tra Supabase:", err);
        badge.innerHTML = '<i class="fa-solid fa-triangle-exclamation text-rose-600"></i> <span class="text-rose-800">Offline (Dùng Local)</span>';
        badge.className = "px-3 py-1 rounded-full bg-rose-100 text-xs font-bold font-heading flex items-center gap-1.5";
    }
}

async function handleSyncToSupabase() {
    const btn = document.getElementById('btn-sync-supabase');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Đang đồng bộ...';
    }

    try {
        const res = await fetch('/api/supabase/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await res.json();

        if (data.success) {
            Swal.fire({
                icon: 'success',
                title: 'Đồng bộ thành công!',
                text: data.message || `Đã đồng bộ ${data.synced_count} tuyển thủ lên Supabase.`,
                ...SWAL_THEME
            });
            await checkSupabaseStatus();
            await loadAllPlayers();
        } else {
            Swal.fire({
                icon: 'warning',
                title: 'Chưa thể đồng bộ',
                text: data.error || 'Vui lòng kiểm tra lại bảng dữ liệu trên Supabase.',
                ...SWAL_THEME
            });
        }
    } catch (err) {
        Swal.fire({
            icon: 'error',
            title: 'Lỗi đồng bộ',
            text: err.message,
            ...SWAL_THEME
        });
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up mr-1.5"></i> Đồng Bộ Tuyển Thủ Lên Supabase';
        }
    }
}

function saveGeminiApiKey() {
    const key = document.getElementById('input-gemini-key').value.trim();
    if (!key) {
        Swal.fire({ icon: 'info', title: 'Nhập API Key', text: 'Vui lòng dán Gemini API Key.', ...SWAL_THEME });
        return;
    }
    localStorage.setItem('fbcs_gemini_key', key);
    Swal.fire({ icon: 'success', title: 'Đã lưu', text: 'Gemini API Key đã được lưu thành công trên trình duyệt.', ...SWAL_THEME });
}

// ==========================================
// TAB: SIMULATION 5VS5 (MÔ PHỎNG & TRẬN ĐẤU TÙY CHỈNH)
// ==========================================
function renderSimulationBoard() {
    const t1Container = document.getElementById('sim-t1-slots');
    const t2Container = document.getElementById('sim-t2-slots');
    if (!t1Container || !t2Container) return;

    t1Container.innerHTML = '';
    t2Container.innerHTML = '';

    for (let i = 0; i < 5; i++) {
        t1Container.appendChild(createSimulationSlotCard('team1', i));
        t2Container.appendChild(createSimulationSlotCard('team2', i));
    }

    updateSimulationLiveStats();
}

let activeSlotDropdown = null;

function createSimulationSlotCard(team, slotIdx) {
    const isTeam1 = team === 'team1';
    const currentPid = isTeam1 ? simTeam1[slotIdx] : simTeam2[slotIdx];
    const playerObj = allPlayers.find(p => p.id === currentPid);
    const unmatchedRawName = simUnmatchedSlotInfo[team]?.[slotIdx] || null;

    const div = document.createElement('div');
    div.className = `relative flex items-center gap-2.5 p-2.5 rounded-2xl bg-white border ${
        isTeam1 ? 'border-blue-200/90 hover:border-blue-300' : 'border-rose-200/90 hover:border-rose-300'
    } shadow-xs transition`;

    // Slot badge
    const badgeColor = isTeam1 ? 'bg-blue-100 text-blue-800' : 'bg-rose-100 text-rose-800';
    const avatarSrc = playerObj ? playerObj.avatar : (unmatchedRawName ? `https://api.dicebear.com/7.x/bottts/svg?seed=${encodeURIComponent(unmatchedRawName)}` : 'https://api.dicebear.com/7.x/bottts/svg?seed=empty');

    const escapedRawName = unmatchedRawName ? unmatchedRawName.replace(/'/g, "\\'") : '';

    // Label on combobox trigger
    let displayLabel = `-- Chọn tuyển thủ (Slot ${slotIdx + 1}) --`;
    let labelStyle = 'text-slate-400 font-normal';
    if (playerObj) {
        displayLabel = `${playerObj.nickname} (Elo ${Math.round(playerObj.hidden_elo)})`;
        labelStyle = 'text-slate-900 font-bold';
    } else if (unmatchedRawName) {
        displayLabel = `⚠️ ${unmatchedRawName} (Chưa có trong data)`;
        labelStyle = 'text-amber-800 font-bold';
    }

    div.innerHTML = `
        <span class="w-7 h-7 rounded-xl ${badgeColor} text-xs font-black flex items-center justify-center shadow-2xs flex-shrink-0">${slotIdx + 1}</span>
        <div class="relative flex-shrink-0">
            <img src="${avatarSrc}" class="w-8 h-8 rounded-xl object-cover bg-slate-100 border border-slate-200" alt="Avatar">
            ${playerObj?.form?.icon ? `<span class="absolute -bottom-1 -right-1 text-[9px]">${playerObj.form.icon}</span>` : ''}
        </div>

        <!-- Searchable Combobox Container -->
        <div class="relative flex-1 min-w-0" id="sim-combobox-${team}-${slotIdx}">
            <button type="button" onclick="toggleSlotDropdown('${team}', ${slotIdx}, event)" class="w-full bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-xl px-2.5 py-1.5 text-xs flex items-center justify-between gap-1.5 transition text-left focus:outline-none focus:border-indigo-500">
                <span class="truncate ${labelStyle}">${displayLabel}</span>
                <i class="fa-solid fa-chevron-down text-slate-400 text-[10px] flex-shrink-0"></i>
            </button>

            <!-- Dropdown Menu Panel -->
            <div id="sim-menu-${team}-${slotIdx}" class="hidden absolute left-0 top-full mt-1.5 w-full min-w-[280px] max-w-[340px] bg-white border border-slate-200 rounded-2xl shadow-xl z-50 p-2.5 space-y-2">
                <!-- Search Input with icon -->
                <div class="relative">
                    <i class="fa-solid fa-search absolute left-3 top-2.5 text-slate-400 text-[11px]"></i>
                    <input type="text" id="sim-search-${team}-${slotIdx}" oninput="filterSlotDropdown('${team}', ${slotIdx}, this.value)" placeholder="Gõ tên tìm tuyển thủ..." class="w-full bg-slate-50 border border-slate-200 rounded-xl pl-8 pr-2.5 py-1.5 text-xs text-slate-800 focus:outline-none focus:border-indigo-500 focus:bg-white transition" onclick="event.stopPropagation()">
                </div>

                <!-- Scrollable Items List -->
                <div id="sim-list-${team}-${slotIdx}" class="max-h-52 overflow-y-auto custom-scrollbar space-y-1 text-xs">
                    <!-- Populated dynamically via JS -->
                </div>
            </div>
        </div>

        <!-- Slot Actions -->
        <div class="flex items-center gap-1 flex-shrink-0">
            ${unmatchedRawName && !playerObj ? `
                <button type="button" onclick="quickCreatePlayerFromSlot('${team}', ${slotIdx}, '${escapedRawName}')" class="px-2 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-[11px] font-bold shadow-xs transition flex items-center gap-1">
                    <i class="fa-solid fa-user-plus text-[10px]"></i>
                    <span>+ Tạo</span>
                </button>
            ` : `
                <button type="button" onclick="quickCreatePlayerFromSlot('${team}', ${slotIdx})" title="Thêm tuyển thủ mới" class="p-1.5 text-slate-400 hover:text-indigo-600 rounded-lg hover:bg-slate-100 transition">
                    <i class="fa-solid fa-user-plus text-xs"></i>
                </button>
            `}
            <button type="button" onclick="clearSimulationSlot('${team}', ${slotIdx})" title="Xóa slot" class="p-1.5 text-slate-400 hover:text-rose-600 rounded-lg hover:bg-slate-100 transition">
                <i class="fa-solid fa-xmark text-xs"></i>
            </button>
        </div>
    `;

    return div;
}

function toggleSlotDropdown(team, slotIdx, event) {
    if (event) event.stopPropagation();

    if (activeSlotDropdown && activeSlotDropdown.team === team && activeSlotDropdown.slotIdx === slotIdx) {
        closeAllSlotDropdowns();
        return;
    }

    closeAllSlotDropdowns();

    const menu = document.getElementById(`sim-menu-${team}-${slotIdx}`);
    if (!menu) return;

    menu.classList.remove('hidden');
    activeSlotDropdown = { team, slotIdx };

    populateSlotDropdownList(team, slotIdx, '');

    const searchInput = document.getElementById(`sim-search-${team}-${slotIdx}`);
    if (searchInput) {
        searchInput.value = '';
        setTimeout(() => searchInput.focus(), 50);
    }
}

function closeAllSlotDropdowns() {
    document.querySelectorAll('[id^="sim-menu-"]').forEach(m => m.classList.add('hidden'));
    activeSlotDropdown = null;
}

function filterSlotDropdown(team, slotIdx, query) {
    populateSlotDropdownList(team, slotIdx, query);
}

function populateSlotDropdownList(team, slotIdx, query = '') {
    const listContainer = document.getElementById(`sim-list-${team}-${slotIdx}`);
    if (!listContainer) return;

    listContainer.innerHTML = '';
    const q = (query || '').trim().toLowerCase();
    const currentPid = team === 'team1' ? simTeam1[slotIdx] : simTeam2[slotIdx];
    const unmatchedRawName = simUnmatchedSlotInfo[team]?.[slotIdx] || null;

    // Option 1: Clear/Deselect Slot
    if (!q || '--'.includes(q) || 'bỏ chọn'.includes(q) || 'xóa'.includes(q)) {
        const clearItem = document.createElement('div');
        clearItem.className = 'flex items-center gap-2 p-1.5 rounded-xl text-slate-500 hover:bg-slate-100 cursor-pointer transition select-none';
        clearItem.innerHTML = `
            <i class="fa-solid fa-ban text-slate-400 w-5 text-center text-xs"></i>
            <span class="font-medium text-xs">-- Để trống slot này --</span>
        `;
        clearItem.onclick = (e) => {
            e.stopPropagation();
            onSimulationSlotChange(team, slotIdx, '');
            closeAllSlotDropdowns();
        };
        listContainer.appendChild(clearItem);
    }

    // Option 2: Unmatched raw name from OCR (if any)
    if (unmatchedRawName && (!q || unmatchedRawName.toLowerCase().includes(q))) {
        const unmatchedItem = document.createElement('div');
        unmatchedItem.className = 'flex items-center justify-between p-2 rounded-xl bg-amber-50/80 border border-amber-200/80 hover:bg-amber-100/80 cursor-pointer transition select-none';
        const escapedName = unmatchedRawName.replace(/'/g, "\\'");
        unmatchedItem.innerHTML = `
            <div class="flex items-center gap-2 truncate">
                <span class="text-amber-600 font-bold">⚠️</span>
                <span class="font-bold text-amber-900 truncate text-xs">${unmatchedRawName}</span>
            </div>
            <button type="button" onclick="event.stopPropagation(); quickCreatePlayerFromSlot('${team}', ${slotIdx}, '${escapedName}'); closeAllSlotDropdowns()" class="px-2 py-0.5 rounded-lg bg-amber-600 text-white font-bold text-[10px] shadow-2xs flex-shrink-0">
                + Tạo Mới
            </button>
        `;
        unmatchedItem.onclick = (e) => {
            e.stopPropagation();
            quickCreatePlayerFromSlot(team, slotIdx, unmatchedRawName);
            closeAllSlotDropdowns();
        };
        listContainer.appendChild(unmatchedItem);
    }

    // Option 3: Filtered Players List
    const sorted = [...allPlayers].sort((a, b) => a.nickname.localeCompare(b.nickname));
    const filtered = sorted.filter(p => {
        if (!q) return true;
        return p.nickname.toLowerCase().includes(q) || p.id.toLowerCase().includes(q);
    });

    filtered.forEach(p => {
        const isSelected = p.id === currentPid;
        const item = document.createElement('div');
        item.className = `flex items-center justify-between p-1.5 rounded-xl cursor-pointer transition select-none ${
            isSelected ? 'bg-indigo-50 text-indigo-900 font-bold border border-indigo-200' : 'hover:bg-slate-100 text-slate-700'
        }`;

        item.innerHTML = `
            <div class="flex items-center gap-2 min-w-0">
                <img src="${p.avatar}" class="w-6 h-6 rounded-lg object-cover bg-slate-100 border border-slate-200 flex-shrink-0">
                <span class="truncate font-semibold text-xs text-slate-900">${p.nickname}</span>
                <span class="text-[10px] text-slate-400">${p.form?.icon || ''}</span>
            </div>
            <div class="flex items-center gap-1.5 flex-shrink-0 ml-2">
                <span class="text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                    Elo ${Math.round(p.hidden_elo)}
                </span>
                ${isSelected ? '<i class="fa-solid fa-check text-indigo-600 text-xs"></i>' : ''}
            </div>
        `;

        item.onclick = (e) => {
            e.stopPropagation();
            onSimulationSlotChange(team, slotIdx, p.id);
            closeAllSlotDropdowns();
        };
        listContainer.appendChild(item);
    });

    // If query typed and no players found
    if (filtered.length === 0 && (!unmatchedRawName || !unmatchedRawName.toLowerCase().includes(q))) {
        const emptyNotice = document.createElement('div');
        emptyNotice.className = 'p-3 text-center text-slate-400 text-xs space-y-2';
        const escapedQ = query.replace(/'/g, "\\'");
        emptyNotice.innerHTML = `
            <p>Không tìm thấy tuyển thủ <b>"${query}"</b></p>
            <button type="button" onclick="event.stopPropagation(); quickCreatePlayerFromSlot('${team}', ${slotIdx}, '${escapedQ}'); closeAllSlotDropdowns()" class="w-full px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs shadow-xs transition flex items-center justify-center gap-1.5">
                <i class="fa-solid fa-user-plus text-[11px]"></i>
                <span>+ Thêm mới "${query}"</span>
            </button>
        `;
        listContainer.appendChild(emptyNotice);
    }
}

// Global click & escape listeners to close dropdowns
document.addEventListener('click', (e) => {
    if (!e.target.closest('[id^="sim-combobox-"]')) {
        closeAllSlotDropdowns();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeAllSlotDropdowns();
    }
});

function onSimulationSlotChange(team, slotIdx, selectedVal) {
    if (selectedVal && selectedVal.startsWith('__unmatched__:')) {
        const rawName = selectedVal.replace('__unmatched__:', '');
        quickCreatePlayerFromSlot(team, slotIdx, rawName);
        return;
    }

    if (!selectedVal) {
        if (team === 'team1') simTeam1[slotIdx] = null;
        else simTeam2[slotIdx] = null;
        if (simUnmatchedSlotInfo[team]) delete simUnmatchedSlotInfo[team][slotIdx];
        renderSimulationBoard();
        return;
    }

    // Check duplicate
    const currentTeam = team === 'team1' ? simTeam1 : simTeam2;
    const otherTeam = team === 'team1' ? simTeam2 : simTeam1;

    const inOtherSlotCurrent = currentTeam.findIndex((pid, idx) => idx !== slotIdx && pid === selectedVal);
    const inOtherSlotOther = otherTeam.findIndex(pid => pid === selectedVal);

    if (inOtherSlotCurrent > -1) {
        currentTeam[inOtherSlotCurrent] = null;
    }
    if (inOtherSlotOther > -1) {
        otherTeam[inOtherSlotOther] = null;
    }

    if (team === 'team1') simTeam1[slotIdx] = selectedVal;
    else simTeam2[slotIdx] = selectedVal;

    if (simUnmatchedSlotInfo[team]) delete simUnmatchedSlotInfo[team][slotIdx];

    renderSimulationBoard();
}

function clearSimulationSlot(team, slotIdx) {
    if (team === 'team1') simTeam1[slotIdx] = null;
    else simTeam2[slotIdx] = null;
    if (simUnmatchedSlotInfo[team]) delete simUnmatchedSlotInfo[team][slotIdx];
    renderSimulationBoard();
}

function clearSimulationSlots() {
    simTeam1 = [null, null, null, null, null];
    simTeam2 = [null, null, null, null, null];
    simUnmatchedSlotInfo = { team1: {}, team2: {} };
    renderSimulationBoard();
}

function quickCreatePlayerFromSlot(team, slotIdx, defaultNickname = '') {
    pendingSlotAssignment = { team, slotIdx };
    openPlayerModal('add');

    if (defaultNickname) {
        document.getElementById('form-nickname').value = defaultNickname;
        const cleanId = defaultNickname.toLowerCase().replace(/[^a-z0-9]/g, '') || `player_${Math.floor(Math.random() * 10000)}`;
        document.getElementById('form-id').value = cleanId;
        document.getElementById('form-avatar-preview').src = `https://api.dicebear.com/7.x/bottts/svg?seed=${cleanId}`;
    }
}

function updateSimulationLiveStats() {
    const t1Players = simTeam1.map(id => allPlayers.find(p => p.id === id)).filter(Boolean);
    const t2Players = simTeam2.map(id => allPlayers.find(p => p.id === id)).filter(Boolean);

    const t1TotalElo = Math.round(t1Players.reduce((sum, p) => sum + (p.hidden_elo || 1200), 0));
    const t2TotalElo = Math.round(t2Players.reduce((sum, p) => sum + (p.hidden_elo || 1200), 0));

    const t1AvgElo = t1Players.length ? Math.round(t1TotalElo / t1Players.length) : 0;
    const t2AvgElo = t2Players.length ? Math.round(t2TotalElo / t2Players.length) : 0;

    const diffElo = Math.abs(t1TotalElo - t2TotalElo);

    let prob1 = 50;
    let prob2 = 50;
    if (t1Players.length > 0 || t2Players.length > 0) {
        const d = (t1TotalElo - t2TotalElo) / (Math.max(t1Players.length, t2Players.length) || 5);
        prob1 = Math.round((1.0 / (1.0 + Math.pow(10.0, -d / 400.0))) * 100);
        prob1 = Math.max(5, Math.min(95, prob1));
        prob2 = 100 - prob1;
    }

    const t1TotalElem = document.getElementById('sim-t1-total-elo');
    if (t1TotalElem) t1TotalElem.innerText = t1TotalElo;
    const t1AvgElem = document.getElementById('sim-t1-avg-elo');
    if (t1AvgElem) t1AvgElem.innerText = t1AvgElo;

    const t2TotalElem = document.getElementById('sim-t2-total-elo');
    if (t2TotalElem) t2TotalElem.innerText = t2TotalElo;
    const t2AvgElem = document.getElementById('sim-t2-avg-elo');
    if (t2AvgElem) t2AvgElem.innerText = t2AvgElo;

    const diffElem = document.getElementById('sim-diff-elo');
    if (diffElem) diffElem.innerText = diffElo;
    const probText = document.getElementById('sim-prob-text');
    if (probText) probText.innerText = `${prob1}% - ${prob2}%`;

    const barT1 = document.getElementById('sim-bar-t1');
    const barT2 = document.getElementById('sim-bar-t2');
    if (barT1 && barT2) {
        barT1.style.width = `${prob1}%`;
        barT2.style.width = `${prob2}%`;
    }

    const t1CountBadge = document.getElementById('sim-t1-count-badge');
    if (t1CountBadge) t1CountBadge.innerText = `${t1Players.length}/5`;
    const t2CountBadge = document.getElementById('sim-t2-count-badge');
    if (t2CountBadge) t2CountBadge.innerText = `${t2Players.length}/5`;
}

function transferSimulationToMatchmaker() {
    const chosenIds = [...simTeam1, ...simTeam2].filter(Boolean);
    if (chosenIds.length === 0) {
        Swal.fire({
            icon: 'info',
            title: 'Chưa chọn tuyển thủ',
            text: 'Vui lòng chọn tuyển thủ vào các slot trước khi chuyển sang Chia Đội.',
            ...SWAL_THEME
        });
        return;
    }

    selectedMatchmaker = chosenIds.slice(0, 10);
    switchTab('matchmaker');
    renderMatchmakerPlayers();
    Swal.fire({
        icon: 'success',
        title: `Đã chọn ${selectedMatchmaker.length}/10 tuyển thủ`,
        text: 'Bạn có thể bấm Chia Đội Tối Ưu để thuật toán AI cân bằng lại 2 đội.',
        timer: 1500,
        showConfirmButton: false,
        ...SWAL_THEME
    });
}

function transferOcrToSimulation() {
    switchTab('simulation');
    renderSimulationBoard();
}

let isSubmittingSimulationMatch = false;
let currentMatchModalWinner = 'team1';
let currentScoreboardImageBase64 = null;
let currentScoreboardMimeType = 'image/jpeg';
let currentAiScoreboardAnalysis = null;

function submitSimulationWinner(winningTeam) {
    const t1Filled = simTeam1.filter(Boolean);
    const t2Filled = simTeam2.filter(Boolean);

    if (t1Filled.length !== 5 || t2Filled.length !== 5) {
        Swal.fire({
            icon: 'warning',
            title: 'Chưa đủ 10 người',
            text: `Vui lòng chọn đủ 5 người cho Đội Xanh (hiện có ${t1Filled.length}/5) và 5 người cho Đội Đỏ (hiện có ${t2Filled.length}/5).`,
            ...SWAL_THEME
        });
        return;
    }

    const allChosen = [...t1Filled, ...t2Filled];
    const uniqueChosen = new Set(allChosen);
    if (uniqueChosen.size !== 10) {
        Swal.fire({
            icon: 'error',
            title: 'Trùng lặp người chơi',
            text: 'Có tuyển thủ bị chọn nhiều hơn 1 lần giữa 2 đội. Vui lòng kiểm tra lại các slot.',
            ...SWAL_THEME
        });
        return;
    }

    openMatchResultModal(winningTeam);
}

function openMatchResultModal(winningTeam) {
    currentMatchModalWinner = winningTeam;
    const modal = document.getElementById('match-result-modal');
    if (!modal) return;

    const isTeam1 = winningTeam === 'team1';
    const winnerName = isTeam1 ? 'Đội Xanh (Team 1)' : 'Đội Đỏ (Team 2)';
    const loserName = isTeam1 ? 'Đội Đỏ (Team 2)' : 'Đội Xanh (Team 1)';
    const colorClass = isTeam1 ? 'text-blue-600' : 'text-rose-600';
    const bgBadgeClass = isTeam1 ? 'bg-blue-100 text-blue-600' : 'bg-rose-100 text-rose-600';

    const winnerNameElem = document.getElementById('modal-winner-team-name');
    if (winnerNameElem) {
        winnerNameElem.innerText = `${winnerName} Thắng`;
        winnerNameElem.className = `${colorClass} font-extrabold`;
    }

    const badgeIcon = document.getElementById('modal-winner-badge-icon');
    if (badgeIcon) {
        badgeIcon.className = `w-10 h-10 rounded-2xl ${bgBadgeClass} flex items-center justify-center text-lg font-bold shadow-xs`;
    }

    const stdWinnerName = document.getElementById('modal-std-winner-name');
    if (stdWinnerName) stdWinnerName.innerText = winnerName;
    const stdLoserName = document.getElementById('modal-std-loser-name');
    if (stdLoserName) stdLoserName.innerText = loserName;

    // Reset default view
    switchMatchResultTab('standard');
    removeScoreboardImage();

    modal.classList.remove('hidden');
}

function closeMatchResultModal() {
    const modal = document.getElementById('match-result-modal');
    if (modal) modal.classList.add('hidden');
    removeScoreboardImage();
}

function switchMatchResultTab(tabName) {
    const btnStd = document.getElementById('btn-tab-standard');
    const btnAi = document.getElementById('btn-tab-ai-scoreboard');
    const contentStd = document.getElementById('match-tab-content-standard');
    const contentAi = document.getElementById('match-tab-content-ai');

    if (!btnStd || !btnAi || !contentStd || !contentAi) return;

    if (tabName === 'standard') {
        btnStd.className = "py-2.5 px-3 rounded-xl text-xs font-bold font-heading transition flex items-center justify-center gap-1.5 bg-white text-indigo-700 shadow-xs";
        btnAi.className = "py-2.5 px-3 rounded-xl text-xs font-bold font-heading transition flex items-center justify-center gap-1.5 text-slate-600 hover:text-slate-900";
        contentStd.classList.remove('hidden');
        contentAi.classList.add('hidden');
    } else {
        btnAi.className = "py-2.5 px-3 rounded-xl text-xs font-bold font-heading transition flex items-center justify-center gap-1.5 bg-white text-purple-700 shadow-xs";
        btnStd.className = "py-2.5 px-3 rounded-xl text-xs font-bold font-heading transition flex items-center justify-center gap-1.5 text-slate-600 hover:text-slate-900";
        contentAi.classList.remove('hidden');
        contentStd.classList.add('hidden');
    }
}

async function confirmSaveStandardMatch() {
    if (isSubmittingSimulationMatch) return;
    isSubmittingSimulationMatch = true;

    const btn = document.getElementById('btn-confirm-save-standard');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Đang lưu...';
    }

    try {
        const res = await fetch('/api/update_match_result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                team1: simTeam1,
                team2: simTeam2,
                winner: currentMatchModalWinner,
                notes: 'Lưu kết quả chuẩn (không kèm ảnh)'
            })
        });

        const data = await res.json();
        if (!data.success) {
            throw new Error(data.error || 'Lỗi khi lưu kết quả trận đấu');
        }

        closeMatchResultModal();
        Swal.fire({
            icon: 'success',
            title: 'Ghi nhận thành công!',
            text: 'Elo ẩn và Phong độ của 10 tuyển thủ đã được cập nhật.',
            timer: 2000,
            showConfirmButton: false,
            ...SWAL_THEME
        });

        await loadAllPlayers();
        await loadStatus();
    } catch (err) {
        Swal.fire({
            icon: 'error',
            title: 'Lỗi lưu trận đấu',
            text: err.message,
            ...SWAL_THEME
        });
    } finally {
        isSubmittingSimulationMatch = false;
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-check"></i> <span>Xác Nhận Lưu Kết Quả Ngay</span>';
        }
    }
}

function handleScoreboardFileSelect(event) {
    const file = event.target.files?.[0];
    if (file) {
        handleScoreboardPastedFile(file);
    }
}

function handleScoreboardPastedFile(file) {
    if (!file.type.startsWith('image/')) {
        Swal.fire({ icon: 'warning', title: 'Tệp không hợp lệ', text: 'Vui lòng chọn hoặc dán file ảnh.', ...SWAL_THEME });
        return;
    }

    currentScoreboardMimeType = file.type;
    const reader = new FileReader();
    reader.onload = (e) => {
        currentScoreboardImageBase64 = e.target.result;

        const emptyArea = document.getElementById('scoreboard-dropzone-empty');
        const previewContainer = document.getElementById('scoreboard-preview-container');
        const previewImg = document.getElementById('scoreboard-preview-img');
        const fileName = document.getElementById('scoreboard-file-name');

        if (emptyArea) emptyArea.classList.add('hidden');
        if (previewContainer) previewContainer.classList.remove('hidden');
        if (previewImg) previewImg.src = currentScoreboardImageBase64;
        if (fileName) fileName.innerText = file.name || 'Ảnh bảng điểm vừa dán';

        // Tự động kích hoạt nút phân tích
        const runBtn = document.getElementById('btn-run-scoreboard-ai');
        if (runBtn) runBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeScoreboardImage() {
    currentScoreboardImageBase64 = null;
    currentAiScoreboardAnalysis = null;

    const fileInput = document.getElementById('scoreboard-file-input');
    if (fileInput) fileInput.value = '';

    const emptyArea = document.getElementById('scoreboard-dropzone-empty');
    const previewContainer = document.getElementById('scoreboard-preview-container');
    const resultsContainer = document.getElementById('scoreboard-ai-results');
    const loadingElem = document.getElementById('scoreboard-ai-loading');
    const controlsElem = document.getElementById('scoreboard-ai-controls');

    if (emptyArea) emptyArea.classList.remove('hidden');
    if (previewContainer) previewContainer.classList.add('hidden');
    if (resultsContainer) resultsContainer.classList.add('hidden');
    if (loadingElem) loadingElem.classList.add('hidden');
    if (controlsElem) controlsElem.classList.remove('hidden');
}

async function runScoreboardAiAnalysis() {
    if (!currentScoreboardImageBase64) {
        Swal.fire({
            icon: 'info',
            title: 'Chưa có ảnh',
            text: 'Vui lòng dán ảnh (Ctrl + V) hoặc bấm tải lên ảnh bảng điểm kết thúc trận đấu trước khi phân tích.',
            ...SWAL_THEME
        });
        return;
    }

    const loadingElem = document.getElementById('scoreboard-ai-loading');
    const controlsElem = document.getElementById('scoreboard-ai-controls');
    const resultsContainer = document.getElementById('scoreboard-ai-results');

    if (loadingElem) loadingElem.classList.remove('hidden');
    if (controlsElem) controlsElem.classList.add('hidden');
    if (resultsContainer) resultsContainer.classList.add('hidden');

    try {
        const apiKey = localStorage.getItem('fbcs_gemini_api_key') || '';
        const res = await fetch('/api/analyze_match_scoreboard', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: currentScoreboardImageBase64,
                mime_type: currentScoreboardMimeType,
                team1: simTeam1,
                team2: simTeam2,
                winner: currentMatchModalWinner,
                api_key: apiKey
            })
        });

        const data = await res.json();
        if (!data.success) {
            throw new Error(data.error || 'Không thể phân tích ảnh bảng điểm.');
        }

        currentAiScoreboardAnalysis = data;
        renderScoreboardAiResults(data);

    } catch (err) {
        Swal.fire({
            icon: 'error',
            title: 'Lỗi phân tích AI',
            text: err.message,
            ...SWAL_THEME
        });
        if (controlsElem) controlsElem.classList.remove('hidden');
    } finally {
        if (loadingElem) loadingElem.classList.add('hidden');
    }
}

function renderScoreboardAiResults(data) {
    const resultsContainer = document.getElementById('scoreboard-ai-results');
    const controlsElem = document.getElementById('scoreboard-ai-controls');
    const summaryElem = document.getElementById('ai-match-summary-text');
    const mvpBadge = document.getElementById('ai-mvp-badge');
    const svpBadge = document.getElementById('ai-svp-badge');
    const tbody = document.getElementById('scoreboard-players-table-body');

    if (!resultsContainer || !tbody) return;

    if (summaryElem) summaryElem.innerText = data.ai_summary || 'Đã phân tích thông số 10 tuyển thủ.';
    if (mvpBadge) mvpBadge.innerText = `👑 MVP: ${data.match_mvp || '-'}`;
    if (svpBadge) svpBadge.innerText = `⭐ SVP: ${data.match_svp || '-'}`;

    tbody.innerHTML = '';

    const list = data.players_analysis || [];
    list.forEach(p => {
        const playerObj = allPlayers.find(item => item.id === p.player_id) || {};
        const isTeam1 = p.team === 1;
        const isWinner = (isTeam1 && currentMatchModalWinner === 'team1') || (!isTeam1 && currentMatchModalWinner === 'team2');

        let tagBadge = `<span class="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 font-semibold text-[10px]">Tròn vai</span>`;
        if (p.performance_tag === 'MVP') {
            tagBadge = `<span class="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-black border border-amber-300 text-[10px]">👑 MVP</span>`;
        } else if (p.performance_tag === 'SVP') {
            tagBadge = `<span class="px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-800 font-black border border-indigo-300 text-[10px]">⭐ SVP</span>`;
        } else if (p.performance_tag === 'GREAT') {
            tagBadge = `<span class="px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold border border-emerald-300 text-[10px]">🔥 Tốt</span>`;
        } else if (p.performance_tag === 'UNDERPERFORMING') {
            tagBadge = `<span class="px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 font-medium border border-amber-200 text-[10px]">⚠️ Dưới sức</span>`;
        } else if (p.performance_tag === 'FEEDER') {
            tagBadge = `<span class="px-1.5 py-0.5 rounded bg-rose-100 text-rose-800 font-bold border border-rose-300 text-[10px]">💀 Thọt</span>`;
        }

        const deltaVal = p.recommended_delta || (isWinner ? 16 : -16);
        const deltaFormatted = deltaVal > 0 ? `+${deltaVal}` : `${deltaVal}`;
        const inputColor = deltaVal >= 0 ? 'text-emerald-700 bg-emerald-50/60 border-emerald-300' : 'text-rose-700 bg-rose-50/60 border-rose-300';

        const tr = document.createElement('tr');
        tr.className = `hover:bg-slate-50 transition ${isTeam1 ? 'bg-blue-50/20' : 'bg-rose-50/20'}`;
        tr.innerHTML = `
            <td class="py-2 px-3">
                <div class="flex items-center gap-2">
                    <span class="w-4 h-4 rounded-full ${isTeam1 ? 'bg-blue-600' : 'bg-rose-600'} text-white text-[9px] font-bold flex items-center justify-center flex-shrink-0">
                        ${isTeam1 ? '1' : '2'}
                    </span>
                    <img src="${playerObj.avatar || `https://api.dicebear.com/7.x/bottts/svg?seed=${p.player_id}`}" class="w-6 h-6 rounded-lg object-cover bg-slate-100 border border-slate-200 flex-shrink-0">
                    <div class="truncate">
                        <span class="font-bold text-slate-900 block truncate">${p.nickname}</span>
                        <span class="text-[10px] text-slate-400 truncate block">${p.comment || ''}</span>
                    </div>
                </div>
            </td>
            <td class="py-2 px-2 text-center font-medium text-slate-700">
                ${p.champion && p.champion !== '-' ? `<span class="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200">${p.champion}</span>` : '<span class="text-slate-400">-</span>'}
            </td>
            <td class="py-2 px-2 text-center font-bold text-slate-800 whitespace-nowrap">
                ${p.kda || '-'}
            </td>
            <td class="py-2 px-2 text-center whitespace-nowrap">
                ${tagBadge}
            </td>
            <td class="py-2 px-3 text-center">
                <div class="flex items-center justify-center">
                    <input type="number" step="0.5" id="ai-delta-${p.player_id}" value="${deltaVal}" class="w-18 text-center font-black py-1 px-1.5 rounded-xl border text-xs shadow-2xs focus:outline-none focus:ring-1 focus:ring-indigo-500 ${inputColor}">
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });

    resultsContainer.classList.remove('hidden');
    if (controlsElem) controlsElem.classList.remove('hidden');
}

async function confirmSaveAiCustomMatch() {
    if (!currentAiScoreboardAnalysis) return;
    if (isSubmittingSimulationMatch) return;
    isSubmittingSimulationMatch = true;

    const btn = document.getElementById('btn-confirm-save-ai');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Đang lưu kết quả...';
    }

    try {
        // Thu thập các delta tùy chỉnh từ bảng
        const customDeltas = {};
        const playersList = currentAiScoreboardAnalysis.players_analysis || [];
        playersList.forEach(p => {
            const inputElem = document.getElementById(`ai-delta-${p.player_id}`);
            if (inputElem) {
                const val = parseFloat(inputElem.value);
                customDeltas[p.player_id] = isNaN(val) ? p.recommended_delta : val;
            } else {
                customDeltas[p.player_id] = p.recommended_delta;
            }
        });

        const res = await fetch('/api/update_match_result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                team1: simTeam1,
                team2: simTeam2,
                winner: currentMatchModalWinner,
                player_deltas: customDeltas,
                player_performances: playersList,
                ai_summary: currentAiScoreboardAnalysis.ai_summary,
                notes: `AI Scoreboard: MVP ${currentAiScoreboardAnalysis.match_mvp || '-'}, SVP ${currentAiScoreboardAnalysis.match_svp || '-'}`
            })
        });

        const data = await res.json();
        if (!data.success) {
            throw new Error(data.error || 'Lỗi khi lưu kết quả trận đấu');
        }

        closeMatchResultModal();
        Swal.fire({
            icon: 'success',
            title: 'Đã tối ưu hóa điểm Elo!',
            text: 'Điểm Elo cá nhân hóa theo KDA và Phong độ đã được cập nhật thành công.',
            timer: 2500,
            showConfirmButton: false,
            ...SWAL_THEME
        });

        await loadAllPlayers();
        await loadStatus();
    } catch (err) {
        Swal.fire({
            icon: 'error',
            title: 'Lỗi lưu trận đấu',
            text: err.message,
            ...SWAL_THEME
        });
    } finally {
        isSubmittingSimulationMatch = false;
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-circle-check"></i> <span>Xác Nhận & Áp Dụng Elo Phân Tích</span>';
        }
    }
}

// ==========================================
// SCREENSHOT OCR (QUÉT ẢNH PHÒNG ĐẤU AI)
// ==========================================
window.addEventListener('paste', handleGlobalScreenshotPaste);

function handleGlobalScreenshotPaste(e) {
    const items = (e.clipboardData || e.originalEvent?.clipboardData)?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === 'file' && item.type.startsWith('image/')) {
            const blob = item.getAsFile();
            if (!blob) continue;

            // Nếu modal kết quả trận đấu đang mở -> nạp vào bảng điểm modal
            const matchModal = document.getElementById('match-result-modal');
            if (matchModal && !matchModal.classList.contains('hidden')) {
                switchMatchResultTab('ai_scoreboard');
                handleScoreboardPastedFile(blob);
                return;
            }

            // Ngược lại -> nạp vào OCR sảnh chờ thông thường
            processScreenshotFile(blob);
            break;
        }
    }
}

const dropzone = document.getElementById('screenshot-dropzone');
if (dropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('border-indigo-500', 'bg-indigo-50/80');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('border-indigo-500', 'bg-indigo-50/80');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer?.files;
        if (files && files.length > 0 && files[0].type.startsWith('image/')) {
            processScreenshotFile(files[0]);
        }
    });
}

function handleScreenshotFileSelect(e) {
    const files = e.target?.files;
    if (files && files.length > 0) {
        processScreenshotFile(files[0]);
    }
}

async function processScreenshotFile(file) {
    const spinner = document.getElementById('ocr-loading-spinner');
    if (spinner) spinner.classList.remove('hidden');

    const reader = new FileReader();
    reader.onload = async function (evt) {
        const base64Data = evt.target.result;
        let apiKey = localStorage.getItem('fbcs_gemini_key') || '';

        try {
            let res = await fetch('/api/ocr_screenshot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: base64Data,
                    mime_type: file.type || 'image/jpeg',
                    api_key: apiKey
                })
            });

            let data = await res.json();

            // Nếu chưa có API Key, cho phép người dùng nhập ngay tại chỗ
            if (!data.success && data.error && data.error.includes('Chưa cấu hình Gemini API Key')) {
                if (spinner) spinner.classList.add('hidden');
                
                const { value: inputKey } = await Swal.fire({
                    title: '🔑 Nhập Gemini API Key',
                    text: 'Tính năng AI Vision cần Gemini API Key để nhận diện ảnh chụp sảnh chờ phòng đấu.',
                    input: 'password',
                    inputPlaceholder: 'Dán Gemini API Key của bạn vào đây...',
                    showCancelButton: true,
                    confirmButtonText: 'Lưu & Quét Lại',
                    cancelButtonText: 'Hủy',
                    ...SWAL_THEME,
                    inputValidator: (val) => {
                        if (!val || !val.trim()) return 'Vui lòng không để trống API Key!';
                    }
                });

                if (inputKey) {
                    apiKey = inputKey.trim();
                    localStorage.setItem('fbcs_gemini_key', apiKey);
                    const keyInputElem = document.getElementById('input-gemini-key');
                    if (keyInputElem) keyInputElem.value = apiKey;
                    
                    if (spinner) spinner.classList.remove('hidden');
                    res = await fetch('/api/ocr_screenshot', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image: base64Data,
                            mime_type: file.type || 'image/jpeg',
                            api_key: apiKey
                        })
                    });
                    data = await res.json();
                } else {
                    return;
                }
            }

            if (spinner) spinner.classList.add('hidden');

            if (data.success) {
                lastOcrResult = data;

                // 1. Phân bổ vào các slot Mô Phỏng 5vs5
                simUnmatchedSlotInfo = { team1: {}, team2: {} };

                if (data.team1_slots && data.team1_slots.length > 0) {
                    data.team1_slots.forEach((slot, idx) => {
                        if (idx < 5) {
                            if (slot.matched_id) {
                                simTeam1[idx] = slot.matched_id;
                            } else if (slot.raw_name) {
                                simTeam1[idx] = null;
                                simUnmatchedSlotInfo.team1[idx] = slot.raw_name;
                            }
                        }
                    });
                }
                if (data.team2_slots && data.team2_slots.length > 0) {
                    data.team2_slots.forEach((slot, idx) => {
                        if (idx < 5) {
                            if (slot.matched_id) {
                                simTeam2[idx] = slot.matched_id;
                            } else if (slot.raw_name) {
                                simTeam2[idx] = null;
                                simUnmatchedSlotInfo.team2[idx] = slot.raw_name;
                            }
                        }
                    });
                }

                // 2. Điền vào matchmaker nếu có người khớp
                if (data.matched_player_ids && data.matched_player_ids.length > 0) {
                    selectedMatchmaker = data.matched_player_ids.slice(0, 10);
                    renderMatchmakerPlayers();
                }

                renderSimulationBoard();

                const matchedNicknames = (data.matched_player_ids || []).map(pid => {
                    const found = allPlayers.find(x => x.id === pid);
                    return found ? found.nickname : pid;
                });

                const allDetected = data.detected_names || [];
                const unmatchedCount = allDetected.length - (data.matched_player_ids || []).length;

                const summaryBox = document.getElementById('ocr-result-summary');
                const namesElem = document.getElementById('ocr-detected-names');
                if (summaryBox && namesElem) {
                    summaryBox.classList.remove('hidden');
                    namesElem.innerText = `${matchedNicknames.length} tuyển thủ: ${matchedNicknames.join(', ')}` + 
                        (unmatchedCount > 0 ? ` (kèm ${unmatchedCount} tên mới)` : '');
                }

                Swal.fire({
                    icon: 'success',
                    title: `🎯 Nhận diện thành công ${allDetected.length || data.count || 0} tuyển thủ!`,
                    html: `
                        <div class="text-left text-xs text-slate-700 mt-2 space-y-2">
                            ${matchedNicknames.length ? `<p><b>Đã khớp dữ liệu (${matchedNicknames.length}):</b> ${matchedNicknames.join(', ')}</p>` : ''}
                            ${allDetected.length ? `<p class="text-[11px] text-slate-500"><b>Tất cả tên đọc được từ ảnh:</b> ${allDetected.join(', ')}</p>` : ''}
                            ${unmatchedCount > 0 ? `<p class="text-[11px] text-amber-700 bg-amber-50 p-2.5 rounded-xl border border-amber-200">💡 <b>Có ${unmatchedCount} tên chưa có trong danh sách</b>. Bạn có thể mở Giao Diện 5vs5 để bấm <b>+ Tạo</b> tuyển thủ mới ngay lập tức!</p>` : ''}
                        </div>
                    `,
                    confirmButtonText: '🎮 Mở Giao Diện 5vs5',
                    showCancelButton: true,
                    cancelButtonText: selectedMatchmaker.length === 10 ? '⚡ Chia Đội Ngay' : 'Đóng',
                    confirmButtonColor: '#7c3aed',
                    cancelButtonColor: '#4f46e5',
                    ...SWAL_THEME
                }).then((result) => {
                    if (result.isConfirmed) {
                        switchTab('simulation');
                    } else if (result.dismiss === Swal.DismissReason.cancel && selectedMatchmaker.length === 10) {
                        handleCreateTeams();
                    }
                });

            } else {
                Swal.fire({
                    icon: 'warning',
                    title: 'Chưa tìm thấy tuyển thủ phù hợp',
                    text: data.error || 'AI không nhận diện được tên tuyển thủ nào trong bức ảnh này. Vui lòng thử ảnh rõ nét hơn.',
                    ...SWAL_THEME
                });
            }

        } catch (err) {
            if (spinner) spinner.classList.add('hidden');
            console.error("Lỗi OCR:", err);
            Swal.fire({
                icon: 'error',
                title: 'Lỗi',
                text: 'Không thể kết nối đến dịch vụ phân tích ảnh.',
                ...SWAL_THEME
            });
        }
    };
    reader.readAsDataURL(file);
}
