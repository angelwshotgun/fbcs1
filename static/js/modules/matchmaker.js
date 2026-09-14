// ==========================================
// TAB 1: MATCHMAKER (CHIA ĐỘI CÂN BẰNG)
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

function transferMatchmakerResultToSimulation() {
    if (!currentTeamsResult || !currentTeamsResult.team1 || !currentTeamsResult.team2) {
        Swal.fire({
            icon: 'warning',
            title: 'Chưa có kết quả chia đội',
            text: 'Vui lòng chọn 10 tuyển thủ và bấm Chia Đội trước khi chuyển sang mô phỏng.',
            ...SWAL_THEME
        });
        return;
    }

    simTeam1 = currentTeamsResult.team1.map(p => p.id);
    simTeam2 = currentTeamsResult.team2.map(p => p.id);
    simUnmatchedSlotInfo = { team1: {}, team2: {} };

    switchTab('simulation');
    renderSimulationBoard();
    updateSimulationLiveStats();

    Swal.fire({
        icon: 'success',
        title: 'Đã chuyển sang Mô Phỏng 5vs5!',
        text: '10 tuyển thủ từ kết quả chia đội đã được xếp đủ vào 2 đội hình.',
        timer: 1500,
        showConfirmButton: false,
        ...SWAL_THEME
    });
}

function submitMatchWinner(winningTeam) {
    if (!currentTeamsResult || !currentTeamsResult.team1 || !currentTeamsResult.team2) return;
    simTeam1 = currentTeamsResult.team1.map(p => p.id);
    simTeam2 = currentTeamsResult.team2.map(p => p.id);
    openMatchResultModal(winningTeam);
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

