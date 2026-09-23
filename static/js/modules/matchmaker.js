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
            <div class="flex items-center justify-center gap-1.5 mt-1.5">
                <span class="text-[10px] px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 font-bold border border-indigo-100">
                    Elo ${Math.round(p.hidden_elo)}
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

    const balanceMode = document.getElementById('matchmaking-mode')?.value || 'pure_elo';

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

    const totalElo1 = data.team1_total_elo || Math.round(data.team1_power);
    const totalElo2 = data.team2_total_elo || Math.round(data.team2_power);
    const avgElo1 = data.team1_avg_elo || Math.round(totalElo1 / 5);
    const avgElo2 = data.team2_avg_elo || Math.round(totalElo2 / 5);
    const diffElo = Math.abs(totalElo1 - totalElo2);

    const diffElem = document.getElementById('res-power-diff');
    if (diffElem) diffElem.innerText = `${diffElo} Elo`;

    const probElem = document.getElementById('res-win-prob-label');
    if (probElem) probElem.innerText = `${data.team1_win_prob}% - ${data.team2_win_prob}%`;

    // Mode badge
    const modeBadge = document.getElementById('res-mode-badge');
    if (modeBadge) {
        modeBadge.innerHTML = `<i class="fa-solid fa-crosshairs"></i> <span>Chế độ: Chuẩn Elo Ẩn</span>`;
        modeBadge.className = "px-3 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-bold font-heading flex items-center gap-1.5";
    }

    // RNG info badge
    const rngBadge = document.getElementById('res-rng-badge');
    if (rngBadge) {
        if (data.rng_applied) {
            rngBadge.innerHTML = `<i class="fa-solid fa-dice"></i> <span>RNG Cân Bằng: Sai số ±${diffElo} Elo</span>`;
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

    // Render H2H Balance Banner
    const h2h = data.h2h_analysis;
    const banner = document.getElementById('h2h-balance-banner');
    if (banner) {
        if (h2h && h2h.has_history && h2h.total_encounters > 0) {
            banner.classList.remove('hidden');
            const descEl = document.getElementById('h2h-summary-desc');
            if (descEl) descEl.innerText = h2h.summary || '';
            const titleEl = document.getElementById('h2h-summary-title');
            if (titleEl) {
                const bias = h2h.h2h_bias_elo || 0;
                if (Math.abs(bias) <= 15) {
                    titleEl.innerHTML = `Lịch sử đối đầu: <span class="text-emerald-700 font-bold">Cân bằng (±${Math.abs(bias)} Elo)</span>`;
                    banner.className = "mb-6 p-4 rounded-2xl border text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-emerald-50/70 border-emerald-200";
                } else if (bias > 15) {
                    titleEl.innerHTML = `Lịch sử đối đầu: <span class="text-blue-700 font-bold">Đội 1 có lợi thế (+${bias} Elo thực chiến)</span>`;
                    banner.className = "mb-6 p-4 rounded-2xl border text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-blue-50/70 border-blue-200";
                } else {
                    titleEl.innerHTML = `Lịch sử đối đầu: <span class="text-rose-700 font-bold">Đội 2 có lợi thế (+${Math.abs(bias)} Elo thực chiến)</span>`;
                    banner.className = "mb-6 p-4 rounded-2xl border text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-rose-50/70 border-rose-200";
                }
            }
            const rivList = document.getElementById('h2h-rivalries-list');
            if (rivList) {
                rivList.innerHTML = '';
                (h2h.rivalries || []).forEach(r => {
                    const span = document.createElement('span');
                    span.className = "px-2 py-0.5 rounded-lg bg-white border border-slate-200 text-slate-700 font-semibold text-[10px] shadow-xs";
                    span.innerHTML = `<b class="text-slate-900">${r.p1}</b> vs <b class="text-slate-900">${r.p2}</b>: ${r.w1}W - ${r.w2}W`;
                    rivList.appendChild(span);
                });
            }
        } else {
            banner.classList.add('hidden');
        }
    }

    const t1PowerElem = document.getElementById('team1-power-text');
    if (t1PowerElem) t1PowerElem.innerText = totalElo1;
    const t1AvgElem = document.getElementById('team1-avg-text');
    if (t1AvgElem) t1AvgElem.innerText = avgElo1;

    const t2PowerElem = document.getElementById('team2-power-text');
    if (t2PowerElem) t2PowerElem.innerText = totalElo2;
    const t2AvgElem = document.getElementById('team2-avg-text');
    if (t2AvgElem) t2AvgElem.innerText = avgElo2;

    const t1ProbBadge = document.getElementById('team1-prob-badge');
    if (t1ProbBadge) t1ProbBadge.innerText = `${data.team1_win_prob}% Thắng`;
    const t2ProbBadge = document.getElementById('team2-prob-badge');
    if (t2ProbBadge) t2ProbBadge.innerText = `${data.team2_win_prob}% Thắng`;

    // Render Team 1
    const t1List = document.getElementById('team1-players-list');
    t1List.innerHTML = '';
    data.team1.forEach(p => {
        t1List.appendChild(createTeamPlayerCard(p, 'blue'));
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
                span.innerHTML = `<span>${s.icon || '🤝'}</span> <span><b class="font-heading">${s.names}</b> (${isPositive ? '+' : ''}${s.bonus} Elo)</span>`;
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
        t2List.appendChild(createTeamPlayerCard(p, 'rose'));
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
                span.innerHTML = `<span>${s.icon || '🤝'}</span> <span><b class="font-heading">${s.names}</b> (${isPositive ? '+' : ''}${s.bonus} Elo)</span>`;
                t2SynList.appendChild(span);
            });
        } else {
            t2SynContainer.classList.add('hidden');
        }
    }

    // Auto scroll to results
    section.scrollIntoView({ behavior: 'smooth' });
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

function createTeamPlayerCard(p, teamColor) {
    const div = document.createElement('div');
    div.className = "flex items-center justify-between p-2.5 rounded-xl bg-white border border-slate-200/90 shadow-xs";
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

