// ==========================================
// MODULE: CHI TIẾT TUYỂN THỦ (PLAYER DETAILS)
// ==========================================

let currentPlayerDetails = null;
let currentPlayerId = null;
let playerRadarChart = null;
let playerTrendChart = null;
let currentPlayerMatchFilter = 'all';

/**
 * Mở giao diện chi tiết tuyển thủ
 */
async function openPlayerDetail(playerId, pushState = true) {
    if (!playerId) return;
    currentPlayerId = playerId.trim().toLowerCase();

    // 1. Chuyển tab sang Player Detail
    switchTab('player-detail');

    // 2. Cập nhật URL History để người dùng có thể bookmark hoặc share link
    if (pushState && window.history && window.history.pushState) {
        window.history.pushState({ playerId: currentPlayerId }, '', `/player/${currentPlayerId}`);
    }

    // 3. Hiển thị trạng thái đang tải
    const loadingEl = document.getElementById('player-detail-loading');
    const contentEl = document.getElementById('player-detail-content');
    if (loadingEl) loadingEl.classList.remove('hidden');
    if (contentEl) contentEl.classList.add('hidden');

    // 4. Đồng bộ dropdown chọn tuyển thủ
    populatePlayerDetailDropdown(currentPlayerId);

    // 5. Gọi API lấy thông tin chi tiết
    try {
        const res = await fetch(`/api/players/${currentPlayerId}/details`);
        const data = await res.json();

        if (!data.success) {
            Swal.fire({
                icon: 'error',
                title: 'Không tìm thấy tuyển thủ',
                text: data.error || 'Tuyển thủ không tồn tại trong hệ thống',
                confirmButtonColor: '#4f46e5'
            });
            closePlayerDetail();
            return;
        }

        currentPlayerDetails = data;
        renderPlayerDetailUI(data);

        if (loadingEl) loadingEl.classList.add('hidden');
        if (contentEl) contentEl.classList.remove('hidden');

    } catch (err) {
        console.error("Lỗi khi tải chi tiết tuyển thủ:", err);
        Swal.fire({
            icon: 'error',
            title: 'Lỗi nạp dữ liệu',
            text: 'Không thể kết nối máy chủ để lấy thông tin chi tiết tuyển thủ.',
            confirmButtonColor: '#4f46e5'
        });
        if (loadingEl) loadingEl.classList.add('hidden');
    }
}

/**
 * Đóng trang chi tiết, quay lại Bảng Xếp Hạng
 */
function closePlayerDetail() {
    switchTab('leaderboard');
    if (window.history && window.history.pushState) {
        window.history.pushState({}, '', '/');
    }
}

/**
 * Điền danh sách tuyển thủ vào ô select chuyển nhanh
 */
function populatePlayerDetailDropdown(selectedId) {
    const select = document.getElementById('player-detail-select');
    if (!select) return;

    select.innerHTML = '';
    const playersList = (typeof allPlayers !== 'undefined' && allPlayers.length > 0)
        ? [...allPlayers].sort((a, b) => b.power_score - a.power_score)
        : [];

    playersList.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = `${p.nickname} (#${p.rank || '-'} - ${p.power_score} pts)`;
        if (p.id.toLowerCase() === selectedId) {
            opt.selected = true;
        }
        select.appendChild(opt);
    });
}

/**
 * Hiển thị dữ liệu lên toàn bộ giao diện
 */
function renderPlayerDetailUI(data) {
    const p = data.player || {};
    const stats = data.stats || {};
    const ai = data.ai_analysis || {};
    const matches = data.matches || [];

    // --- 1. HERO BANNER ---
    const avatarEl = document.getElementById('detail-avatar');
    if (avatarEl) avatarEl.src = p.avatar || `https://api.dicebear.com/7.x/bottts/svg?seed=${p.id}`;

    const nickEl = document.getElementById('detail-nickname');
    if (nickEl) nickEl.innerText = p.nickname || p.id;

    const idEl = document.getElementById('detail-id');
    if (idEl) idEl.innerText = `@${p.id}`;

    const rankBadgeEl = document.getElementById('detail-rank-badge');
    if (rankBadgeEl) {
        if (p.rank && p.rank !== '-') {
            rankBadgeEl.innerText = `#${p.rank} Server`;
            rankBadgeEl.className = p.rank === 1
                ? "px-3 py-1 rounded-full text-xs font-black font-heading bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 text-slate-950 shadow-xs"
                : (p.rank <= 3
                    ? "px-3 py-1 rounded-full text-xs font-black font-heading bg-slate-200 text-slate-800 shadow-2xs"
                    : "px-3 py-1 rounded-full text-xs font-bold font-heading bg-slate-100 text-slate-700");
        } else {
            rankBadgeEl.innerText = "Chưa xếp hạng";
            rankBadgeEl.className = "px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-500";
        }
    }

    const tierBadgeEl = document.getElementById('detail-tier-badge');
    if (tierBadgeEl) {
        tierBadgeEl.innerHTML = `<span>${p.tier_icon || '🛡️'}</span> <span>${p.tier_name || 'Tier B'}</span>`;
        tierBadgeEl.className = `px-2.5 py-1 rounded-xl text-xs font-bold border inline-flex items-center gap-1.5 shadow-2xs ${p.tier_badge_class || 'bg-slate-100 text-slate-700 border-slate-200'}`;
    }

    const roleBadgeEl = document.getElementById('detail-role-badge');
    if (roleBadgeEl) {
        roleBadgeEl.innerText = `Sở trường: ${stats.primary_role || 'ALL'}`;
    }

    // AI Persona Title
    const aiTitleEl = document.getElementById('detail-ai-title');
    if (aiTitleEl) {
        const titleText = ai.persona_title || 'Chiến Binh Thực Chiến';
        aiTitleEl.innerHTML = `<i class="fa-solid fa-crown text-amber-500 mr-1.5"></i><span>${titleText}</span>`;
    }

    // Form Status Indicator (Fire/Cold/Neutral)
    const formInd = document.getElementById('detail-form-status-indicator');
    if (formInd) {
        const formStatus = p.form?.status || 'neutral';
        if (formStatus === 'on_fire') {
            formInd.innerHTML = '🔥';
            formInd.title = 'Phong độ cực cao (On Fire)';
            formInd.className = 'absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-rose-50 border-2 border-rose-300 flex items-center justify-center text-xs shadow-xs animate-pulse';
        } else if (formStatus === 'cold') {
            formInd.innerHTML = '❄️';
            formInd.title = 'Phong độ cần xốc lại';
            formInd.className = 'absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-blue-50 border-2 border-blue-300 flex items-center justify-center text-xs shadow-xs';
        } else {
            formInd.innerHTML = '⚡';
            formInd.title = 'Phong độ ổn định';
            formInd.className = 'absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-white border border-slate-200 flex items-center justify-center text-xs shadow-xs';
        }
    }

    // Metric Summary Cards
    const powerEl = document.getElementById('detail-power-score');
    if (powerEl) powerEl.innerText = p.power_score;

    const rapmSubEl = document.getElementById('detail-rapm-sub');
    if (rapmSubEl) {
        const rapmSign = p.rapm > 0 ? '+' : '';
        const conf = p.confidence !== undefined ? ` (${Math.round(p.confidence * 100)}%)` : '';
        rapmSubEl.innerText = `RAPM: ${rapmSign}${p.rapm}${conf}`;
    }

    const wrEl = document.getElementById('detail-winrate');
    if (wrEl) wrEl.innerText = `${p.winrate}%`;

    const recSubEl = document.getElementById('detail-record-sub');
    if (recSubEl) recSubEl.innerText = `${p.wins} Thắng / ${p.losses} Thua`;

    const matchesCountEl = document.getElementById('detail-matches-count');
    if (matchesCountEl) matchesCountEl.innerText = stats.total_matches || p.matches || 0;

    const effSubEl = document.getElementById('detail-effective-sub');
    if (effSubEl) effSubEl.innerText = `H.Dụng: ${p.effective_matches !== undefined ? p.effective_matches : p.matches}`;

    // Recent 5 & Streak
    const recent5Container = document.getElementById('detail-recent-5');
    if (recent5Container) {
        const recent5List = p.recent_5 || p.form?.recent_5 || [];
        recent5Container.innerHTML = recent5List.map(r => {
            if (r === 'W') return `<span class="w-5 h-5 rounded-md bg-emerald-100 text-emerald-800 border border-emerald-300 text-[10px] font-bold inline-flex items-center justify-center">W</span>`;
            return `<span class="w-5 h-5 rounded-md bg-rose-100 text-rose-800 border border-rose-300 text-[10px] font-bold inline-flex items-center justify-center">L</span>`;
        }).join('') || '<span class="text-slate-400 text-xs">-</span>';
    }

    const streakEl = document.getElementById('detail-streak-badge');
    if (streakEl) {
        const st = stats.current_streak || {};
        if (st.count > 0) {
            const stLabel = st.type === 'W' ? `Thắng liên tiếp x${st.count} 🔥` : `Thua liên tiếp x${st.count}`;
            streakEl.innerText = stLabel;
            streakEl.className = st.type === 'W' ? "text-[10px] text-emerald-600 font-bold block" : "text-[10px] text-rose-500 font-medium block";
        } else {
            streakEl.innerText = "Chuỗi: N/A";
        }
    }

    // Render Badges Collection
    renderBadgesCollection(p.badges || []);

    // --- 2. AI INTELLIGENCE SECTION ---
    renderAiSection(ai, p);

    // --- 3. CHARTS SECTION ---
    renderCharts(stats.radar_scores, stats.trend_data);

    // --- 4. SIDE STATS & ADVANCED METRICS ---
    renderSideAndAdvancedStats(stats);

    // --- 5. SYNERGIES & RIVALS ---
    renderTeammatesAndRivals(stats.best_teammates || [], stats.rivals || []);

    // --- 6. MATCH HISTORY ---
    renderMatchHistoryList(matches, currentPlayerMatchFilter);
}

/**
 * Hiển thị huy hiệu và danh hiệu
 */
function renderBadgesCollection(badges) {
    const container = document.getElementById('detail-badges-container');
    const countEl = document.getElementById('detail-badge-count');
    if (!container) return;

    if (countEl) countEl.innerText = `${badges.length} danh hiệu đạt được`;

    if (badges.length === 0) {
        container.innerHTML = '<span class="text-xs text-slate-400 italic">Chưa có danh hiệu nào. Hãy tham gia thêm các trận đấu để mở khóa!</span>';
        return;
    }

    container.innerHTML = badges.map(b => `
        <div class="px-3 py-1.5 rounded-2xl border text-xs font-bold inline-flex items-center gap-1.5 shadow-2xs hover:scale-105 transition-all cursor-help ${b.badge_class || 'bg-slate-100 text-slate-700 border-slate-200'}" title="${b.desc || b.label}">
            <span class="text-sm">${b.icon || '🎖️'}</span>
            <span>${b.label}</span>
        </div>
    `).join('');
}

/**
 * Hiển thị khối AI Personalization
 */
function renderAiSection(ai, player) {
    const sourceBadge = document.getElementById('detail-ai-source-badge');
    if (sourceBadge) {
        if (ai.source === 'gemini') {
            sourceBadge.innerText = 'Gemini AI Powered';
            sourceBadge.className = 'px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-400/30';
        } else {
            sourceBadge.innerText = 'FBCS AI Engine';
            sourceBadge.className = 'px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider bg-indigo-500/20 text-indigo-300 border border-indigo-400/30';
        }
    }

    const playstyleEl = document.getElementById('detail-ai-playstyle');
    if (playstyleEl) {
        playstyleEl.innerText = ai.playstyle_evaluation || 'Đang cập nhật đánh giá phong cách chơi...';
    }

    const commentaryEl = document.getElementById('detail-ai-commentary');
    if (commentaryEl) {
        commentaryEl.innerText = ai.coach_commentary || 'HLV AI đang theo dõi thêm các trận đấu để đưa ra lời bình luận chuyên sâu.';
    }

    const tacticalEl = document.getElementById('detail-ai-tactical');
    if (tacticalEl) {
        tacticalEl.innerText = ai.tactical_tips || 'Nên tạo khoảng trống và bọc lót tốt để tối ưu hóa hiệu quả.';
    }

    // Strengths
    const strengthsEl = document.getElementById('detail-ai-strengths');
    if (strengthsEl) {
        const sList = ai.strengths || [];
        if (sList.length > 0) {
            strengthsEl.innerHTML = sList.map(s => `
                <li class="flex items-start gap-2">
                    <i class="fa-solid fa-check text-emerald-400 mt-1 text-[11px]"></i>
                    <span>${s}</span>
                </li>
            `).join('');
        } else {
            strengthsEl.innerHTML = '<li class="text-slate-400 italic">Đang phân tích dữ liệu thế mạnh...</li>';
        }
    }

    // Weaknesses
    const weaknessesEl = document.getElementById('detail-ai-weaknesses');
    if (weaknessesEl) {
        const wList = ai.weaknesses || [];
        if (wList.length > 0) {
            weaknessesEl.innerHTML = wList.map(w => `
                <li class="flex items-start gap-2">
                    <i class="fa-solid fa-arrow-trend-up text-amber-400 mt-1 text-[11px]"></i>
                    <span>${w}</span>
                </li>
            `).join('');
        } else {
            weaknessesEl.innerHTML = '<li class="text-slate-400 italic">Đang phân tích điểm cần cải thiện...</li>';
        }
    }

    // AI Custom Badges
    const aiBadgesEl = document.getElementById('detail-ai-custom-badges');
    if (aiBadgesEl) {
        const customBadges = ai.custom_badges || [];
        if (customBadges.length > 0) {
            aiBadgesEl.innerHTML = customBadges.map(cb => `
                <div class="px-2.5 py-1 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-xs font-bold inline-flex items-center gap-1.5 transition cursor-help" title="${cb.desc || cb.label}">
                    <span>${cb.icon || '✨'}</span>
                    <span class="text-slate-100">${cb.label}</span>
                </div>
            `).join('');
        } else {
            aiBadgesEl.innerHTML = '<span class="text-xs text-slate-400 italic">Chưa có huy hiệu đặc biệt từ AI</span>';
        }
    }
}

/**
 * Vẽ các biểu đồ Chart.js
 */
function renderCharts(radarScores, trendData) {
    if (typeof Chart === 'undefined') {
        console.warn("Chart.js chưa được nạp");
        return;
    }

    const scores = radarScores || { power: 50, winrate: 50, form: 50, experience: 50, impact: 50, synergy: 50 };

    // Cập nhật text dưới Radar
    const rPower = document.getElementById('radar-val-power');
    const rWr = document.getElementById('radar-val-wr');
    const rSyn = document.getElementById('radar-val-synergy');
    if (rPower) rPower.innerText = scores.power;
    if (rWr) rWr.innerText = `${scores.winrate}%`;
    if (rSyn) rSyn.innerText = `${scores.synergy}%`;

    // 1. Radar Chart
    const radarCtx = document.getElementById('chart-player-radar');
    if (radarCtx) {
        if (playerRadarChart) {
            playerRadarChart.destroy();
        }

        playerRadarChart = new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: ['Thực Lực', 'Tỷ Lệ Thắng', 'Phong Độ', 'Kinh Nghiệm', 'Tầm Ảnh Hưởng', 'Ăn Ý Đồng Đội'],
                datasets: [{
                    label: 'Chỉ Số',
                    data: [
                        scores.power,
                        scores.winrate,
                        scores.form,
                        scores.experience,
                        scores.impact,
                        scores.synergy
                    ],
                    backgroundColor: 'rgba(99, 102, 241, 0.25)',
                    borderColor: 'rgb(79, 70, 229)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgb(67, 56, 202)',
                    pointBorderColor: '#ffffff',
                    pointHoverBackgroundColor: '#ffffff',
                    pointHoverBorderColor: 'rgb(79, 70, 229)',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(148, 163, 184, 0.2)' },
                        grid: { color: 'rgba(148, 163, 184, 0.2)' },
                        pointLabels: {
                            font: { family: '"Outfit", sans-serif', size: 11, weight: 'bold' },
                            color: '#475569'
                        },
                        suggestedMin: 0,
                        suggestedMax: 100,
                        ticks: { stepSize: 25, display: false }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                return ` ${ctx.label}: ${ctx.raw} / 100`;
                            }
                        }
                    }
                }
            }
        });
    }

    // 2. Trend Line Chart
    const trendCtx = document.getElementById('chart-player-trend');
    if (trendCtx) {
        if (playerTrendChart) {
            playerTrendChart.destroy();
        }

        const tList = (trendData && trendData.length > 0)
            ? trendData
            : [{ match_index: 1, running_winrate: scores.winrate, result: 'W' }];

        const labels = tList.map(t => `#${t.match_index}`);
        const dataPoints = tList.map(t => t.running_winrate);
        const pointColors = tList.map(t => t.result === 'W' ? '#10b981' : '#f43f5e');

        playerTrendChart = new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Tỷ Lệ Thắng Lũy Kế (%)',
                    data: dataPoints,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointBackgroundColor: pointColors,
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 1.5,
                    pointRadius: 4.5,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        suggestedMin: 0,
                        suggestedMax: 100,
                        grid: { color: 'rgba(226, 232, 240, 0.8)' },
                        ticks: {
                            font: { size: 10 },
                            callback: function(value) { return value + '%'; }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 10 } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                const item = tList[ctx.dataIndex];
                                const resLabel = item?.result === 'W' ? 'Thắng (Victory)' : 'Thua (Defeat)';
                                return ` Trận này: ${resLabel} | Tỷ lệ thắng: ${ctx.raw}%`;
                            }
                        }
                    }
                }
            }
        });
    }
}

/**
 * Hiển thị thống kê bên Xanh / Đỏ & trận cân não
 */
function renderSideAndAdvancedStats(stats) {
    const side = stats.side_stats || {};
    const t1 = side.team1 || { matches: 0, wins: 0, winrate: 0 };
    const t2 = side.team2 || { matches: 0, wins: 0, winrate: 0 };

    const bWr = document.getElementById('side-blue-wr');
    const bRec = document.getElementById('side-blue-record');
    if (bWr) bWr.innerText = `${t1.winrate}%`;
    if (bRec) bRec.innerText = `${t1.wins}W - ${t1.losses}L`;

    const rWr = document.getElementById('side-red-wr');
    const rRec = document.getElementById('side-red-record');
    if (rWr) rWr.innerText = `${t2.winrate}%`;
    if (rRec) rRec.innerText = `${t2.wins}W - ${t2.losses}L`;

    const maxWin = document.getElementById('streak-max-win');
    if (maxWin) maxWin.innerText = `${stats.longest_win_streak || 0} trận`;

    const closeStat = document.getElementById('stat-close-matches');
    const closeWr = document.getElementById('stat-close-winrate');
    const cMatches = stats.close_matches || { total: 0, wins: 0 };
    if (closeStat) closeStat.innerText = `${cMatches.total} trận`;
    if (closeWr) {
        const cPercent = cMatches.total > 0 ? Math.round((cMatches.wins / cMatches.total) * 100) : 0;
        closeWr.innerText = `${cMatches.wins}W (${cPercent}% thắng)`;
    }
}

/**
 * Hiển thị danh sách đồng đội ăn ý & đối thủ duyên nợ
 */
function renderTeammatesAndRivals(teammates, rivals) {
    const tmList = document.getElementById('detail-teammates-list');
    if (tmList) {
        if (teammates.length === 0) {
            tmList.innerHTML = '<div class="text-xs text-slate-400 py-6 text-center">Chưa đủ dữ liệu (cần từ 2 trận chung đội)</div>';
        } else {
            tmList.innerHTML = teammates.slice(0, 10).map(t => `
                <div onclick="openPlayerDetail('${t.id}')" class="p-3 rounded-2xl border border-slate-200/80 hover:border-indigo-300 hover:bg-slate-50 transition-all flex items-center justify-between shadow-2xs cursor-pointer group">
                    <div class="flex items-center gap-3">
                        <img src="${t.avatar}" class="w-9 h-9 rounded-xl bg-slate-100 border border-slate-200 object-cover group-hover:scale-105 transition-transform" alt="${t.nickname}">
                        <div>
                            <span class="text-xs font-bold font-heading text-slate-800 block group-hover:text-indigo-600 transition">${t.nickname}</span>
                            <span class="text-[10px] text-slate-400">${t.matches} trận cùng team</span>
                        </div>
                    </div>
                    <div class="text-right">
                        <span class="text-xs font-black font-heading ${t.winrate >= 60 ? 'text-emerald-600' : 'text-slate-700'}">${t.winrate}% WR</span>
                        <span class="text-[10px] text-slate-400 block">${t.wins}W / ${t.losses}L</span>
                    </div>
                </div>
            `).join('');
        }
    }

    const rvList = document.getElementById('detail-rivals-list');
    if (rvList) {
        if (rivals.length === 0) {
            rvList.innerHTML = '<div class="text-xs text-slate-400 py-6 text-center">Chưa có đối thủ thường xuyên</div>';
        } else {
            rvList.innerHTML = rivals.slice(0, 10).map(r => `
                <div onclick="openPlayerDetail('${r.id}')" class="p-3 rounded-2xl border border-slate-200/80 hover:border-rose-300 hover:bg-slate-50 transition-all flex items-center justify-between shadow-2xs cursor-pointer group">
                    <div class="flex items-center gap-3">
                        <img src="${r.avatar}" class="w-9 h-9 rounded-xl bg-slate-100 border border-slate-200 object-cover group-hover:scale-105 transition-transform" alt="${r.nickname}">
                        <div>
                            <span class="text-xs font-bold font-heading text-slate-800 block group-hover:text-rose-600 transition">${r.nickname}</span>
                            <span class="text-[10px] text-slate-400">${r.matches} trận chạm trán</span>
                        </div>
                    </div>
                    <div class="text-right">
                        <span class="text-xs font-black font-heading ${r.winrate_against >= 50 ? 'text-emerald-600' : 'text-rose-600'}">${r.winrate_against}% Thắng</span>
                        <span class="text-[10px] text-slate-400 block">${r.wins_against}W - ${r.losses_against}L</span>
                    </div>
                </div>
            `).join('');
        }
    }
}

/**
 * Bộ lọc trận đấu của tuyển thủ
 */
function filterPlayerMatches(filterType) {
    currentPlayerMatchFilter = filterType;

    const btnAll = document.getElementById('btn-match-filter-all');
    const btnWin = document.getElementById('btn-match-filter-win');
    const btnLoss = document.getElementById('btn-match-filter-loss');

    [btnAll, btnWin, btnLoss].forEach(b => {
        if (b) {
            b.className = "px-3 py-1.5 rounded-xl text-slate-600 hover:text-slate-900 transition";
        }
    });

    if (filterType === 'all' && btnAll) btnAll.className = "px-3 py-1.5 rounded-xl bg-white text-indigo-700 shadow-2xs transition";
    if (filterType === 'win' && btnWin) btnWin.className = "px-3 py-1.5 rounded-xl bg-white text-emerald-700 shadow-2xs transition";
    if (filterType === 'loss' && btnLoss) btnLoss.className = "px-3 py-1.5 rounded-xl bg-white text-rose-700 shadow-2xs transition";

    if (currentPlayerDetails && currentPlayerDetails.matches) {
        renderMatchHistoryList(currentPlayerDetails.matches, filterType);
    }
}

/**
 * Hiển thị danh sách lịch sử trận đấu
 */
function renderMatchHistoryList(matches, filterType = 'all') {
    const container = document.getElementById('player-matches-container');
    if (!container) return;

    // Cập nhật số đếm trên các nút bộ lọc
    const totalCount = matches.length;
    const winCount = matches.filter(m => m.is_winner).length;
    const lossCount = totalCount - winCount;

    const cAll = document.getElementById('count-matches-all');
    const cWin = document.getElementById('count-matches-win');
    const cLoss = document.getElementById('count-matches-loss');
    if (cAll) cAll.innerText = totalCount;
    if (cWin) cWin.innerText = winCount;
    if (cLoss) cLoss.innerText = lossCount;

    let filteredMatches = matches;
    if (filterType === 'win') {
        filteredMatches = matches.filter(m => m.is_winner);
    } else if (filterType === 'loss') {
        filteredMatches = matches.filter(m => !m.is_winner);
    }

    if (filteredMatches.length === 0) {
        container.innerHTML = `
            <div class="text-center py-12 text-slate-400 space-y-2">
                <i class="fa-solid fa-gamepad text-3xl text-slate-300"></i>
                <div class="text-sm font-bold text-slate-600">Không có trận đấu nào khớp với bộ lọc này</div>
            </div>
        `;
        return;
    }

    container.innerHTML = filteredMatches.map(m => {
        const isWin = m.is_winner;
        const sideLabel = m.player_side === 'team1' ? 'Đội Xanh (Blue)' : 'Đội Đỏ (Red)';
        const sideColor = m.player_side === 'team1' ? 'bg-blue-50 text-blue-700 border-blue-200' : 'bg-rose-50 text-rose-700 border-rose-200';

        const resultBadge = isWin
            ? `<span class="px-2.5 py-1 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-200 font-black text-xs uppercase tracking-wider flex items-center gap-1 shadow-2xs">
                 <i class="fa-solid fa-trophy text-emerald-500"></i> CHIẾN THẮNG
               </span>`
            : `<span class="px-2.5 py-1 rounded-xl bg-rose-50 text-rose-700 border border-rose-200 font-bold text-xs uppercase tracking-wider flex items-center gap-1">
                 <i class="fa-solid fa-xmark text-rose-500"></i> THẤT BẠI
               </span>`;

        // Teammates avatars
        const teammatesHtml = (m.teammates || []).map(t => `
            <div class="flex items-center gap-1 text-[11px] bg-slate-50 border border-slate-200/80 px-2 py-1 rounded-xl" title="${t.nickname}">
                <img src="${t.avatar}" class="w-5 h-5 rounded-lg object-cover">
                <span class="font-medium text-slate-700 truncate max-w-[80px]">${t.nickname}</span>
            </div>
        `).join('');

        // Opponents avatars
        const opponentsHtml = (m.opponents || []).map(o => `
            <div class="flex items-center gap-1 text-[11px] bg-slate-50 border border-slate-200/80 px-2 py-1 rounded-xl opacity-80" title="${o.nickname}">
                <img src="${o.avatar}" class="w-5 h-5 rounded-lg object-cover">
                <span class="font-medium text-slate-700 truncate max-w-[80px]">${o.nickname}</span>
            </div>
        `).join('');

        // Badges: stomp, closeness
        let balanceBadge = '';
        if (m.is_stomp) {
            balanceBadge = `<span class="px-2 py-0.5 rounded-lg text-[10px] font-bold bg-amber-50 text-amber-700 border border-amber-200">Trận áp đảo</span>`;
        } else if (m.match_closeness >= 0.7 || m.balance_rating === 'perfect') {
            balanceBadge = `<span class="px-2 py-0.5 rounded-lg text-[10px] font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">Cân tài cân sức</span>`;
        }

        // Kill score if available
        let scoreHtml = '';
        if (m.team1_kills > 0 || m.team2_kills > 0) {
            scoreHtml = `
                <div class="text-xs font-mono font-black px-2.5 py-1 rounded-xl bg-slate-100 border border-slate-200">
                    <span class="${m.player_side === 'team1' ? 'text-indigo-600 underline font-extrabold' : 'text-slate-600'}">${m.team1_kills}</span>
                    <span class="text-slate-400 mx-1">:</span>
                    <span class="${m.player_side === 'team2' ? 'text-indigo-600 underline font-extrabold' : 'text-slate-600'}">${m.team2_kills}</span>
                </div>
            `;
        }

        return `
            <div class="p-4 rounded-3xl border transition-all ${isWin ? 'border-emerald-200/80 bg-white hover:border-emerald-300' : 'border-slate-200/80 bg-white hover:border-slate-300'} shadow-xs space-y-3">
                <div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2.5">
                    <div class="flex items-center gap-2">
                        ${resultBadge}
                        <span class="px-2 py-0.5 rounded-lg text-[10px] font-bold border ${sideColor}">
                            ${sideLabel}
                        </span>
                        ${balanceBadge}
                    </div>

                    <div class="flex items-center gap-3 text-xs text-slate-400">
                        ${scoreHtml}
                        <span class="font-mono text-[11px]">${m.match_code}</span>
                    </div>
                </div>

                <!-- Teams Lineup -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                    <!-- Player Team -->
                    <div class="space-y-1">
                        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">Đồng đội cùng phe:</span>
                        <div class="flex flex-wrap gap-1.5">
                            ${teammatesHtml || '<span class="text-slate-400 italic">Không rõ</span>'}
                        </div>
                    </div>

                    <!-- Opponent Team -->
                    <div class="space-y-1">
                        <span class="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">Đối thủ đối đầu:</span>
                        <div class="flex flex-wrap gap-1.5">
                            ${opponentsHtml || '<span class="text-slate-400 italic">Không rõ</span>'}
                        </div>
                    </div>
                </div>

                ${m.ai_summary ? `
                    <div class="p-2.5 rounded-2xl bg-indigo-50/50 border border-indigo-100 text-xs text-indigo-900 flex items-start gap-2">
                        <i class="fa-solid fa-wand-magic-sparkles text-indigo-600 text-xs mt-0.5"></i>
                        <span class="leading-relaxed">${m.ai_summary}</span>
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

/**
 * Gọi lại AI phân tích hồ sơ tuyển thủ
 */
async function reanalyzeCurrentPlayerAI() {
    if (!currentPlayerId) return;

    const btn = document.getElementById('btn-reanalyze-ai');
    const icon = document.getElementById('icon-reanalyze-ai');
    const text = document.getElementById('text-reanalyze-ai');

    if (btn) btn.disabled = true;
    if (icon) icon.className = 'fa-solid fa-spinner fa-spin text-amber-300';
    if (text) text.innerText = 'AI Đang Phân Tích...';

    try {
        const res = await fetch(`/api/players/${currentPlayerId}/ai_analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const data = await res.json();
        if (data.success && data.ai_analysis) {
            if (currentPlayerDetails) {
                currentPlayerDetails.ai_analysis = data.ai_analysis;
                if (data.player) currentPlayerDetails.player = data.player;
                renderPlayerDetailUI(currentPlayerDetails);
            }

            Swal.fire({
                icon: 'success',
                title: 'Phân tích AI hoàn tất!',
                text: 'Hồ sơ tuyển thủ và danh hiệu cá nhân hóa đã được cập nhật thành công.',
                timer: 2000,
                showConfirmButton: false
            });
        } else {
            throw new Error(data.error || 'Lỗi khi gọi AI');
        }
    } catch (err) {
        console.error("Lỗi khi phân tích lại AI:", err);
        Swal.fire({
            icon: 'error',
            title: 'Lỗi AI',
            text: err.message || 'Không thể hoàn thành phân tích AI. Vui lòng thử lại.',
            confirmButtonColor: '#4f46e5'
        });
    } finally {
        if (btn) btn.disabled = false;
        if (icon) icon.className = 'fa-solid fa-wand-magic-sparkles text-amber-300';
        if (text) text.innerText = 'AI Phân Tích Lại';
    }
}

/**
 * Làm mới dữ liệu tuyển thủ hiện tại
 */
function refreshCurrentPlayerDetail() {
    if (currentPlayerId) {
        openPlayerDetail(currentPlayerId, false);
    }
}
