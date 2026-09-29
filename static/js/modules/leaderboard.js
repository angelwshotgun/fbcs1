// TAB 4: LEADERBOARD (BẢNG XẾP HẠNG)
// ==========================================
let currentLeaderboardMode = 'active'; // 'active' (mặc định) hoặc 'all' (tổng)
let currentLeaderboardBadgeFilter = 'all';

function switchLeaderboardMode(mode) {
    currentLeaderboardMode = mode;
    const btnActive = document.getElementById('btn-leaderboard-mode-active');
    const btnAll = document.getElementById('btn-leaderboard-mode-all');
    const descActive = document.getElementById('leaderboard-mode-desc-active');
    const descAll = document.getElementById('leaderboard-mode-desc-all');

    if (mode === 'active') {
        if (btnActive) btnActive.className = "flex-1 sm:flex-initial px-4 sm:px-5 py-2.5 rounded-xl font-bold font-heading text-xs sm:text-sm transition-all shadow-xs bg-indigo-600 text-white flex items-center justify-center gap-2";
        if (btnAll) btnAll.className = "flex-1 sm:flex-initial px-4 sm:px-5 py-2.5 rounded-xl font-bold font-heading text-xs sm:text-sm transition-all text-slate-600 hover:text-slate-900 hover:bg-slate-200/60 flex items-center justify-center gap-2";
        if (descActive) descActive.classList.remove('hidden');
        if (descAll) descAll.classList.add('hidden');
    } else {
        if (btnActive) btnActive.className = "flex-1 sm:flex-initial px-4 sm:px-5 py-2.5 rounded-xl font-bold font-heading text-xs sm:text-sm transition-all text-slate-600 hover:text-slate-900 hover:bg-slate-200/60 flex items-center justify-center gap-2";
        if (btnAll) btnAll.className = "flex-1 sm:flex-initial px-4 sm:px-5 py-2.5 rounded-xl font-bold font-heading text-xs sm:text-sm transition-all shadow-xs bg-indigo-600 text-white flex items-center justify-center gap-2";
        if (descActive) descActive.classList.add('hidden');
        if (descAll) descAll.classList.remove('hidden');
    }

    renderLeaderboard();
}

function setLeaderboardBadgeFilter(filterKey) {
    currentLeaderboardBadgeFilter = filterKey;

    const filterBtns = [
        'all', 'tier_s', 'tier_a', 'tier_b', 'tier_c', 'top3', 'streak'
    ];

    filterBtns.forEach(key => {
        const btn = document.getElementById(`btn-badge-filter-${key}`);
        if (!btn) return;
        if (key === filterKey) {
            btn.className = "px-3 py-1.5 rounded-xl bg-indigo-600 text-white shadow-xs transition flex items-center gap-1";
        } else {
            btn.className = "px-3 py-1.5 rounded-xl bg-slate-100 text-slate-700 hover:bg-slate-200 transition flex items-center gap-1";
        }
    });

    renderLeaderboard();
}

function renderLeaderboard() {
    const tbody = document.getElementById('leaderboard-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    // 1. Toàn bộ danh sách tuyển thủ đã thi đấu ít nhất 1 trận, sắp xếp theo Điểm Thực Lực
    const allPlayed = (allPlayers || [])
        .filter(p => (p.matches || 0) > 0)
        .sort((a, b) => b.power_score - a.power_score);

    // Gán thứ hạng Tổng Server tuyệt đối
    allPlayed.forEach((p, idx) => {
        p.globalRank = idx + 1;
    });

    // 2. Tuyển thủ thường xuyên & gần đây: thi đấu >= 5 trận và số trận hiệu dụng >= 2.0
    const isActiveRecent = (p) => (p.matches || 0) >= 5 && (p.effective_matches || 0) >= 2.0;
    const activePool = allPlayed.filter(isActiveRecent);

    // Cập nhật số lượng trên các nút chế độ
    const badgeActive = document.getElementById('badge-active-count');
    const badgeAll = document.getElementById('badge-all-count');
    if (badgeActive) badgeActive.innerText = activePool.length;
    if (badgeAll) badgeAll.innerText = allPlayed.length;

    // Chọn pool theo chế độ đang xem
    const currentPool = (currentLeaderboardMode === 'active') ? activePool : allPlayed;

    // 3. Áp dụng bộ lọc phân bậc (Tier / Top3 / Streak)
    let rankedPlayers = currentPool;
    if (currentLeaderboardBadgeFilter !== 'all') {
        rankedPlayers = rankedPlayers.filter(p => {
            if (currentLeaderboardBadgeFilter === 'tier_s') return p.tier === 'S';
            if (currentLeaderboardBadgeFilter === 'tier_a') return p.tier === 'A';
            if (currentLeaderboardBadgeFilter === 'tier_b') return p.tier === 'B';
            if (currentLeaderboardBadgeFilter === 'tier_c') return p.tier === 'C';
            if (currentLeaderboardBadgeFilter === 'top3') {
                const bKeys = (p.badges || []).map(b => b.key);
                return bKeys.includes('top1') || bKeys.includes('top2') || bKeys.includes('top3');
            }
            if (currentLeaderboardBadgeFilter === 'streak') {
                const bKeys = (p.badges || []).map(b => b.key);
                return bKeys.includes('streak');
            }
            return true;
        });
    }

    const countLabel = document.getElementById('leaderboard-filtered-count');
    if (countLabel) {
        if (currentLeaderboardMode === 'active') {
            countLabel.innerHTML = `Đang xem: <b class="text-indigo-600 font-bold">${rankedPlayers.length}</b> / ${activePool.length} tuyển thủ thường xuyên &amp; gần đây`;
        } else {
            countLabel.innerHTML = `Đang xem: <b class="text-indigo-600 font-bold">${rankedPlayers.length}</b> / ${allPlayed.length} toàn bộ tuyển thủ server`;
        }
    }

    if (rankedPlayers.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center py-12 text-slate-400">
                    <div class="flex flex-col items-center justify-center gap-2">
                        <i class="fa-solid fa-filter text-slate-300 text-3xl"></i>
                        <span class="font-bold text-slate-600 text-sm">Không có tuyển thủ nào khớp với bộ lọc này</span>
                        <p class="text-xs text-slate-400 max-w-md">
                            Hãy thử chuyển sang tab "Bảng Xếp Hạng Tổng" hoặc bấm "Tất Cả" để xem toàn bộ danh sách!
                        </p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    rankedPlayers.forEach((p, idx) => {
        // Hạng hiển thị:
        // Ở chế độ active: hạng tương đối 1, 2, 3... trong nhóm active
        // Ở chế độ tổng: hạng tuyệt đối trên toàn server
        const displayRank = (currentLeaderboardMode === 'active') ? (idx + 1) : p.globalRank;

        let rankBadge = `<span class="font-bold text-slate-500">#${displayRank}</span>`;
        if (displayRank === 1) rankBadge = `<span class="w-7 h-7 rounded-full bg-amber-400 text-slate-900 font-black flex items-center justify-center mx-auto shadow-xs">1</span>`;
        if (displayRank === 2) rankBadge = `<span class="w-7 h-7 rounded-full bg-slate-300 text-slate-900 font-black flex items-center justify-center mx-auto shadow-xs">2</span>`;
        if (displayRank === 3) rankBadge = `<span class="w-7 h-7 rounded-full bg-amber-700 text-white font-black flex items-center justify-center mx-auto shadow-xs">3</span>`;

        let rankSubtext = '';
        if (currentLeaderboardMode === 'active' && p.globalRank !== displayRank) {
            rankSubtext = `<span class="text-[9px] text-slate-400 block font-mono" title="Thứ hạng trong Bảng Xếp Hạng Tổng Server">Tổng #${p.globalRank}</span>`;
        } else if (currentLeaderboardMode === 'all') {
            if (isActiveRecent(p)) {
                rankSubtext = `<span class="px-1.5 py-0.5 rounded text-[9px] bg-emerald-50 text-emerald-700 border border-emerald-200 font-bold block mt-1" title="Thường xuyên thi đấu gần đây">Thường xuyên</span>`;
            } else {
                rankSubtext = `<span class="px-1.5 py-0.5 rounded text-[9px] bg-slate-100 text-slate-500 border border-slate-200 block mt-1" title="Ít thi đấu gần đây hoặc khách mời">Ít đấu</span>`;
            }
        }

        const recentBadges = (p.recent_5 || p.form?.recent_5 || []).map(r => {
            if (r === 'W') return `<span class="w-5 h-5 rounded-md bg-emerald-100 text-emerald-800 border border-emerald-200 text-[10px] font-bold inline-flex items-center justify-center">W</span>`;
            return `<span class="w-5 h-5 rounded-md bg-rose-100 text-rose-800 border border-rose-200 text-[10px] font-bold inline-flex items-center justify-center">L</span>`;
        }).join('');

        // Tier Badge with stars
        const tierBadgeHtml = `
            <span class="px-2.5 py-1 rounded-xl text-xs font-bold border inline-flex items-center gap-1.5 shadow-2xs ${p.tier_badge_class || 'bg-slate-100 text-slate-700 border-slate-200'}" title="${p.tier_desc || ''}">
                <span>${p.tier_icon || '🛡️'}</span>
                <span>${p.tier_name || 'Tier B'}</span>
            </span>
        `;

        // Render Các Danh Hiệu Thực Chiến
        const extraBadgesHtml = (p.badges || []).map(b => `
            <span class="px-2 py-0.5 rounded-full text-[10px] inline-flex items-center gap-1 border shadow-2xs transition hover:scale-105 cursor-help ${b.badge_class || 'bg-slate-100 text-slate-700 border-slate-200'}" title="${b.desc || b.label}">
                <span>${b.icon}</span>
                <span>${b.label}</span>
            </span>
        `).join('');

        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-50 transition";
        tr.innerHTML = `
            <td class="py-3.5 px-4 text-center">${rankBadge}${rankSubtext}</td>
            <td class="py-3.5 px-4">
                <div class="flex items-center gap-3">
                    <img src="${p.avatar}" class="w-9 h-9 rounded-xl bg-slate-100 border border-slate-200 object-cover" alt="${p.nickname}">
                    <div>
                        <span class="font-bold text-slate-900 block">${p.nickname}</span>
                        <span class="text-xs text-slate-400">@${p.id}</span>
                    </div>
                </div>
            </td>
            <td class="py-3.5 px-4 text-center">
                ${tierBadgeHtml}
            </td>
            <td class="py-3.5 px-4 text-center">
                <span class="text-sm font-black font-heading text-indigo-700">
                    ${p.power_score}
                </span>
                <span class="text-[10px] text-slate-400 block font-mono" title="Chỉ số RAPM và Hệ số tin cậy mẫu">
                    RAPM: ${p.rapm > 0 ? '+' : ''}${p.rapm}${p.confidence !== undefined ? ` (${Math.round(p.confidence * 100)}%)` : ''}
                </span>
            </td>
            <td class="py-3.5 px-4 text-center font-bold text-xs ${p.winrate >= 50 ? 'text-emerald-600' : 'text-slate-500'}">
                ${p.winrate}%
            </td>
            <td class="py-3.5 px-4">
                <div class="flex flex-wrap items-center justify-center gap-1.5 max-w-[320px] mx-auto">
                    ${extraBadgesHtml || '<span class="text-slate-400 text-xs">-</span>'}
                </div>
            </td>
            <td class="py-3.5 px-4 text-center text-xs text-slate-500 whitespace-nowrap">
                <span title="Số trận thực tế: ${p.matches} | Trận hiệu dụng: ${p.effective_matches !== undefined ? p.effective_matches : p.matches}">
                    ${p.matches} (<span class="text-emerald-600 font-bold">${p.wins}W</span> / <span class="text-rose-600 font-bold">${p.losses}L</span>)
                </span>
                ${p.effective_matches !== undefined ? `
                    <span class="text-[10px] text-slate-400 block font-mono" title="Số trận hiệu dụng sau khi trừ phân rã thời gian (Exponential Recency Decay)">
                        H.Dụng: ${p.effective_matches}
                    </span>
                ` : ''}
            </td>
            <td class="py-3.5 px-4 text-center whitespace-nowrap">
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

