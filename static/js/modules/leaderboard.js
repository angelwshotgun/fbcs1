// TAB 4: LEADERBOARD (BẢNG XẾP HẠNG)
// ==========================================
let currentLeaderboardBadgeFilter = 'all';

function setLeaderboardBadgeFilter(filterKey) {
    currentLeaderboardBadgeFilter = filterKey;

    const filterBtns = [
        'all', 'top1_podium', 'mvp', 'svp', 'clutch_stomp', 'high_wr_vet'
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

    // Chỉ hiển thị tuyển thủ đã thi đấu ít nhất 1 trận (loại bỏ người chơi chưa đánh trận nào)
    let rankedPlayers = allPlayers
        .filter(p => (p.matches || 0) > 0)
        .sort((a, b) => b.hidden_elo - a.hidden_elo);

    // Áp dụng bộ lọc danh hiệu nếu có
    if (currentLeaderboardBadgeFilter !== 'all') {
        rankedPlayers = rankedPlayers.filter(p => {
            const bKeys = (p.badges || []).map(b => b.key);
            if (currentLeaderboardBadgeFilter === 'top1_podium') {
                return bKeys.includes('top1') || bKeys.includes('podium');
            }
            if (currentLeaderboardBadgeFilter === 'mvp') {
                return bKeys.includes('mvp');
            }
            if (currentLeaderboardBadgeFilter === 'svp') {
                return bKeys.includes('svp');
            }
            if (currentLeaderboardBadgeFilter === 'clutch_stomp') {
                return bKeys.includes('clutch') || bKeys.includes('stomp');
            }
            if (currentLeaderboardBadgeFilter === 'high_wr_vet') {
                return bKeys.includes('high_wr') || bKeys.includes('veteran');
            }
            return true;
        });
    }

    const countLabel = document.getElementById('leaderboard-filtered-count');
    if (countLabel) {
        if (currentLeaderboardBadgeFilter === 'all') {
            countLabel.innerText = `Hiển thị toàn bộ ${rankedPlayers.length} tuyển thủ`;
        } else {
            countLabel.innerHTML = `Lọc được: <b class="text-indigo-600 font-bold">${rankedPlayers.length}</b> tuyển thủ`;
        }
    }

    if (rankedPlayers.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center py-12 text-slate-400">
                    <div class="flex flex-col items-center justify-center gap-2">
                        <i class="fa-solid fa-filter text-slate-300 text-3xl"></i>
                        <span class="font-bold text-slate-600 text-sm">Không có tuyển thủ nào khớp với bộ lọc danh hiệu này</span>
                        <p class="text-xs text-slate-400 max-w-md">
                            Hãy thử chọn danh hiệu khác hoặc bấm "Tất Cả" để xem toàn bộ bảng xếp hạng!
                        </p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    rankedPlayers.forEach((p, idx) => {
        const rank = p.rank || (idx + 1);
        let rankBadge = `<span class="font-bold text-slate-500">#${rank}</span>`;
        if (rank === 1) rankBadge = `<span class="w-7 h-7 rounded-full bg-amber-400 text-slate-900 font-black flex items-center justify-center mx-auto shadow-xs">1</span>`;
        if (rank === 2) rankBadge = `<span class="w-7 h-7 rounded-full bg-slate-300 text-slate-900 font-black flex items-center justify-center mx-auto shadow-xs">2</span>`;
        if (rank === 3) rankBadge = `<span class="w-7 h-7 rounded-full bg-amber-700 text-white font-black flex items-center justify-center mx-auto shadow-xs">3</span>`;

        const recentBadges = (p.form?.recent_5 || []).map(r => {
            if (r === 'W') return `<span class="w-5 h-5 rounded-md bg-emerald-100 text-emerald-800 border border-emerald-200 text-[10px] font-bold inline-flex items-center justify-center">W</span>`;
            return `<span class="w-5 h-5 rounded-md bg-rose-100 text-rose-800 border border-rose-200 text-[10px] font-bold inline-flex items-center justify-center">L</span>`;
        }).join('');

        // Render Impact Role (Chính)
        const roleBadgeHtml = `
            <span class="px-2.5 py-0.5 rounded-full text-[11px] font-bold border inline-flex items-center gap-1 ${p.impact_role?.badge || 'bg-slate-100 text-slate-700 border-slate-200'}" title="${p.impact_role?.desc || ''}">
                <span>${p.impact_role?.icon || '⚖️'}</span>
                <span>${p.impact_role?.label || 'Tròn vai'}</span>
            </span>
        `;

        // Render Các Danh Hiệu Đa Dạng Kèm Tooltip
        const extraBadgesHtml = (p.badges || []).map(b => `
            <span class="px-2 py-0.5 rounded-full text-[10px] inline-flex items-center gap-1 border shadow-2xs transition hover:scale-105 cursor-help ${b.badge_class || 'bg-slate-100 text-slate-700 border-slate-200'}" title="${b.desc || b.label}">
                <span>${b.icon}</span>
                <span>${b.label}</span>
            </span>
        `).join('');

        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-50 transition";
        tr.innerHTML = `
            <td class="py-3.5 px-4 text-center">${rankBadge}</td>
            <td class="py-3.5 px-4">
                <div class="flex items-center gap-3">
                    <img src="${p.avatar}" class="w-9 h-9 rounded-xl bg-slate-100 border border-slate-200 object-cover" alt="${p.nickname}">
                    <div>
                        <span class="font-bold text-slate-900 block">${p.nickname}</span>
                        <span class="text-xs text-slate-400">@${p.id}</span>
                    </div>
                </div>
            </td>
            <td class="py-3.5 px-4 text-center font-extrabold text-indigo-600 text-sm">
                ${Math.round(p.hidden_elo)}
            </td>
            <td class="py-3.5 px-4 text-center">
                <span class="px-2.5 py-0.5 rounded-lg bg-slate-100 font-bold text-amber-700 border border-slate-200 text-xs">
                    ${p.stats_ovr || p.skill}
                </span>
            </td>
            <td class="py-3.5 px-4 text-center">
                <span class="font-bold text-slate-800 text-xs">
                    ${p.form?.icon || '🌱'} ${p.form?.score || 5.0}
                </span>
            </td>
            <td class="py-3.5 px-4 text-center font-bold text-xs ${p.winrate >= 50 ? 'text-emerald-600' : 'text-slate-500'}">
                ${p.winrate}%
            </td>
            <td class="py-3.5 px-4">
                <div class="flex flex-wrap items-center justify-center gap-1.5 max-w-[320px] mx-auto">
                    ${roleBadgeHtml}
                    ${extraBadgesHtml}
                </div>
            </td>
            <td class="py-3.5 px-4 text-center text-xs text-slate-500 whitespace-nowrap">
                ${p.matches} (<span class="text-emerald-600 font-bold">${p.wins}</span> / <span class="text-rose-600 font-bold">${p.losses}</span>)
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

