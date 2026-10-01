// TAB 3: PLAYER MANAGEMENT (CRUD)
// ==========================================
function renderAdminPlayers() {
    const container = document.getElementById('admin-players-container');
    if (!container) return;
    container.innerHTML = '';

    const query = (document.getElementById('admin-player-search')?.value || '').toLowerCase().trim();
    const sortBy = document.getElementById('admin-sort-by')?.value || 'power';

    let list = [...allPlayers];

    if (sortBy === 'power' || sortBy === 'elo') list.sort((a, b) => b.power_score - a.power_score);
    if (sortBy === 'tier') {
        const tierRank = { 'S': 4, 'A': 3, 'B': 2, 'C': 1, 'NEW': 0 };
        list.sort((a, b) => (tierRank[b.tier] || 0) - (tierRank[a.tier] || 0) || b.power_score - a.power_score);
    }
    if (sortBy === 'winrate') list.sort((a, b) => b.winrate - a.winrate);
    if (sortBy === 'matches') list.sort((a, b) => b.matches - a.matches);
    if (sortBy === 'name') list.sort((a, b) => a.nickname.localeCompare(b.nickname));

    list.forEach(p => {
        if (query && !p.id.toLowerCase().includes(query) && !p.nickname.toLowerCase().includes(query)) {
            return;
        }

        const card = document.createElement('div');
        card.className = "bg-white border border-slate-200 hover:border-slate-300 p-4 rounded-2xl flex flex-col justify-between space-y-3.5 transition shadow-xs hover:shadow-sm";

        const recentBadges = (p.recent_5 || []).map(r => {
            if (r === 'W') return `<span class="w-4 h-4 rounded bg-emerald-100 text-emerald-800 text-[10px] font-bold inline-flex items-center justify-center">W</span>`;
            return `<span class="w-4 h-4 rounded bg-rose-100 text-rose-800 text-[10px] font-bold inline-flex items-center justify-center">L</span>`;
        }).join('');

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
                        <span class="px-2.5 py-0.5 rounded-full ${p.tier_badge_class || 'bg-slate-100 text-slate-700'} text-xs font-bold shadow-2xs">
                            ${p.tier_icon || '🛡️'} ${p.tier_name || 'Tier B'}
                        </span>
                    </div>
                </div>

                <!-- Power Score Box -->
                <div class="mt-3.5 p-2.5 rounded-xl bg-slate-50 border border-slate-100 flex items-center justify-between text-xs">
                    <span class="text-slate-500 font-medium">Điểm Thực Lực:</span>
                    <span class="font-black text-indigo-700 font-heading text-sm">
                        ${p.power_score} <span class="text-[10px] font-normal text-slate-400">/ 100</span>
                    </span>
                </div>

                <!-- Match Stats -->
                <div class="flex items-center justify-between text-[11px] text-slate-500 mt-2 px-1">
                    <span>Trận: <b class="text-slate-800">${p.matches}</b> (<b class="text-emerald-600">${p.wins}W</b> / <b class="text-rose-600">${p.losses}L</b>)</span>
                    <span>Tỷ lệ thắng: <b class="${p.winrate >= 50 ? 'text-emerald-600' : 'text-slate-600'} font-bold">${p.winrate}%</b></span>
                </div>

                <!-- Recent 5 matches -->
                <div class="flex items-center justify-between text-[10px] text-slate-400 mt-2 px-1 pt-2 border-t border-slate-100">
                    <span>Gần nhất:</span>
                    <div class="flex items-center gap-1">${recentBadges || '<span class="text-slate-300">Chưa đấu</span>'}</div>
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
        if (document.getElementById('form-duck-image')) document.getElementById('form-duck-image').value = '';
        if (document.getElementById('form-duck-file')) document.getElementById('form-duck-file').value = '';
        if (document.getElementById('form-duck-preview')) document.getElementById('form-duck-preview').src = '/static/images/duck_default.svg';
        document.getElementById('form-display-badge').innerText = '🌱 Tân Binh';
        document.getElementById('form-display-power').innerText = '50.0';
        document.getElementById('form-display-record').innerText = '0 (0/0)';
    } else {
        title.innerHTML = '<i class="fa-solid fa-user-pen text-indigo-600"></i><span>Chỉnh Sửa Tuyển Thủ</span>';
        idInput.disabled = true;
        const p = allPlayers.find(x => x.id === playerId);
        if (!p) return;

        idInput.value = p.id;
        document.getElementById('form-nickname').value = p.nickname;
        document.getElementById('form-avatar').value = p.avatar;
        document.getElementById('form-avatar-preview').src = p.avatar;
        if (document.getElementById('form-duck-image')) document.getElementById('form-duck-image').value = p.duck_image || '';
        if (document.getElementById('form-duck-file')) document.getElementById('form-duck-file').value = '';
        if (document.getElementById('form-duck-preview')) document.getElementById('form-duck-preview').src = p.duck_image || '/static/images/duck_default.svg';

        document.getElementById('form-display-badge').innerText = `${p.tier_icon || '🛡️'} ${p.tier_name || 'Tier B'}`;
        document.getElementById('form-display-power').innerText = `${p.power_score || 50.0}`;
        document.getElementById('form-display-record').innerText = `${p.matches || 0} (${p.wins || 0}W / ${p.losses || 0}L)`;
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

function updateDuckPreview(url) {
    const preview = document.getElementById('form-duck-preview');
    if (preview) {
        preview.src = url ? url : '/static/images/duck_default.svg';
    }
}

function resetDefaultDuckImage() {
    const input = document.getElementById('form-duck-image');
    const fileInput = document.getElementById('form-duck-file');
    const preview = document.getElementById('form-duck-preview');
    if (input) input.value = '';
    if (fileInput) fileInput.value = '';
    if (preview) preview.src = '/static/images/duck_default.svg';
}

function handleDuckFileSelect(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    if (file.size > 2 * 1024 * 1024) {
        Swal.fire({
            icon: 'warning',
            title: 'File quá lớn',
            text: 'Vui lòng chọn ảnh dung lượng dưới 2MB để đảm bảo hiệu năng Canvas.',
            ...SWAL_THEME
        });
        event.target.value = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        const dataUrl = e.target.result;
        const input = document.getElementById('form-duck-image');
        const preview = document.getElementById('form-duck-preview');
        if (input) input.value = dataUrl;
        if (preview) preview.src = dataUrl;
    };
    reader.readAsDataURL(file);
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
    const duck_image = document.getElementById('form-duck-image') ? document.getElementById('form-duck-image').value.trim() : '';

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
        duck_image
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


function switchAdminSubTab(subTab) {
    currentAdminSubTab = subTab;
    const btnPlayers = document.getElementById('admin-subtab-btn-players');
    const btnMatches = document.getElementById('admin-subtab-btn-matches');
    const viewPlayers = document.getElementById('admin-subtab-players-view');
    const viewMatches = document.getElementById('admin-subtab-matches-view');

    if (!btnPlayers || !btnMatches || !viewPlayers || !viewMatches) return;

    if (subTab === 'players') {
        btnPlayers.className = "py-2 px-4 rounded-xl text-xs font-bold font-heading transition flex items-center gap-2 bg-white text-indigo-700 shadow-xs";
        btnMatches.className = "py-2 px-4 rounded-xl text-xs font-bold font-heading transition flex items-center gap-2 text-slate-600 hover:text-slate-900";
        viewPlayers.classList.remove('hidden');
        viewMatches.classList.add('hidden');
        renderAdminPlayers();
    } else {
        btnMatches.className = "py-2 px-4 rounded-xl text-xs font-bold font-heading transition flex items-center gap-2 bg-white text-purple-700 shadow-xs";
        btnPlayers.className = "py-2 px-4 rounded-xl text-xs font-bold font-heading transition flex items-center gap-2 text-slate-600 hover:text-slate-900";
        viewMatches.classList.remove('hidden');
        viewPlayers.classList.add('hidden');
        loadAdminMatches();
    }
}

async function loadBalanceReport() {
    try {
        const res = await fetch('/api/balance_report?n=20');
        const data = await res.json();
        if (!data.success) return;

        const dist = data.rating_distribution || {};
        const total = (dist.perfect || 0) + (dist.fair || 0) + (dist.unbalanced || 0) + (dist.stomp || 0);

        const perfectEl = document.getElementById('stat-perfect-count');
        const fairEl = document.getElementById('stat-fair-count');
        const unbalEl = document.getElementById('stat-unbalanced-count');
        const stompEl = document.getElementById('stat-stomp-count');

        if (perfectEl) perfectEl.innerText = dist.perfect || 0;
        if (fairEl) fairEl.innerText = dist.fair || 0;
        if (unbalEl) unbalEl.innerText = dist.unbalanced || 0;
        if (stompEl) stompEl.innerText = dist.stomp || 0;

        const calcRate = (cnt) => total > 0 ? `${Math.round((cnt / total) * 100)}%` : '0%';
        const pRate = document.getElementById('stat-perfect-rate');
        const fRate = document.getElementById('stat-fair-rate');
        const uRate = document.getElementById('stat-unbalanced-rate');
        const sRate = document.getElementById('stat-stomp-rate');

        if (pRate) pRate.innerText = calcRate(dist.perfect || 0);
        if (fRate) fRate.innerText = calcRate(dist.fair || 0);
        if (uRate) uRate.innerText = calcRate(dist.unbalanced || 0);
        if (sRate) sRate.innerText = calcRate(dist.stomp || 0);

        const avgEl = document.getElementById('stat-avg-closeness');
        if (avgEl) {
            avgEl.innerText = data.matches_with_kill_data > 0 ? `${Math.round(data.average_closeness * 100)}%` : 'Chưa có số liệu';
        }

        const trendText = document.getElementById('balance-trend-text');
        const trendBadge = document.getElementById('balance-trend-badge');
        if (trendText && trendBadge) {
            if (data.trend === 'improving') {
                trendText.innerText = 'Xu hướng: Đang cân bằng tốt hơn';
                trendBadge.className = 'px-3 py-1 rounded-full text-xs font-bold border flex items-center gap-1.5 bg-emerald-50 text-emerald-700 border-emerald-200';
            } else if (data.trend === 'declining') {
                trendText.innerText = 'Xu hướng: Có dấu hiệu lệch kèo';
                trendBadge.className = 'px-3 py-1 rounded-full text-xs font-bold border flex items-center gap-1.5 bg-rose-50 text-rose-700 border-rose-200';
            } else {
                trendText.innerText = 'Xu hướng: Ổn định';
                trendBadge.className = 'px-3 py-1 rounded-full text-xs font-bold border flex items-center gap-1.5 bg-slate-50 text-slate-600 border-slate-200';
            }
        }
    } catch (e) {
        console.warn('[BalanceReport] Failed to load:', e);
    }
}

async function loadAdminMatches() {
    const tbody = document.getElementById('admin-matches-table-body');
    if (!tbody) return;

    // Load Balance Report stats concurrently
    loadBalanceReport();

    tbody.innerHTML = `
        <tr>
            <td colspan="7" class="py-8 text-center text-slate-400">
                <i class="fa-solid fa-spinner fa-spin text-xl mb-2 text-indigo-600"></i>
                <p>Đang tải danh sách lịch sử trận đấu...</p>
            </td>
        </tr>
    `;

    try {
        const res = await fetch('/api/matches');
        const data = await res.json();
        if (data.success) {
            allAdminMatches = data.matches || [];
            renderAdminMatchesTable(allAdminMatches);
            const countBadge = document.getElementById('admin-matches-count-badge');
            if (countBadge) countBadge.innerText = allAdminMatches.length;
            const totalBadge = document.getElementById('admin-total-matches-badge');
            if (totalBadge) totalBadge.innerText = `${allAdminMatches.length} Trận Đấu`;
        } else {
            tbody.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-rose-500 font-bold">${data.error || 'Lỗi tải lịch sử trận đấu'}</td></tr>`;
        }
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-rose-500 font-bold">${err.message}</td></tr>`;
    }
}

function getBalanceBadgeHtml(match) {
    const t1k = match.team1_kills || 0;
    const t2k = match.team2_kills || 0;
    if (t1k === 0 && t2k === 0) return '';
    
    const rating = match.balance_rating || 'unknown';
    const icons = { perfect: '🟢', fair: '🟡', unbalanced: '🟠', stomp: '🔴', unknown: '⚪' };
    const labels = { perfect: 'Sát nút', fair: 'Cân bằng', unbalanced: 'Lệch kèo', stomp: 'Hủy diệt', unknown: '' };
    const colors = { perfect: 'bg-emerald-50 text-emerald-700 border-emerald-200', fair: 'bg-amber-50 text-amber-700 border-amber-200', unbalanced: 'bg-orange-50 text-orange-700 border-orange-200', stomp: 'bg-rose-50 text-rose-700 border-rose-200', unknown: 'bg-slate-50 text-slate-500 border-slate-200' };
    
    const icon = icons[rating] || '⚪';
    const label = labels[rating] || '';
    const colorClass = colors[rating] || colors.unknown;
    
    return `
        <div class="flex items-center gap-2 mt-1">
            <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-xs font-bold border ${colorClass}">
                <i class="fa-solid fa-skull-crossbones text-[10px]"></i>
                ${t1k} - ${t2k}
            </span>
            ${label ? `<span class="text-[10px] font-medium ${colorClass.split(' ')[1]}">${icon} ${label}</span>` : ''}
        </div>
    `;
}

function renderAdminMatchesTable(matches) {
    const tbody = document.getElementById('admin-matches-table-body');
    if (!tbody) return;

    if (!matches || matches.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" class="py-12 text-center text-slate-400">
                    <i class="fa-solid fa-gamepad text-3xl mb-2 text-slate-300"></i>
                    <p class="font-bold">Chưa có trận đấu nào được ghi nhận</p>
                    <p class="text-xs">Hãy vào mục "Chia Đội" hoặc "Mô Phỏng 5vs5" để bắt đầu ghi nhận kết quả!</p>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = '';
    matches.forEach(m => {
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-50/80 transition';

        const isT1Win = m.winner === 'team1';
        const winnerBadge = isT1Win
            ? '<span class="px-2.5 py-1 rounded-full bg-blue-100 text-blue-800 font-bold text-[11px] border border-blue-200">Đội Xanh Thắng</span>'
            : '<span class="px-2.5 py-1 rounded-full bg-rose-100 text-rose-800 font-bold text-[11px] border border-rose-200">Đội Đỏ Thắng</span>';

        // Helper: render ELO delta badge
        const deltasMap = m.player_deltas || {};
        function eloDeltaBadge(playerId) {
            const d = deltasMap[playerId];
            if (d === undefined || d === null) return '';
            const val = Number(d);
            if (val > 0) return `<span class="ml-1 text-[10px] font-black text-emerald-600 bg-emerald-50 border border-emerald-200 rounded px-1 py-px leading-none">+${val}</span>`;
            if (val < 0) return `<span class="ml-1 text-[10px] font-black text-rose-600 bg-rose-50 border border-rose-200 rounded px-1 py-px leading-none">${val}</span>`;
            return `<span class="ml-1 text-[10px] font-black text-slate-400 bg-slate-50 border border-slate-200 rounded px-1 py-px leading-none">±0</span>`;
        }

        // Render Team 1 mini avatars + ELO delta
        const t1PlayersHtml = (m.team1 || []).map(p => `
            <div class="flex items-center gap-1.5 py-0.5" title="${p.nickname}">
                <img src="${p.avatar || `https://api.dicebear.com/7.x/bottts/svg?seed=${p.id}`}" class="w-5 h-5 rounded-md object-cover bg-slate-100 border border-slate-200 flex-shrink-0">
                <span class="font-bold text-slate-800 text-[11px] truncate max-w-[100px]">${p.nickname}</span>
                ${eloDeltaBadge(p.id)}
            </div>
        `).join('');

        // Render Team 2 mini avatars + ELO delta
        const t2PlayersHtml = (m.team2 || []).map(p => `
            <div class="flex items-center gap-1.5 py-0.5" title="${p.nickname}">
                <img src="${p.avatar || `https://api.dicebear.com/7.x/bottts/svg?seed=${p.id}`}" class="w-5 h-5 rounded-md object-cover bg-slate-100 border border-slate-200 flex-shrink-0">
                <span class="font-bold text-slate-800 text-[11px] truncate max-w-[100px]">${p.nickname}</span>
                ${eloDeltaBadge(p.id)}
            </div>
        `).join('');

        // Format date
        let dateStr = '-';
        if (m.created_at) {
            try {
                const d = new Date(m.created_at);
                dateStr = d.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
            } catch (e) {
                dateStr = m.created_at;
            }
        }

        const noteText = m.notes || m.ai_summary || '-';

        const balanceBadgeHtml = getBalanceBadgeHtml(m);

        tr.innerHTML = `
            <td class="py-3.5 px-4 text-center font-mono font-bold text-slate-600">#${m.id}</td>
            <td class="py-3.5 px-4 whitespace-nowrap text-slate-500 font-medium">${dateStr}</td>
            <td class="py-3.5 px-4"><div class="space-y-0.5">${t1PlayersHtml}</div></td>
            <td class="py-3.5 px-4"><div class="space-y-0.5">${t2PlayersHtml}</div></td>
            <td class="py-3.5 px-4 text-center whitespace-nowrap">
                <div class="flex flex-col items-center gap-1">
                    ${winnerBadge}
                    ${balanceBadgeHtml}
                </div>
            </td>
            <td class="py-3.5 px-4 text-slate-600 max-w-[220px]">
                <div class="truncate text-[11px]" title="${noteText}">${noteText}</div>
            </td>
            <td class="py-3.5 px-4 text-center whitespace-nowrap">
                <div class="flex items-center justify-center gap-1.5">
                    <button onclick="openEditMatchModal(${m.id})" class="px-2.5 py-1.5 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-bold transition flex items-center gap-1">
                        <i class="fa-solid fa-pen-to-square"></i> Sửa
                    </button>
                    <button onclick="handleDeleteMatch(${m.id})" class="px-2 py-1.5 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-700 font-bold transition">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function openEditMatchModal(matchId) {
    const match = allAdminMatches.find(m => m.id === matchId);
    if (!match) return;

    document.getElementById('edit-match-id').value = match.id;
    document.getElementById('edit-match-id-badge').innerText = match.id;
    document.getElementById('edit-match-code-badge').innerText = match.match_code || `M-${match.id}`;
    document.getElementById('edit-match-notes').value = match.notes || '';
    const t1kInput = document.getElementById('edit-team1-kills');
    const t2kInput = document.getElementById('edit-team2-kills');
    if (t1kInput) t1kInput.value = (match.team1_kills !== undefined && match.team1_kills !== null) ? match.team1_kills : '';
    if (t2kInput) t2kInput.value = (match.team2_kills !== undefined && match.team2_kills !== null) ? match.team2_kills : '';

    // Set winner radio
    if (match.winner === 'team1') {
        document.getElementById('edit-winner-t1').checked = true;
    } else {
        document.getElementById('edit-winner-t2').checked = true;
    }

    // Populate the 10 slots
    const t1Pids = match.team1_players || [];
    const t2Pids = match.team2_players || [];

    for (let i = 0; i < 5; i++) {
        populatePlayerDropdown(`edit-t1-slot-${i}`, t1Pids[i] || '');
        populatePlayerDropdown(`edit-t2-slot-${i}`, t2Pids[i] || '');
    }

    document.getElementById('edit-match-modal').classList.remove('hidden');
}

function populatePlayerDropdown(selectId, selectedPid) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    sel.innerHTML = '';

    const emptyOpt = document.createElement('option');
    emptyOpt.value = '';
    emptyOpt.innerText = '-- Chọn tuyển thủ --';
    sel.appendChild(emptyOpt);

    allPlayers.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.innerText = `${p.nickname} (Elo ${Math.round(p.hidden_elo)})`;
        if (String(p.id).toLowerCase() === String(selectedPid || '').toLowerCase()) {
            opt.selected = true;
        }
        sel.appendChild(opt);
    });
}

function closeEditMatchModal() {
    const modal = document.getElementById('edit-match-modal');
    if (modal) modal.classList.add('hidden');
}

async function handleSaveEditMatch() {
    const matchId = document.getElementById('edit-match-id').value;
    if (!matchId) return;

    const team1 = [];
    const team2 = [];
    for (let i = 0; i < 5; i++) {
        const val1 = document.getElementById(`edit-t1-slot-${i}`)?.value;
        const val2 = document.getElementById(`edit-t2-slot-${i}`)?.value;
        if (val1) team1.push(val1);
        if (val2) team2.push(val2);
    }

    if (team1.length !== 5 || team2.length !== 5) {
        Swal.fire({
            icon: 'warning',
            title: 'Chưa đủ 10 người',
            text: 'Vui lòng chọn đủ 5 tuyển thủ cho Đội 1 và 5 tuyển thủ cho Đội 2.',
            ...SWAL_THEME
        });
        return;
    }

    const allChosen = [...team1, ...team2];
    if (new Set(allChosen).size !== 10) {
        Swal.fire({
            icon: 'error',
            title: 'Trùng lặp tuyển thủ',
            text: 'Có người chơi bị chọn nhiều lần giữa 2 đội. Vui lòng kiểm tra lại.',
            ...SWAL_THEME
        });
        return;
    }

    const winner = document.getElementById('edit-winner-t1')?.checked ? 'team1' : 'team2';
    const notes = document.getElementById('edit-match-notes')?.value || '';
    const team1_kills = parseInt(document.getElementById('edit-team1-kills')?.value) || 0;
    const team2_kills = parseInt(document.getElementById('edit-team2-kills')?.value) || 0;

    const btn = document.getElementById('btn-save-edit-match');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang cập nhật...';
    }

    try {
        const res = await fetch(`/api/matches/${matchId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team1, team2, winner, notes, team1_kills, team2_kills })
        });
        const data = await res.json();
        if (!data.success) {
            throw new Error(data.error || 'Lỗi cập nhật trận đấu');
        }

        closeEditMatchModal();
        Swal.fire({
            icon: 'success',
            title: 'Đã cập nhật trận đấu!',
            text: 'Hệ thống đã tính toán lại toàn bộ Elo, phong độ và tỷ lệ thắng.',
            timer: 2000,
            showConfirmButton: false,
            ...SWAL_THEME
        });

        await loadAdminMatches();
        await loadAllPlayers();
        await loadLeaderboard();
        await loadStatus();
    } catch (err) {
        Swal.fire({ icon: 'error', title: 'Lỗi', text: err.message, ...SWAL_THEME });
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> <span>Lưu Thay Đổi & Tính Lại Elo</span>';
        }
    }
}

async function handleDeleteMatch(matchId) {
    const confirm = await Swal.fire({
        title: `Xác nhận xóa Trận #${matchId}?`,
        text: 'Trận đấu này sẽ bị xóa vĩnh viễn khỏi CSDL. Hệ thống sẽ tự động tính toán lại toàn bộ điểm Elo và phong độ từ trước tới nay!',
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: 'Đồng Ý Xóa',
        cancelButtonText: 'Hủy',
        confirmButtonColor: '#e11d48',
        ...SWAL_THEME
    });

    if (!confirm.isConfirmed) return;

    try {
        const res = await fetch(`/api/matches/${matchId}`, { method: 'DELETE' });
        const data = await res.json();
        if (!data.success) {
            throw new Error(data.error || 'Lỗi khi xóa trận đấu');
        }

        Swal.fire({
            icon: 'success',
            title: 'Đã xóa trận đấu!',
            text: 'Toàn bộ điểm Elo và phong độ đã được tính toán lại thành công.',
            timer: 2000,
            showConfirmButton: false,
            ...SWAL_THEME
        });

        await loadAdminMatches();
        await loadAllPlayers();
        await loadLeaderboard();
        await loadStatus();
    } catch (err) {
        Swal.fire({ icon: 'error', title: 'Lỗi xóa trận', text: err.message, ...SWAL_THEME });
    }
}

