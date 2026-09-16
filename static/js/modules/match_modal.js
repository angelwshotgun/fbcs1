// ==========================================
// MATCH RESULT & SCOREBOARD AI MODAL
// ==========================================

function openMatchResultModal(winningTeam) {
    setMatchModalWinner(winningTeam);
    const modal = document.getElementById('match-result-modal');
    if (!modal) return;

    // Reset default view
    switchMatchResultTab('standard');
    removeScoreboardImage();

    modal.classList.remove('hidden');
}

function setMatchModalWinner(winningTeam) {
    currentMatchModalWinner = winningTeam;

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

    const t1Btn = document.getElementById('modal-toggle-t1');
    const t2Btn = document.getElementById('modal-toggle-t2');
    if (t1Btn && t2Btn) {
        if (isTeam1) {
            t1Btn.className = "px-2.5 py-1 rounded-lg text-xs font-bold font-heading bg-blue-600 text-white shadow-xs transition";
            t2Btn.className = "px-2.5 py-1 rounded-lg text-xs font-bold font-heading text-slate-600 hover:text-slate-900 transition";
        } else {
            t2Btn.className = "px-2.5 py-1 rounded-lg text-xs font-bold font-heading bg-rose-600 text-white shadow-xs transition";
            t1Btn.className = "px-2.5 py-1 rounded-lg text-xs font-bold font-heading text-slate-600 hover:text-slate-900 transition";
        }
    }

    if (currentAiScoreboardAnalysis) {
        renderScoreboardAiResults(currentAiScoreboardAnalysis);
    }
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
        const viewFullBtn = document.getElementById('scoreboard-view-full-btn');

        if (emptyArea) emptyArea.classList.add('hidden');
        if (previewContainer) previewContainer.classList.remove('hidden');
        if (previewImg) previewImg.src = currentScoreboardImageBase64;
        if (fileName) fileName.innerText = file.name || 'Ảnh bảng điểm vừa dán';
        if (viewFullBtn) viewFullBtn.href = currentScoreboardImageBase64;

        // Tự động kích hoạt nút phân tích
        const runBtn = document.getElementById('btn-run-scoreboard-ai');
        if (runBtn) runBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function toggleScoreboardImageSize() {
    const wrapper = document.getElementById('scoreboard-img-wrapper');
    const img = document.getElementById('scoreboard-preview-img');
    const txt = document.getElementById('txt-toggle-img-size');
    if (!wrapper || !img) return;

    if (wrapper.classList.contains('max-h-72')) {
        wrapper.classList.remove('max-h-72');
        wrapper.classList.add('max-h-[550px]');
        img.classList.remove('max-h-72');
        img.classList.add('max-h-[550px]');
        if (txt) txt.innerText = 'Thu Gọn Ảnh';
    } else {
        wrapper.classList.remove('max-h-[550px]');
        wrapper.classList.add('max-h-72');
        img.classList.remove('max-h-[550px]');
        img.classList.add('max-h-72');
        if (txt) txt.innerText = 'Mở Rộng Ảnh';
    }
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
    const viewFullBtn = document.getElementById('scoreboard-view-full-btn');

    if (emptyArea) emptyArea.classList.remove('hidden');
    if (previewContainer) previewContainer.classList.add('hidden');
    if (resultsContainer) resultsContainer.classList.add('hidden');
    if (loadingElem) loadingElem.classList.add('hidden');
    if (controlsElem) controlsElem.classList.remove('hidden');
    if (viewFullBtn) viewFullBtn.href = '#';

    // Reset toggle image state
    const wrapper = document.getElementById('scoreboard-img-wrapper');
    const img = document.getElementById('scoreboard-preview-img');
    const txt = document.getElementById('txt-toggle-img-size');
    if (wrapper && img) {
        wrapper.classList.remove('max-h-[550px]');
        wrapper.classList.add('max-h-72');
        img.classList.remove('max-h-[550px]');
        img.classList.add('max-h-72');
        if (txt) txt.innerText = 'Mở Rộng Ảnh';
    }
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
    const team1List = list.filter(p => p.team === 1 || p.team === '1' || p.team === 'team1');
    const team2List = list.filter(p => !(p.team === 1 || p.team === '1' || p.team === 'team1'));

    const renderPlayerRow = (p, isTeam1) => {
        const playerObj = allPlayers.find(item => item.id === p.player_id) || {};
        const isWinner = (isTeam1 && currentMatchModalWinner === 'team1') || (!isTeam1 && currentMatchModalWinner === 'team2');

        let tagBadge = `<span class="px-2 py-0.5 rounded-lg bg-slate-100 text-slate-700 font-bold text-[11px] whitespace-nowrap inline-block">Tròn vai</span>`;
        if (p.performance_tag === 'MVP') {
            tagBadge = `<span class="px-2 py-0.5 rounded-lg bg-amber-100 text-amber-800 font-black border border-amber-300 text-[11px] shadow-2xs whitespace-nowrap inline-block">👑 MVP</span>`;
        } else if (p.performance_tag === 'SVP') {
            tagBadge = `<span class="px-2 py-0.5 rounded-lg bg-indigo-100 text-indigo-800 font-black border border-indigo-300 text-[11px] shadow-2xs whitespace-nowrap inline-block">⭐ SVP</span>`;
        } else if (p.performance_tag === 'GREAT') {
            tagBadge = `<span class="px-2 py-0.5 rounded-lg bg-emerald-100 text-emerald-800 font-bold border border-emerald-300 text-[11px] whitespace-nowrap inline-block">🔥 Tốt</span>`;
        } else if (p.performance_tag === 'UNDERPERFORMING') {
            tagBadge = `<span class="px-2 py-0.5 rounded-lg bg-amber-50 text-amber-700 font-medium border border-amber-200 text-[11px] whitespace-nowrap inline-block">⚠️ Dưới sức</span>`;
        } else if (p.performance_tag === 'PASSENGER' || p.performance_tag === 'CARRIED') {
            tagBadge = `<span class="px-2 py-0.5 rounded-lg bg-purple-100 text-purple-800 font-black border border-purple-300 text-[11px] whitespace-nowrap inline-block" title="Hưởng ké chiến thắng từ đồng đội, đóng góp hạn chế">🎒 Hưởng ké</span>`;
        } else if (p.performance_tag === 'FEEDER') {
            tagBadge = `<span class="px-2 py-0.5 rounded-lg bg-rose-100 text-rose-800 font-bold border border-rose-300 text-[11px] whitespace-nowrap inline-block">💀 Thọt</span>`;
        }

        const deltaVal = (p.recommended_delta !== undefined && p.recommended_delta !== null)
            ? p.recommended_delta
            : (isWinner ? 16 : -16);
        const inputColor = deltaVal >= 0 
            ? 'text-emerald-700 bg-emerald-50/70 border-emerald-300 focus:border-emerald-500' 
            : 'text-rose-700 bg-rose-50/70 border-rose-300 focus:border-rose-500';

        const tr = document.createElement('tr');
        tr.className = `hover:bg-slate-50/80 transition ${isTeam1 ? 'bg-blue-50/15' : 'bg-rose-50/15'}`;
        tr.innerHTML = `
            <td class="py-2.5 px-3.5">
                <div class="flex items-center gap-2.5">
                    <img src="${playerObj.avatar || `https://api.dicebear.com/7.x/bottts/svg?seed=${p.player_id}`}" class="w-8 h-8 rounded-xl object-cover bg-slate-100 border border-slate-200 flex-shrink-0 shadow-2xs">
                    <div class="min-w-0">
                        <span class="font-bold text-slate-900 block truncate text-xs">${p.nickname}</span>
                        <span class="text-[10px] text-slate-400 block font-mono truncate">ID: ${p.player_id}</span>
                    </div>
                </div>
            </td>
            <td class="py-2.5 px-2.5 text-center font-semibold text-slate-700">
                ${p.champion && p.champion !== '-' ? `<span class="px-2 py-1 rounded-lg bg-slate-100 border border-slate-200 text-xs font-bold inline-block">${p.champion}</span>` : '<span class="text-slate-400">-</span>'}
            </td>
            <td class="py-2.5 px-2.5 text-center font-bold text-slate-800 whitespace-nowrap text-xs">
                ${p.kda || '-'}
            </td>
            <td class="py-2.5 px-2.5 text-center font-medium text-slate-600 whitespace-nowrap text-xs">
                ${p.damage && p.damage !== '-' ? `<span class="text-slate-800 font-bold">${p.damage}</span>` : '<span class="text-slate-400">-</span>'}
            </td>
            <td class="py-2.5 px-2.5 text-center whitespace-nowrap">
                ${tagBadge}
            </td>
            <td class="py-2.5 px-3.5 text-slate-600 text-xs leading-relaxed">
                <div class="font-medium text-slate-700 break-words">${p.comment || '<span class="text-slate-400 italic">Không có nhận xét thêm</span>'}</div>
            </td>
            <td class="py-2.5 px-3.5 text-center">
                <div class="flex items-center justify-center">
                    <input type="number" step="0.5" id="ai-delta-${p.player_id}" value="${deltaVal}" class="w-20 text-center font-black py-1.5 px-2 rounded-xl border text-xs shadow-2xs focus:outline-none focus:ring-2 focus:ring-indigo-500/20 ${inputColor}">
                </div>
            </td>
        `;
        return tr;
    };

    // Render Team 1 Header & Players
    const t1Header = document.createElement('tr');
    t1Header.className = "bg-blue-100/60 border-y border-blue-200 text-blue-900 font-extrabold text-xs";
    t1Header.innerHTML = `
        <td colspan="7" class="py-2 px-3.5">
            <div class="flex items-center justify-between">
                <span class="flex items-center gap-2 font-heading">
                    <span class="w-2.5 h-2.5 rounded-full bg-blue-600 inline-block"></span>
                    <span>ĐỘI XANH (TEAM 1)</span>
                </span>
                <span class="px-2.5 py-0.5 rounded-md text-[10px] font-black uppercase ${currentMatchModalWinner === 'team1' ? 'bg-blue-600 text-white shadow-2xs' : 'bg-slate-200 text-slate-700'}">
                    ${currentMatchModalWinner === 'team1' ? '🏆 Đội Thắng (+)' : 'Đội Thua (-)'}
                </span>
            </div>
        </td>
    `;
    tbody.appendChild(t1Header);
    team1List.forEach(p => tbody.appendChild(renderPlayerRow(p, true)));

    // Render Team 2 Header & Players
    const t2Header = document.createElement('tr');
    t2Header.className = "bg-rose-100/60 border-y border-rose-200 text-rose-900 font-extrabold text-xs";
    t2Header.innerHTML = `
        <td colspan="7" class="py-2 px-3.5">
            <div class="flex items-center justify-between">
                <span class="flex items-center gap-2 font-heading">
                    <span class="w-2.5 h-2.5 rounded-full bg-rose-600 inline-block"></span>
                    <span>ĐỘI ĐỎ (TEAM 2)</span>
                </span>
                <span class="px-2.5 py-0.5 rounded-md text-[10px] font-black uppercase ${currentMatchModalWinner === 'team2' ? 'bg-rose-600 text-white shadow-2xs' : 'bg-slate-200 text-slate-700'}">
                    ${currentMatchModalWinner === 'team2' ? '🏆 Đội Thắng (+)' : 'Đội Thua (-)'}
                </span>
            </div>
        </td>
    `;
    tbody.appendChild(t2Header);
    team2List.forEach(p => tbody.appendChild(renderPlayerRow(p, false)));

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

