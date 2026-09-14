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
        } else if (p.performance_tag === 'PASSENGER' || p.performance_tag === 'CARRIED') {
            tagBadge = `<span class="px-1.5 py-0.5 rounded bg-purple-100 text-purple-800 font-black border border-purple-300 text-[10px]" title="Hưởng ké chiến thắng từ đồng đội, đóng góp hạn chế">🎒 Hưởng ké</span>`;
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

