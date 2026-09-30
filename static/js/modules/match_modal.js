// ==========================================
// MATCH RESULT MODAL (THUẦN KẾT QUẢ THẮNG/THUA - TỨC THÌ)
// ==========================================

// Trạng thái modal (currentMatchModalWinner, isSubmittingMatch) được quản lý tập trung tại state.js
if (typeof currentMatchModalWinner === 'undefined') {
    var currentMatchModalWinner = 'team1';
}
if (typeof isSubmittingMatch === 'undefined') {
    var isSubmittingMatch = false;
}

function openMatchResultModal(winningTeam) {
    setMatchModalWinner(winningTeam || 'team1');
    const modal = document.getElementById('match-result-modal');
    if (!modal) return;

    // Reset kill score inputs
    const t1Input = document.getElementById('input-team1-kills');
    const t2Input = document.getElementById('input-team2-kills');
    const preview = document.getElementById('kill-score-preview');
    if (t1Input) t1Input.value = '';
    if (t2Input) t2Input.value = '';
    if (preview) preview.classList.add('hidden');

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
        badgeIcon.className = `w-11 h-11 rounded-2xl ${bgBadgeClass} flex items-center justify-center text-xl font-bold shadow-xs`;
    }

    const stdWinnerName = document.getElementById('modal-std-winner-name');
    if (stdWinnerName) stdWinnerName.innerText = winnerName;
    const stdLoserName = document.getElementById('modal-std-loser-name');
    if (stdLoserName) stdLoserName.innerText = loserName;

    const t1Btn = document.getElementById('modal-toggle-t1');
    const t2Btn = document.getElementById('modal-toggle-t2');
    if (t1Btn && t2Btn) {
        if (isTeam1) {
            t1Btn.className = "py-3 px-4 rounded-2xl text-xs font-black font-heading bg-blue-600 text-white shadow-md shadow-blue-500/20 transition flex items-center justify-center gap-2";
            t2Btn.className = "py-3 px-4 rounded-2xl text-xs font-black font-heading bg-slate-100 text-slate-700 hover:bg-slate-200 transition flex items-center justify-center gap-2";
        } else {
            t2Btn.className = "py-3 px-4 rounded-2xl text-xs font-black font-heading bg-rose-600 text-white shadow-md shadow-rose-500/20 transition flex items-center justify-center gap-2";
            t1Btn.className = "py-3 px-4 rounded-2xl text-xs font-black font-heading bg-slate-100 text-slate-700 hover:bg-slate-200 transition flex items-center justify-center gap-2";
        }
    }
}

function closeMatchResultModal() {
    const modal = document.getElementById('match-result-modal');
    if (modal) modal.classList.add('hidden');
}

function updateKillScorePreview() {
    const t1Input = document.getElementById('input-team1-kills');
    const t2Input = document.getElementById('input-team2-kills');
    const preview = document.getElementById('kill-score-preview');
    if (!t1Input || !t2Input || !preview) return;

    const t1k = parseInt(t1Input.value) || 0;
    const t2k = parseInt(t2Input.value) || 0;

    if (t1k === 0 && t2k === 0) {
        preview.classList.add('hidden');
        return;
    }

    const total = t1k + t2k;
    const diff = Math.abs(t1k - t2k);
    const closeness = total > 0 ? (1.0 - diff / total) : 0.5;

    let rating, icon, color;
    if (diff <= 5 || closeness >= 0.85) {
        rating = 'Sát Nút'; icon = '🟢'; color = 'text-emerald-600';
    } else if (diff <= 10 || closeness >= 0.60) {
        rating = 'Cân Bằng'; icon = '🟡'; color = 'text-amber-600';
    } else if (diff <= 15 || closeness >= 0.40) {
        rating = 'Lệch Kèo'; icon = '🟠'; color = 'text-orange-600';
    } else {
        rating = 'Stomp / Hủy Diệt'; icon = '🔴'; color = 'text-rose-600';
    }

    preview.classList.remove('hidden');
    preview.innerHTML = `
        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-white border border-slate-200 shadow-2xs">
            <span>${icon}</span>
            <span class="font-bold ${color}">${rating}</span>
            <span class="text-slate-400">•</span>
            <span>Chênh lệch: <b class="${color}">${diff} mạng</b></span>
            <span class="text-slate-400">•</span>
            <span>Độ kịch tính: <b>${(closeness * 100).toFixed(0)}%</b></span>
        </span>
    `;
}

async function confirmSaveStandardMatch() {
    if (isSubmittingMatch) return;
    isSubmittingMatch = true;

    const btn = document.getElementById('btn-confirm-save-standard');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Đang lưu...';
    }

    try {
        const team1Pids = simTeam1 && simTeam1.length === 5 ? simTeam1 : (currentTeamsResult?.team1_names || []);
        const team2Pids = simTeam2 && simTeam2.length === 5 ? simTeam2 : (currentTeamsResult?.team2_names || []);

        if (team1Pids.length !== 5 || team2Pids.length !== 5) {
            throw new Error("Không tìm thấy đủ 10 tuyển thủ của 2 đội hình");
        }

        const res = await fetch('/api/update_match_result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                team1: team1Pids,
                team2: team2Pids,
                winner: currentMatchModalWinner,
                team1_kills: parseInt(document.getElementById('input-team1-kills')?.value) || 0,
                team2_kills: parseInt(document.getElementById('input-team2-kills')?.value) || 0,
                notes: 'Ghi nhận kết quả trực tiếp'
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
            text: 'Điểm Thực Lực (RAPM) và Phân Bậc Tier đã được cập nhật tự động!',
            timer: 2000,
            showConfirmButton: false,
            ...SWAL_THEME
        });

        await loadAllPlayers();
        if (typeof loadAdminMatches === 'function') {
            await loadAdminMatches();
        }
    } catch (err) {
        console.error("Lỗi lưu trận đấu:", err);
        Swal.fire({
            icon: 'error',
            title: 'Lỗi',
            text: err.message || 'Không thể lưu kết quả trận đấu',
            ...SWAL_THEME
        });
    } finally {
        isSubmittingMatch = false;
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-check mr-1.5"></i><span>Xác Nhận & Lưu Kết Quả Ngay</span>';
        }
    }
}
