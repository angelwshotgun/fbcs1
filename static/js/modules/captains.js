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

