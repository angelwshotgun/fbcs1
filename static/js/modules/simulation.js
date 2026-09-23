// TAB: SIMULATION 5VS5 (MÔ PHỎNG & TRẬN ĐẤU TÙY CHỈNH)
// ==========================================
function renderSimulationBoard() {
    const t1Container = document.getElementById('sim-t1-slots');
    const t2Container = document.getElementById('sim-t2-slots');
    if (!t1Container || !t2Container) return;

    t1Container.innerHTML = '';
    t2Container.innerHTML = '';

    for (let i = 0; i < 5; i++) {
        t1Container.appendChild(createSimulationSlotCard('team1', i));
        t2Container.appendChild(createSimulationSlotCard('team2', i));
    }

    updateSimulationLiveStats();
}

let activeSlotDropdown = null;

function createSimulationSlotCard(team, slotIdx) {
    const isTeam1 = team === 'team1';
    const currentPid = isTeam1 ? simTeam1[slotIdx] : simTeam2[slotIdx];
    const playerObj = allPlayers.find(p => p.id === currentPid);
    const unmatchedRawName = simUnmatchedSlotInfo[team]?.[slotIdx] || null;

    const div = document.createElement('div');
    div.className = `relative flex items-center gap-2.5 p-2.5 rounded-2xl bg-white border ${
        isTeam1 ? 'border-blue-200/90 hover:border-blue-300' : 'border-rose-200/90 hover:border-rose-300'
    } shadow-xs transition`;

    // Slot badge
    const badgeColor = isTeam1 ? 'bg-blue-100 text-blue-800' : 'bg-rose-100 text-rose-800';
    const avatarSrc = playerObj ? playerObj.avatar : (unmatchedRawName ? `https://api.dicebear.com/7.x/bottts/svg?seed=${encodeURIComponent(unmatchedRawName)}` : 'https://api.dicebear.com/7.x/bottts/svg?seed=empty');

    const escapedRawName = unmatchedRawName ? unmatchedRawName.replace(/'/g, "\\'") : '';

    // Label on combobox trigger
    let displayLabel = `-- Chọn tuyển thủ (Slot ${slotIdx + 1}) --`;
    let labelStyle = 'text-slate-400 font-normal';
    if (playerObj) {
        displayLabel = `${playerObj.nickname} (Elo ${Math.round(playerObj.hidden_elo)})`;
        labelStyle = 'text-slate-900 font-bold';
    } else if (unmatchedRawName) {
        displayLabel = `⚠️ ${unmatchedRawName} (Chưa có trong data)`;
        labelStyle = 'text-amber-800 font-bold';
    }

    div.innerHTML = `
        <span class="w-7 h-7 rounded-xl ${badgeColor} text-xs font-black flex items-center justify-center shadow-2xs flex-shrink-0">${slotIdx + 1}</span>
        <div class="relative flex-shrink-0">
            <img src="${avatarSrc}" class="w-8 h-8 rounded-xl object-cover bg-slate-100 border border-slate-200" alt="Avatar">
            ${playerObj?.form?.icon ? `<span class="absolute -bottom-1 -right-1 text-[9px]">${playerObj.form.icon}</span>` : ''}
        </div>

        <!-- Searchable Combobox Container -->
        <div class="relative flex-1 min-w-0" id="sim-combobox-${team}-${slotIdx}">
            <button type="button" onclick="toggleSlotDropdown('${team}', ${slotIdx}, event)" class="w-full bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-xl px-2.5 py-1.5 text-xs flex items-center justify-between gap-1.5 transition text-left focus:outline-none focus:border-indigo-500">
                <span class="truncate ${labelStyle}">${displayLabel}</span>
                <i class="fa-solid fa-chevron-down text-slate-400 text-[10px] flex-shrink-0"></i>
            </button>

            <!-- Dropdown Menu Panel -->
            <div id="sim-menu-${team}-${slotIdx}" class="hidden absolute left-0 top-full mt-1.5 w-full min-w-[280px] max-w-[340px] bg-white border border-slate-200 rounded-2xl shadow-xl z-50 p-2.5 space-y-2">
                <!-- Search Input with icon -->
                <div class="relative">
                    <i class="fa-solid fa-search absolute left-3 top-2.5 text-slate-400 text-[11px]"></i>
                    <input type="text" id="sim-search-${team}-${slotIdx}" oninput="filterSlotDropdown('${team}', ${slotIdx}, this.value)" placeholder="Gõ tên tìm tuyển thủ..." class="w-full bg-slate-50 border border-slate-200 rounded-xl pl-8 pr-2.5 py-1.5 text-xs text-slate-800 focus:outline-none focus:border-indigo-500 focus:bg-white transition" onclick="event.stopPropagation()">
                </div>

                <!-- Scrollable Items List -->
                <div id="sim-list-${team}-${slotIdx}" class="max-h-52 overflow-y-auto custom-scrollbar space-y-1 text-xs">
                    <!-- Populated dynamically via JS -->
                </div>
            </div>
        </div>

        <!-- Slot Actions -->
        <div class="flex items-center gap-1 flex-shrink-0">
            ${unmatchedRawName && !playerObj ? `
                <button type="button" onclick="quickCreatePlayerFromSlot('${team}', ${slotIdx}, '${escapedRawName}')" class="px-2 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-[11px] font-bold shadow-xs transition flex items-center gap-1">
                    <i class="fa-solid fa-user-plus text-[10px]"></i>
                    <span>+ Tạo</span>
                </button>
            ` : `
                <button type="button" onclick="quickCreatePlayerFromSlot('${team}', ${slotIdx})" title="Thêm tuyển thủ mới" class="p-1.5 text-slate-400 hover:text-indigo-600 rounded-lg hover:bg-slate-100 transition">
                    <i class="fa-solid fa-user-plus text-xs"></i>
                </button>
            `}
            <button type="button" onclick="clearSimulationSlot('${team}', ${slotIdx})" title="Xóa slot" class="p-1.5 text-slate-400 hover:text-rose-600 rounded-lg hover:bg-slate-100 transition">
                <i class="fa-solid fa-xmark text-xs"></i>
            </button>
        </div>
    `;

    return div;
}

function toggleSlotDropdown(team, slotIdx, event) {
    if (event) event.stopPropagation();

    if (activeSlotDropdown && activeSlotDropdown.team === team && activeSlotDropdown.slotIdx === slotIdx) {
        closeAllSlotDropdowns();
        return;
    }

    closeAllSlotDropdowns();

    const menu = document.getElementById(`sim-menu-${team}-${slotIdx}`);
    if (!menu) return;

    menu.classList.remove('hidden');
    activeSlotDropdown = { team, slotIdx };

    populateSlotDropdownList(team, slotIdx, '');

    const searchInput = document.getElementById(`sim-search-${team}-${slotIdx}`);
    if (searchInput) {
        searchInput.value = '';
        setTimeout(() => searchInput.focus(), 50);
    }
}

function closeAllSlotDropdowns() {
    document.querySelectorAll('[id^="sim-menu-"]').forEach(m => m.classList.add('hidden'));
    activeSlotDropdown = null;
}

function filterSlotDropdown(team, slotIdx, query) {
    populateSlotDropdownList(team, slotIdx, query);
}

function populateSlotDropdownList(team, slotIdx, query = '') {
    const listContainer = document.getElementById(`sim-list-${team}-${slotIdx}`);
    if (!listContainer) return;

    listContainer.innerHTML = '';
    const q = (query || '').trim().toLowerCase();
    const currentPid = team === 'team1' ? simTeam1[slotIdx] : simTeam2[slotIdx];
    const unmatchedRawName = simUnmatchedSlotInfo[team]?.[slotIdx] || null;

    // Option 1: Clear/Deselect Slot
    if (!q || '--'.includes(q) || 'bỏ chọn'.includes(q) || 'xóa'.includes(q)) {
        const clearItem = document.createElement('div');
        clearItem.className = 'flex items-center gap-2 p-1.5 rounded-xl text-slate-500 hover:bg-slate-100 cursor-pointer transition select-none';
        clearItem.innerHTML = `
            <i class="fa-solid fa-ban text-slate-400 w-5 text-center text-xs"></i>
            <span class="font-medium text-xs">-- Để trống slot này --</span>
        `;
        clearItem.onclick = (e) => {
            e.stopPropagation();
            onSimulationSlotChange(team, slotIdx, '');
            closeAllSlotDropdowns();
        };
        listContainer.appendChild(clearItem);
    }

    // Option 2: Unmatched raw name from OCR (if any)
    if (unmatchedRawName && (!q || unmatchedRawName.toLowerCase().includes(q))) {
        const unmatchedItem = document.createElement('div');
        unmatchedItem.className = 'flex items-center justify-between p-2 rounded-xl bg-amber-50/80 border border-amber-200/80 hover:bg-amber-100/80 cursor-pointer transition select-none';
        const escapedName = unmatchedRawName.replace(/'/g, "\\'");
        unmatchedItem.innerHTML = `
            <div class="flex items-center gap-2 truncate">
                <span class="text-amber-600 font-bold">⚠️</span>
                <span class="font-bold text-amber-900 truncate text-xs">${unmatchedRawName}</span>
            </div>
            <button type="button" onclick="event.stopPropagation(); quickCreatePlayerFromSlot('${team}', ${slotIdx}, '${escapedName}'); closeAllSlotDropdowns()" class="px-2 py-0.5 rounded-lg bg-amber-600 text-white font-bold text-[10px] shadow-2xs flex-shrink-0">
                + Tạo Mới
            </button>
        `;
        unmatchedItem.onclick = (e) => {
            e.stopPropagation();
            quickCreatePlayerFromSlot(team, slotIdx, unmatchedRawName);
            closeAllSlotDropdowns();
        };
        listContainer.appendChild(unmatchedItem);
    }

    // Option 3: Filtered Players List
    const sorted = [...allPlayers].sort((a, b) => a.nickname.localeCompare(b.nickname));
    const filtered = sorted.filter(p => {
        if (!q) return true;
        return p.nickname.toLowerCase().includes(q) || p.id.toLowerCase().includes(q);
    });

    filtered.forEach(p => {
        const isSelected = p.id === currentPid;
        const item = document.createElement('div');
        item.className = `flex items-center justify-between p-1.5 rounded-xl cursor-pointer transition select-none ${
            isSelected ? 'bg-indigo-50 text-indigo-900 font-bold border border-indigo-200' : 'hover:bg-slate-100 text-slate-700'
        }`;

        item.innerHTML = `
            <div class="flex items-center gap-2 min-w-0">
                <img src="${p.avatar}" class="w-6 h-6 rounded-lg object-cover bg-slate-100 border border-slate-200 flex-shrink-0">
                <span class="truncate font-semibold text-xs text-slate-900">${p.nickname}</span>
                <span class="text-[10px] text-slate-400">${p.form?.icon || ''}</span>
            </div>
            <div class="flex items-center gap-1.5 flex-shrink-0 ml-2">
                <span class="text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                    Elo ${Math.round(p.hidden_elo)}
                </span>
                ${isSelected ? '<i class="fa-solid fa-check text-indigo-600 text-xs"></i>' : ''}
            </div>
        `;

        item.onclick = (e) => {
            e.stopPropagation();
            onSimulationSlotChange(team, slotIdx, p.id);
            closeAllSlotDropdowns();
        };
        listContainer.appendChild(item);
    });

    // If query typed and no players found
    if (filtered.length === 0 && (!unmatchedRawName || !unmatchedRawName.toLowerCase().includes(q))) {
        const emptyNotice = document.createElement('div');
        emptyNotice.className = 'p-3 text-center text-slate-400 text-xs space-y-2';
        const escapedQ = query.replace(/'/g, "\\'");
        emptyNotice.innerHTML = `
            <p>Không tìm thấy tuyển thủ <b>"${query}"</b></p>
            <button type="button" onclick="event.stopPropagation(); quickCreatePlayerFromSlot('${team}', ${slotIdx}, '${escapedQ}'); closeAllSlotDropdowns()" class="w-full px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs shadow-xs transition flex items-center justify-center gap-1.5">
                <i class="fa-solid fa-user-plus text-[11px]"></i>
                <span>+ Thêm mới "${query}"</span>
            </button>
        `;
        listContainer.appendChild(emptyNotice);
    }
}

// Global click & escape listeners to close dropdowns
document.addEventListener('click', (e) => {
    if (!e.target.closest('[id^="sim-combobox-"]')) {
        closeAllSlotDropdowns();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeAllSlotDropdowns();
    }
});

function onSimulationSlotChange(team, slotIdx, selectedVal) {
    if (selectedVal && selectedVal.startsWith('__unmatched__:')) {
        const rawName = selectedVal.replace('__unmatched__:', '');
        quickCreatePlayerFromSlot(team, slotIdx, rawName);
        return;
    }

    if (!selectedVal) {
        if (team === 'team1') simTeam1[slotIdx] = null;
        else simTeam2[slotIdx] = null;
        if (simUnmatchedSlotInfo[team]) delete simUnmatchedSlotInfo[team][slotIdx];
        renderSimulationBoard();
        return;
    }

    // Check duplicate
    const currentTeam = team === 'team1' ? simTeam1 : simTeam2;
    const otherTeam = team === 'team1' ? simTeam2 : simTeam1;

    const inOtherSlotCurrent = currentTeam.findIndex((pid, idx) => idx !== slotIdx && pid === selectedVal);
    const inOtherSlotOther = otherTeam.findIndex(pid => pid === selectedVal);

    if (inOtherSlotCurrent > -1) {
        currentTeam[inOtherSlotCurrent] = null;
    }
    if (inOtherSlotOther > -1) {
        otherTeam[inOtherSlotOther] = null;
    }

    if (team === 'team1') simTeam1[slotIdx] = selectedVal;
    else simTeam2[slotIdx] = selectedVal;

    if (simUnmatchedSlotInfo[team]) delete simUnmatchedSlotInfo[team][slotIdx];

    renderSimulationBoard();
}

function clearSimulationSlot(team, slotIdx) {
    if (team === 'team1') simTeam1[slotIdx] = null;
    else simTeam2[slotIdx] = null;
    if (simUnmatchedSlotInfo[team]) delete simUnmatchedSlotInfo[team][slotIdx];
    renderSimulationBoard();
}

function clearSimulationSlots() {
    simTeam1 = [null, null, null, null, null];
    simTeam2 = [null, null, null, null, null];
    simUnmatchedSlotInfo = { team1: {}, team2: {} };
    renderSimulationBoard();
}

function quickCreatePlayerFromSlot(team, slotIdx, defaultNickname = '') {
    pendingSlotAssignment = { team, slotIdx };
    openPlayerModal('add');

    if (defaultNickname) {
        document.getElementById('form-nickname').value = defaultNickname;
        const cleanId = defaultNickname.toLowerCase().replace(/[^a-z0-9]/g, '') || `player_${Math.floor(Math.random() * 10000)}`;
        document.getElementById('form-id').value = cleanId;
        document.getElementById('form-avatar-preview').src = `https://api.dicebear.com/7.x/bottts/svg?seed=${cleanId}`;
    }
}

function updateSimulationLiveStats() {
    const t1Players = simTeam1.map(id => allPlayers.find(p => p.id === id)).filter(Boolean);
    const t2Players = simTeam2.map(id => allPlayers.find(p => p.id === id)).filter(Boolean);

    const t1TotalElo = Math.round(t1Players.reduce((sum, p) => sum + (p.hidden_elo || 1200), 0));
    const t2TotalElo = Math.round(t2Players.reduce((sum, p) => sum + (p.hidden_elo || 1200), 0));

    const t1AvgElo = t1Players.length ? Math.round(t1TotalElo / t1Players.length) : 0;
    const t2AvgElo = t2Players.length ? Math.round(t2TotalElo / t2Players.length) : 0;

    const diffElo = Math.abs(t1TotalElo - t2TotalElo);

    let prob1 = 50;
    let prob2 = 50;
    if (t1Players.length > 0 || t2Players.length > 0) {
        const d = (t1TotalElo - t2TotalElo) / (Math.max(t1Players.length, t2Players.length) || 5);
        prob1 = Math.round((1.0 / (1.0 + Math.pow(10.0, -d / 400.0))) * 100);
        prob1 = Math.max(5, Math.min(95, prob1));
        prob2 = 100 - prob1;
    }

    const t1TotalElem = document.getElementById('sim-t1-total-elo');
    if (t1TotalElem) t1TotalElem.innerText = t1TotalElo;
    const t1AvgElem = document.getElementById('sim-t1-avg-elo');
    if (t1AvgElem) t1AvgElem.innerText = t1AvgElo;

    const t2TotalElem = document.getElementById('sim-t2-total-elo');
    if (t2TotalElem) t2TotalElem.innerText = t2TotalElo;
    const t2AvgElem = document.getElementById('sim-t2-avg-elo');
    if (t2AvgElem) t2AvgElem.innerText = t2AvgElo;

    const diffElem = document.getElementById('sim-diff-elo');
    if (diffElem) diffElem.innerText = diffElo;
    const probText = document.getElementById('sim-prob-text');
    if (probText) probText.innerText = `${prob1}% - ${prob2}%`;

    const barT1 = document.getElementById('sim-bar-t1');
    const barT2 = document.getElementById('sim-bar-t2');
    if (barT1 && barT2) {
        barT1.style.width = `${prob1}%`;
        barT2.style.width = `${prob2}%`;
    }

    const t1CountBadge = document.getElementById('sim-t1-count-badge');
    if (t1CountBadge) t1CountBadge.innerText = `${t1Players.length}/5`;
    const t2CountBadge = document.getElementById('sim-t2-count-badge');
    if (t2CountBadge) t2CountBadge.innerText = `${t2Players.length}/5`;
}

function transferSimulationToMatchmaker() {
    const chosenIds = [...simTeam1, ...simTeam2].filter(Boolean);
    if (chosenIds.length === 0) {
        Swal.fire({
            icon: 'info',
            title: 'Chưa chọn tuyển thủ',
            text: 'Vui lòng chọn tuyển thủ vào các slot trước khi chuyển sang Chia Đội.',
            ...SWAL_THEME
        });
        return;
    }

    selectedMatchmaker = chosenIds.slice(0, 10);
    switchTab('matchmaker');
    renderMatchmakerPlayers();
    Swal.fire({
        icon: 'success',
        title: `Đã chọn ${selectedMatchmaker.length}/10 tuyển thủ`,
        text: 'Bạn có thể bấm Chia Đội Tối Ưu để thuật toán AI cân bằng lại 2 đội.',
        timer: 1500,
        showConfirmButton: false,
        ...SWAL_THEME
    });
}

function transferOcrToSimulation() {
    switchTab('simulation');
    renderSimulationBoard();
}

function submitSimulationWinner(winningTeam) {
    const t1Filled = simTeam1.filter(Boolean);
    const t2Filled = simTeam2.filter(Boolean);

    if (t1Filled.length !== 5 || t2Filled.length !== 5) {
        Swal.fire({
            icon: 'warning',
            title: 'Chưa đủ 10 người',
            text: `Vui lòng chọn đủ 5 người cho Đội Xanh (hiện có ${t1Filled.length}/5) và 5 người cho Đội Đỏ (hiện có ${t2Filled.length}/5).`,
            ...SWAL_THEME
        });
        return;
    }

    const allChosen = [...t1Filled, ...t2Filled];
    const uniqueChosen = new Set(allChosen);
    if (uniqueChosen.size !== 10) {
        Swal.fire({
            icon: 'error',
            title: 'Trùng lặp người chơi',
            text: 'Có tuyển thủ bị chọn nhiều hơn 1 lần giữa 2 đội. Vui lòng kiểm tra lại các slot.',
            ...SWAL_THEME
        });
        return;
    }

    openMatchResultModal(winningTeam);
}



// SCREENSHOT OCR (QUÉT ẢNH PHÒNG ĐẤU AI)
// ==========================================
window.addEventListener('paste', handleGlobalScreenshotPaste);

function handleGlobalScreenshotPaste(e) {
    const items = (e.clipboardData || e.originalEvent?.clipboardData)?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === 'file' && item.type.startsWith('image/')) {
            const blob = item.getAsFile();
            if (!blob) continue;

            // Nếu modal kết quả trận đấu đang mở -> nạp vào bảng điểm modal
            const matchModal = document.getElementById('match-result-modal');
            if (matchModal && !matchModal.classList.contains('hidden')) {
                switchMatchResultTab('ai_scoreboard');
                handleScoreboardPastedFile(blob);
                return;
            }

            // Ngược lại -> nạp vào OCR sảnh chờ thông thường
            processScreenshotFile(blob);
            break;
        }
    }
}

const dropzone = document.getElementById('screenshot-dropzone');
if (dropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('border-indigo-500', 'bg-indigo-50/80');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('border-indigo-500', 'bg-indigo-50/80');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer?.files;
        if (files && files.length > 0 && files[0].type.startsWith('image/')) {
            processScreenshotFile(files[0]);
        }
    });
}

function handleScreenshotFileSelect(e) {
    const files = e.target?.files;
    if (files && files.length > 0) {
        processScreenshotFile(files[0]);
    }
}

async function processScreenshotFile(file) {
    const spinner = document.getElementById('ocr-loading-spinner');
    if (spinner) spinner.classList.remove('hidden');

    const compressed = await compressImage(file);
    const base64Data = compressed ? compressed.base64 : null;
    const mimeType = compressed ? compressed.mimeType : (file.type || 'image/jpeg');

    if (!base64Data) {
        if (spinner) spinner.classList.add('hidden');
        return;
    }

    let apiKey = localStorage.getItem('fbcs_gemini_key') || localStorage.getItem('fbcs_gemini_api_key') || '';

    try {
        let res = await fetch('/api/ocr_screenshot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: base64Data,
                mime_type: mimeType,
                api_key: apiKey
            })
        });

            let data = await res.json();

            // Nếu chưa có API Key, cho phép người dùng nhập ngay tại chỗ
            if (!data.success && data.error && data.error.includes('Chưa cấu hình Gemini API Key')) {
                if (spinner) spinner.classList.add('hidden');
                
                const { value: inputKey } = await Swal.fire({
                    title: '🔑 Nhập Gemini API Key',
                    text: 'Tính năng AI Vision cần Gemini API Key để nhận diện ảnh chụp sảnh chờ phòng đấu.',
                    input: 'password',
                    inputPlaceholder: 'Dán Gemini API Key của bạn vào đây...',
                    showCancelButton: true,
                    confirmButtonText: 'Lưu & Quét Lại',
                    cancelButtonText: 'Hủy',
                    ...SWAL_THEME,
                    inputValidator: (val) => {
                        if (!val || !val.trim()) return 'Vui lòng không để trống API Key!';
                    }
                });

                if (inputKey) {
                    apiKey = inputKey.trim();
                    localStorage.setItem('fbcs_gemini_key', apiKey);
                    const keyInputElem = document.getElementById('input-gemini-key');
                    if (keyInputElem) keyInputElem.value = apiKey;
                    
                    if (spinner) spinner.classList.remove('hidden');
                    res = await fetch('/api/ocr_screenshot', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image: base64Data,
                            mime_type: file.type || 'image/jpeg',
                            api_key: apiKey
                        })
                    });
                    data = await res.json();
                } else {
                    return;
                }
            }

            if (spinner) spinner.classList.add('hidden');

            if (data.success) {
                lastOcrResult = data;

                // 1. Phân bổ vào các slot Mô Phỏng 5vs5
                simUnmatchedSlotInfo = { team1: {}, team2: {} };

                if (data.team1_slots && data.team1_slots.length > 0) {
                    data.team1_slots.forEach((slot, idx) => {
                        if (idx < 5) {
                            if (slot.matched_id) {
                                simTeam1[idx] = slot.matched_id;
                            } else if (slot.raw_name) {
                                simTeam1[idx] = null;
                                simUnmatchedSlotInfo.team1[idx] = slot.raw_name;
                            }
                        }
                    });
                }
                if (data.team2_slots && data.team2_slots.length > 0) {
                    data.team2_slots.forEach((slot, idx) => {
                        if (idx < 5) {
                            if (slot.matched_id) {
                                simTeam2[idx] = slot.matched_id;
                            } else if (slot.raw_name) {
                                simTeam2[idx] = null;
                                simUnmatchedSlotInfo.team2[idx] = slot.raw_name;
                            }
                        }
                    });
                }

                // 2. Điền vào matchmaker nếu có người khớp
                if (data.matched_player_ids && data.matched_player_ids.length > 0) {
                    selectedMatchmaker = data.matched_player_ids.slice(0, 10);
                    renderMatchmakerPlayers();
                }

                renderSimulationBoard();

                const matchedNicknames = (data.matched_player_ids || []).map(pid => {
                    const found = allPlayers.find(x => x.id === pid);
                    return found ? found.nickname : pid;
                });

                const allDetected = data.detected_names || [];
                const unmatchedCount = allDetected.length - (data.matched_player_ids || []).length;

                const summaryBox = document.getElementById('ocr-result-summary');
                const namesElem = document.getElementById('ocr-detected-names');
                if (summaryBox && namesElem) {
                    summaryBox.classList.remove('hidden');
                    namesElem.innerText = `${matchedNicknames.length} tuyển thủ: ${matchedNicknames.join(', ')}` + 
                        (unmatchedCount > 0 ? ` (kèm ${unmatchedCount} tên mới)` : '');
                }

                Swal.fire({
                    icon: 'success',
                    title: `🎯 Nhận diện thành công ${allDetected.length || data.count || 0} tuyển thủ!`,
                    html: `
                        <div class="text-left text-xs text-slate-700 mt-2 space-y-2">
                            ${matchedNicknames.length ? `<p><b>Đã khớp dữ liệu (${matchedNicknames.length}):</b> ${matchedNicknames.join(', ')}</p>` : ''}
                            ${allDetected.length ? `<p class="text-[11px] text-slate-500"><b>Tất cả tên đọc được từ ảnh:</b> ${allDetected.join(', ')}</p>` : ''}
                            ${unmatchedCount > 0 ? `<p class="text-[11px] text-amber-700 bg-amber-50 p-2.5 rounded-xl border border-amber-200">💡 <b>Có ${unmatchedCount} tên chưa có trong danh sách</b>. Bạn có thể mở Giao Diện 5vs5 để bấm <b>+ Tạo</b> tuyển thủ mới ngay lập tức!</p>` : ''}
                        </div>
                    `,
                    confirmButtonText: '🎮 Mở Giao Diện 5vs5',
                    showCancelButton: true,
                    cancelButtonText: selectedMatchmaker.length === 10 ? '⚡ Chia Đội Ngay' : 'Đóng',
                    confirmButtonColor: '#7c3aed',
                    cancelButtonColor: '#4f46e5',
                    ...SWAL_THEME
                }).then((result) => {
                    if (result.isConfirmed) {
                        switchTab('simulation');
                    } else if (result.dismiss === Swal.DismissReason.cancel && selectedMatchmaker.length === 10) {
                        handleCreateTeams();
                    }
                });

            } else {
                Swal.fire({
                    icon: 'warning',
                    title: 'Chưa tìm thấy tuyển thủ phù hợp',
                    text: data.error || 'AI không nhận diện được tên tuyển thủ nào trong bức ảnh này. Vui lòng thử ảnh rõ nét hơn.',
                    ...SWAL_THEME
                });
            }

        }
    } catch (err) {
        if (spinner) spinner.classList.add('hidden');
        console.error("Lỗi OCR:", err);
        Swal.fire({
            icon: 'error',
            title: 'Lỗi',
            text: 'Không thể kết nối đến dịch vụ phân tích ảnh.',
            ...SWAL_THEME
        });
    }
}


