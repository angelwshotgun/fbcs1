// ==========================================
// FBCS AI 3.0 - MAIN APPLICATION ENTRYPOINT
// ==========================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadAllPlayers();
    await loadStatus();

    // Kiểm tra nếu người dùng truy cập trực tiếp URL trang chi tiết tuyển thủ
    let targetPlayerId = window.INITIAL_PLAYER_ID || '';

    if (!targetPlayerId && window.location.pathname.startsWith('/player/')) {
        targetPlayerId = decodeURIComponent(window.location.pathname.replace('/player/', '').split('/')[0]).trim();
    } else if (!targetPlayerId && window.location.hash.startsWith('#player/')) {
        targetPlayerId = decodeURIComponent(window.location.hash.replace('#player/', '').split('/')[0]).trim();
    }

    if (targetPlayerId) {
        openPlayerDetail(targetPlayerId, false);
    }
});

// Hỗ trợ nút Back/Forward của trình duyệt
window.addEventListener('popstate', (event) => {
    if (event.state && event.state.playerId) {
        openPlayerDetail(event.state.playerId, false);
    } else if (window.location.pathname.startsWith('/player/')) {
        const pid = decodeURIComponent(window.location.pathname.replace('/player/', '').split('/')[0]).trim();
        if (pid) openPlayerDetail(pid, false);
    } else {
        closePlayerDetail();
    }
});

