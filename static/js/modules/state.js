// ==========================================
// FBCS AI 3.0 - GLOBAL APPLICATION STATE
// ==========================================

var allPlayers = [];
var selectedMatchmaker = [];
var selectedCaptains = [];
var selectedCaptainMembers = [];
var currentTeamsResult = null;

// Trạng thái cho tab Mô Phỏng 5vs5 (Simulation)
var simTeam1 = [null, null, null, null, null];
var simTeam2 = [null, null, null, null, null];
var simUnmatchedSlotInfo = { team1: {}, team2: {} };
var pendingSlotAssignment = null;
var lastOcrResult = null;

// Trạng thái cho Ghi Nhận Kết Quả & AI Scoreboard Modal
var isSubmittingSimulationMatch = false;
var currentMatchModalWinner = 'team1';
var currentScoreboardImageBase64 = null;
var currentScoreboardMimeType = 'image/jpeg';
var currentAiScoreboardAnalysis = null;

// Trạng thái cho Admin
var allAdminMatches = [];
var currentAdminSubTab = 'players';

var SWAL_THEME = {
    background: '#ffffff',
    color: '#0f172a',
    confirmButtonColor: '#4f46e5',
    cancelButtonColor: '#94a3b8'
};

// Helper nén và tối ưu hóa ảnh trước khi gửi AI Vision (giúp gửi nhanh hơn 20x và tránh timeout)
function compressImage(file, maxWidth = 1920, maxHeight = 1080, quality = 0.85) {
    return new Promise((resolve) => {
        if (!file) {
            resolve(null);
            return;
        }

        let isDone = false;
        const timeoutHandle = setTimeout(() => {
            if (!isDone) {
                isDone = true;
                resolve(null);
            }
        }, 6000);

        const finish = (result) => {
            if (!isDone) {
                isDone = true;
                clearTimeout(timeoutHandle);
                resolve(result);
            }
        };

        const reader = new FileReader();
        reader.onload = (e) => {
            const rawBase64 = e.target.result;
            const originalMime = file.type || 'image/jpeg';
            try {
                const img = new Image();
                img.onload = () => {
                    try {
                        let width = img.width || 1280;
                        let height = img.height || 720;
                        if (width > maxWidth || height > maxHeight) {
                            if (width / height > maxWidth / maxHeight) {
                                height = Math.round((height * maxWidth) / width);
                                width = maxWidth;
                            } else {
                                width = Math.round((width * maxHeight) / height);
                                height = maxHeight;
                            }
                        }
                        const canvas = document.createElement('canvas');
                        canvas.width = width;
                        canvas.height = height;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, width, height);
                        const compressedBase64 = canvas.toDataURL('image/jpeg', quality);
                        finish({ base64: compressedBase64, mimeType: 'image/jpeg' });
                    } catch (canvasErr) {
                        finish({ base64: rawBase64, mimeType: originalMime });
                    }
                };
                img.onerror = () => {
                    finish({ base64: rawBase64, mimeType: originalMime });
                };
                img.src = rawBase64;
            } catch (err) {
                finish({ base64: rawBase64, mimeType: originalMime });
            }
        };
        reader.onerror = () => finish(null);
        reader.readAsDataURL(file);
    });
}

