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
        if (!file || !file.type.startsWith('image/')) {
            resolve(null);
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                let width = img.width;
                let height = img.height;
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
                resolve({ base64: compressedBase64, mimeType: 'image/jpeg' });
            };
            img.onerror = () => {
                resolve({ base64: e.target.result, mimeType: file.type || 'image/jpeg' });
            };
            img.src = e.target.result;
        };
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(file);
    });
}

