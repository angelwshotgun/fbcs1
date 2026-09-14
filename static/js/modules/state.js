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
