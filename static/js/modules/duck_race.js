/**
 * Duck Race Minigame Engine
 * 100% Client-side HTML5 Canvas Animation & Web Audio API
 * Tính năng độc lập: Quản lý danh sách tuyển thủ (Thêm / Xoá),
 * Bốc thăm Ban/Pick, và Loại người thắng cuộc để đua tiếp vòng sau!
 */

// Sound Manager using Web Audio API
class DuckAudioFx {
    constructor() {
        this.ctx = null;
        this.enabled = true;
    }

    init() {
        if (!this.ctx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (AudioContext) {
                this.ctx = new AudioContext();
            }
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    playTone(freq, type = 'sine', duration = 0.15, gainVal = 0.15) {
        if (!this.enabled) return;
        try {
            this.init();
            if (!this.ctx) return;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
            gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(this.ctx.currentTime + duration);
        } catch (e) {}
    }

    playCountdown(final = false) {
        if (!final) {
            this.playTone(440, 'triangle', 0.15, 0.2);
        } else {
            this.playTone(880, 'square', 0.35, 0.25);
        }
    }

    playQuack() {
        if (!this.enabled) return;
        try {
            this.init();
            if (!this.ctx) return;
            const t = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(320, t);
            osc.frequency.linearRampToValueAtTime(180, t + 0.18);
            gain.gain.setValueAtTime(0.12, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(t);
            osc.stop(t + 0.2);
        } catch (e) {}
    }

    playBoost() {
        if (!this.enabled) return;
        try {
            this.init();
            if (!this.ctx) return;
            const t = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(200, t);
            osc.frequency.exponentialRampToValueAtTime(900, t + 0.4);
            gain.gain.setValueAtTime(0.18, t);
            gain.gain.exponentialRampToValueAtTime(0.01, t + 0.45);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(t);
            osc.stop(t + 0.45);
        } catch (e) {}
    }

    playLightning() {
        if (!this.enabled) return;
        try {
            this.init();
            if (!this.ctx) return;
            const t = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(80, t);
            osc.frequency.linearRampToValueAtTime(40, t + 0.3);
            gain.gain.setValueAtTime(0.3, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(t);
            osc.stop(t + 0.35);
        } catch (e) {}
    }

    playWhirlpool() {
        if (!this.enabled) return;
        try {
            this.init();
            if (!this.ctx) return;
            const t = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(250, t);
            osc.frequency.linearRampToValueAtTime(100, t + 0.5);
            gain.gain.setValueAtTime(0.2, t);
            gain.gain.exponentialRampToValueAtTime(0.01, t + 0.5);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(t);
            osc.stop(t + 0.5);
        } catch (e) {}
    }

    playFinishFanfare() {
        if (!this.enabled) return;
        const notes = [523.25, 659.25, 783.99, 1046.50];
        notes.forEach((freq, idx) => {
            setTimeout(() => {
                this.playTone(freq, 'triangle', 0.28, 0.22);
            }, idx * 120);
        });
    }

    playBump() {
        if (!this.enabled) return;
        try {
            this.init();
            if (!this.ctx) return;
            const t = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(240, t);
            osc.frequency.exponentialRampToValueAtTime(480, t + 0.05);
            osc.frequency.exponentialRampToValueAtTime(140, t + 0.14);
            gain.gain.setValueAtTime(0.24, t);
            gain.gain.exponentialRampToValueAtTime(0.01, t + 0.15);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(t);
            osc.stop(t + 0.15);
        } catch (e) {}
    }

    playSplash() {
        if (!this.enabled) return;
        try {
            this.init();
            if (!this.ctx) return;
            const t = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(320, t);
            osc.frequency.linearRampToValueAtTime(180, t + 0.18);
            gain.gain.setValueAtTime(0.22, t);
            gain.gain.exponentialRampToValueAtTime(0.01, t + 0.2);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(t);
            osc.stop(t + 0.2);
        } catch (e) {}
    }

    playThunder() {
        if (!this.enabled) return;
        try {
            this.init();
            if (!this.ctx) return;
            const t = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(70, t);
            osc.frequency.linearRampToValueAtTime(35, t + 0.65);
            gain.gain.setValueAtTime(0.3, t);
            gain.gain.exponentialRampToValueAtTime(0.01, t + 0.7);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(t);
            osc.stop(t + 0.7);
        } catch (e) {}
    }
}

const duckAudio = new DuckAudioFx();

function toggleDuckRaceSound() {
    duckAudio.enabled = !duckAudio.enabled;
    const icons = [
        document.getElementById('duck-sound-icon'),
        document.getElementById('tab-duck-sound-icon')
    ];
    icons.forEach(icon => {
        if (icon) {
            if (duckAudio.enabled) {
                icon.className = 'fa-solid fa-volume-high text-amber-500';
            } else {
                icon.className = 'fa-solid fa-volume-xmark text-slate-400';
            }
        }
    });
}

// Particle System for Water, Clashes, Weather and Celebration
class ParticleSystem {
    constructor() {
        this.particles = [];
    }

    addConfetti(x, y, count = 30) {
        const colors = ['#f59e0b', '#ef4444', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#ffffff'];
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x,
                y,
                vx: (Math.random() - 0.5) * 8,
                vy: (Math.random() - 0.9) * 10,
                size: Math.random() * 6 + 3,
                color: colors[Math.floor(Math.random() * colors.length)],
                life: 1.0,
                decay: Math.random() * 0.015 + 0.008,
                rotation: Math.random() * Math.PI * 2,
                vRot: (Math.random() - 0.5) * 0.2,
                type: 'confetti'
            });
        }
    }

    addWaterSplash(x, y, count = 6) {
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: x + (Math.random() - 0.5) * 12,
                y: y + (Math.random() - 0.5) * 8,
                vx: -Math.random() * 2 - 0.5,
                vy: (Math.random() - 0.5) * 1.5,
                size: Math.random() * 3 + 1,
                color: 'rgba(255, 255, 255, 0.6)',
                life: 1.0,
                decay: Math.random() * 0.04 + 0.03,
                type: 'water'
            });
        }
    }

    addFireTrail(x, y) {
        this.particles.push({
            x: x - 10,
            y: y + (Math.random() - 0.5) * 6,
            vx: -Math.random() * 3 - 2,
            vy: (Math.random() - 0.5) * 1,
            size: Math.random() * 5 + 3,
            color: Math.random() > 0.5 ? '#f97316' : '#eab308',
            life: 1.0,
            decay: 0.05,
            type: 'fire'
        });
    }

    addBumpSpark(x, y, count = 6) {
        const colors = ['#fde047', '#f59e0b', '#fb923c', '#ffffff'];
        for (let i = 0; i < count; i++) {
            const angle = Math.random() * Math.PI * 2;
            const spd = Math.random() * 3.5 + 1.5;
            this.particles.push({
                x,
                y,
                vx: Math.cos(angle) * spd,
                vy: Math.sin(angle) * spd,
                size: Math.random() * 4 + 2.5,
                color: colors[Math.floor(Math.random() * colors.length)],
                life: 1.0,
                decay: 0.065,
                type: 'spark'
            });
        }
    }

    addDraftingTrail(x, y) {
        this.particles.push({
            x,
            y: y + (Math.random() - 0.5) * 8,
            vx: -Math.random() * 3 - 3.5,
            vy: (Math.random() - 0.5) * 0.6,
            size: Math.random() * 8 + 4,
            color: 'rgba(56, 189, 248, 0.5)',
            life: 0.85,
            decay: 0.06,
            type: 'draft'
        });
    }

    addRainDrops(startDrawX, endDrawX, count = 6) {
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: startDrawX + Math.random() * (endDrawX - startDrawX + 100),
                y: -10,
                vx: 3 + Math.random() * 2,
                vy: 9 + Math.random() * 4,
                len: 12 + Math.random() * 8,
                color: 'rgba(186, 230, 253, 0.45)',
                life: 1.0,
                decay: 0.035,
                type: 'rain'
            });
        }
    }

    addWindLeaves(startDrawX, endDrawX, count = 2) {
        const colors = ['#ea580c', '#eab308', '#dc2626', '#ca8a04'];
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: startDrawX - 20,
                y: 35 + Math.random() * 280,
                vx: 4 + Math.random() * 3,
                vy: (Math.random() - 0.5) * 1.5,
                size: Math.random() * 5 + 3,
                color: colors[Math.floor(Math.random() * colors.length)],
                rotation: Math.random() * Math.PI * 2,
                vRot: (Math.random() - 0.5) * 0.15,
                life: 1.0,
                decay: 0.012,
                type: 'leaf'
            });
        }
    }

    update() {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.life -= p.decay;

            if (p.type === 'confetti') {
                p.vy += 0.18; // gravity
                p.rotation += p.vRot;
            } else if (p.type === 'leaf') {
                p.rotation += p.vRot;
            }

            if (p.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }

    draw(ctx) {
        ctx.save();
        for (const p of this.particles) {
            ctx.globalAlpha = Math.max(0, p.life);
            if (p.type === 'confetti') {
                ctx.save();
                ctx.translate(p.x, p.y);
                ctx.rotate(p.rotation);
                ctx.fillStyle = p.color;
                ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
                ctx.restore();
            } else if (p.type === 'spark') {
                ctx.save();
                ctx.translate(p.x, p.y);
                ctx.fillStyle = p.color;
                ctx.beginPath();
                // 4-pointed sparkle
                ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            } else if (p.type === 'draft') {
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.ellipse(p.x, p.y, p.size, p.size * 0.3, 0, 0, Math.PI * 2);
                ctx.fill();
            } else if (p.type === 'rain') {
                ctx.strokeStyle = p.color;
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(p.x - p.vx * 1.5, p.y - p.vy * 1.5);
                ctx.stroke();
            } else if (p.type === 'leaf') {
                ctx.save();
                ctx.translate(p.x, p.y);
                ctx.rotate(p.rotation);
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.ellipse(0, 0, p.size, p.size * 0.55, 0, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            } else {
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        ctx.restore();
    }
}

// Embedded Default Duck SVG Data URI (100% resilient fallback)
const DEFAULT_DUCK_DATA_URI = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128"><defs><linearGradient id="duckBody" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="%23fff176"/><stop offset="50%" stop-color="%23fdd835"/><stop offset="100%" stop-color="%23fbc02d"/></linearGradient><linearGradient id="duckWing" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="%23ffee58"/><stop offset="100%" stop-color="%23f9a825"/></linearGradient><linearGradient id="duckBeak" x1="0%" y1="0%" x2="100%" y2="50%"><stop offset="0%" stop-color="%23ff9800"/><stop offset="100%" stop-color="%23e65100"/></linearGradient></defs><path d="M 22 75 C 10 70 8 50 20 46 C 26 44 32 55 35 63 Z" fill="%23fbc02d" stroke="%23f57f17" stroke-width="2.5"/><ellipse cx="60" cy="80" rx="42" ry="32" fill="url(%23duckBody)" stroke="%23f57f17" stroke-width="2.5"/><path d="M 44 72 C 38 72 32 78 35 86 C 39 96 54 98 68 93 C 78 89 82 81 78 76 C 73 70 54 72 44 72 Z" fill="url(%23duckWing)" stroke="%23f57f17" stroke-width="2.5"/><path d="M 72 70 C 72 65 74 54 77 46 C 78 40 82 30 92 30 C 104 30 110 40 108 52 C 107 60 102 67 96 74 Z" fill="url(%23duckBody)" stroke="%23f57f17" stroke-width="2.5"/><path d="M 103 48 C 112 47 124 50 126 55 C 127 57 122 62 113 63 C 103 64 98 62 98 56 Z" fill="url(%23duckBeak)" stroke="%23bf360c" stroke-width="2"/><ellipse cx="94" cy="42" rx="6.5" ry="8" fill="%231e293b"/><ellipse cx="96" cy="39.5" rx="2.5" ry="3.5" fill="%23ffffff"/><circle cx="92.5" cy="45" r="1.2" fill="%23ffffff"/><ellipse cx="89" cy="54" rx="4.5" ry="3" fill="%23ff7043" opacity="0.4"/></svg>`;

// Main Duck Race Game Class
class DuckRaceGame {
    constructor(canvasId, allPlayers = []) {
        this.canvasId = canvasId;
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.allPlayers = allPlayers;
        this.teamMode = 'all'; // 'all' | 'team1' | 'team2'
        this.targetDuration = 30; // 15 | 30 | 45 | 60 seconds

        this.ducks = [];
        this.finishOrder = [];
        this.state = 'idle'; // 'idle' | 'countdown' | 'racing' | 'finished'
        this.animFrameId = null;
        this.particles = new ParticleSystem();
        this.waveOffset = 0;
        this.nextEventCheck = 0;
        this.activeToastTimer = null;
        this.lastTime = performance.now();

        // Image caches: avatar & custom duck sprites
        this.avatarImages = {}; // { playerId: HTMLImageElement }
        this.duckImages = {};   // { playerId: HTMLImageElement }
        this.defaultDuckImg = new Image();
        this.defaultDuckImg.src = '/static/images/duck_default.svg';
        this.defaultDuckImg.onerror = () => {
            this.defaultDuckImg.src = DEFAULT_DUCK_DATA_URI;
        };

        // Camera & Virtual Track Specs (Zoomed Follow View)
        this.worldHeight = 360;
        this.zoom = 1.25;
        this.cameraX = 0;
        this.cameraY = 0;
        this.targetCameraX = 0;
        this.virtualTrackLength = 2600;
        this.startLineX = 90;
        this.finishLineX = 2480;
        this.duckSize = 52;

        // Dynamic Weather & Environment
        this.weather = 'sunny'; // 'sunny' | 'sunset' | 'wind' | 'thunderstorm'
        this.lightningFlash = 0;
        this.nextLightningTime = 0;
        this.activeClashes = [];
    }

    setupWeather() {
        const weathers = ['sunny', 'sunny', 'sunset', 'wind', 'thunderstorm'];
        this.weather = weathers[Math.floor(Math.random() * weathers.length)];
        this.lightningFlash = 0;
        this.nextLightningTime = performance.now() + 2500 + Math.random() * 4000;
    }

    init() {
        if (!this.canvas) {
            this.canvas = document.getElementById(this.canvasId);
            this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        }
        if (!this.canvas) return;

        this.preloadImages();
        this.resizeCanvas();
        if (!this.boundResize) {
            window.addEventListener('resize', this.boundResize = () => this.resizeCanvas());
        }
        this.setupDucks();
        this.renderStatic();
        this.updateStatusText('Sẵn sàng xuất phát!');
        this.updateModeButtonsUI();
        this.updateDurationButtonsUI();
    }

    preloadImages() {
        this.allPlayers.forEach(p => {
            if (p.avatar && !this.avatarImages[p.id]) {
                const img = new Image();
                img.referrerPolicy = 'no-referrer';
                img.src = p.avatar;
                this.avatarImages[p.id] = img;
            }
            if (p.duck_image && !this.duckImages[p.id]) {
                const img = new Image();
                img.referrerPolicy = 'no-referrer';
                img.src = p.duck_image;
                this.duckImages[p.id] = img;
            }
        });
    }

    getDuckImage(duck) {
        if (duck.duck_image && this.duckImages[duck.id] && this.duckImages[duck.id].complete && this.duckImages[duck.id].naturalWidth > 0) {
            return this.duckImages[duck.id];
        }
        return this.defaultDuckImg;
    }

    setAllPlayers(players) {
        this.allPlayers = players;
        this.preloadImages();
        this.setupDucks();
        this.resizeCanvas();
        this.renderStatic();

        const countSpans = [
            document.getElementById('duck-total-count'),
            document.getElementById('tab-duck-total-count')
        ];
        countSpans.forEach(s => { if (s) s.innerText = this.ducks.length; });
    }

    setTeamMode(mode) {
        if (this.state === 'racing' || this.state === 'countdown') return;
        this.teamMode = mode;
        this.reset();
        this.updateModeButtonsUI();
    }

    setDuration(seconds) {
        if (this.state === 'racing' || this.state === 'countdown') return;
        this.targetDuration = Number(seconds) || 30;
        this.reset();
        this.updateDurationButtonsUI();
    }

    updateModeButtonsUI() {
        const modes = ['all', 'team1', 'team2'];
        modes.forEach(m => {
            const btns = [
                document.getElementById(`btn-mode-${m}`),
                document.getElementById(`tab-btn-mode-${m}`)
            ];
            btns.forEach(btn => {
                if (!btn) return;
                if (this.teamMode === m) {
                    btn.className = 'px-3 py-1 rounded-lg text-xs font-bold transition bg-amber-500 text-slate-950 shadow-xs';
                } else {
                    btn.className = 'px-3 py-1 rounded-lg text-xs font-bold text-slate-300 hover:text-white transition';
                }
            });
        });
    }

    updateDurationButtonsUI() {
        const durations = [15, 30, 45, 60];
        durations.forEach(d => {
            const btns = [
                document.getElementById(`btn-dur-${d}`),
                document.getElementById(`btn-tab-dur-${d}`)
            ];
            btns.forEach(btn => {
                if (!btn) return;
                if (this.targetDuration === d) {
                    btn.className = 'px-2.5 py-1 rounded-lg text-xs font-bold transition bg-amber-500 text-slate-950 shadow-xs';
                } else {
                    btn.className = 'px-2.5 py-1 rounded-lg text-xs font-bold text-slate-600 hover:text-slate-900 transition';
                }
            });
        });
    }

    getActivePlayers() {
        if (this.teamMode === 'team1') {
            const t1 = this.allPlayers.filter(p => p.team === 1);
            return t1.length > 0 ? t1 : [...this.allPlayers];
        } else if (this.teamMode === 'team2') {
            const t2 = this.allPlayers.filter(p => p.team === 2);
            return t2.length > 0 ? t2 : [...this.allPlayers];
        }
        return [...this.allPlayers];
    }

    resizeCanvas() {
        if (!this.canvas) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(640, Math.floor(rect.width));
        
        // Open river height: at least 420px, up to 520px
        const height = Math.max(420, Math.min(520, Math.floor(window.innerHeight * 0.52)));

        this.canvas.width = width * dpr;
        this.canvas.height = height * dpr;
        this.canvas.style.width = width + 'px';
        this.canvas.style.height = height + 'px';

        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(dpr, dpr);
        this.logicalWidth = width;
        this.logicalHeight = height;

        // Virtual World Scaling: Fit river height exactly into canvas height
        this.worldHeight = 360;
        this.zoom = this.logicalHeight / this.worldHeight;

        // Dynamic Virtual Track Length based on race duration
        this.virtualTrackLength = Math.max(2400, this.targetDuration * 85);
        this.startLineX = 90;
        this.finishLineX = this.virtualTrackLength - 120;
    }

    setupDucks() {
        this.ducks = [];
        this.finishOrder = [];
        this.cameraX = 0;
        this.targetCameraX = 0;

        this.setupWeather();

        const activePlayers = this.getActivePlayers();
        const N = Math.max(1, activePlayers.length);

        const trackDist = this.finishLineX - this.startLineX;
        const totalFrames = Math.max(300, this.targetDuration * 60);
        const nominalSpeed = trackDist / totalFrames;

        const countSpans = [
            document.getElementById('duck-total-count'),
            document.getElementById('tab-duck-total-count')
        ];
        countSpans.forEach(s => { if (s) s.innerText = activePlayers.length; });

        // Open river vertical swimming zone (in world coordinates [0, this.worldHeight])
        const waterTop = 36;
        const waterBottom = this.worldHeight - 36;
        const waterSpan = waterBottom - waterTop;

        // Determine columns for starting grid: 1 col for <=5 ducks, 2 cols for 6-12 ducks, 3 cols for >12 ducks
        const cols = N > 12 ? 3 : (N > 5 ? 2 : 1);
        const rows = Math.ceil(N / cols);
        const rowStep = rows > 1 ? (waterSpan - 36) / (rows - 1) : 0;

        activePlayers.forEach((p, idx) => {
            const variance = (Math.random() - 0.5) * 0.22;
            const baseSpd = nominalSpeed * (1 + variance);

            const col = idx % cols;
            const row = Math.floor(idx / cols);

            // Staggered grid placement so ducks don't align in a single cramped vertical column
            let baseY = rows === 1 
                ? (waterTop + waterSpan / 2) 
                : (waterTop + 18 + row * rowStep);
            
            // Stagger alternate column slightly for dynamic natural flock look
            if (cols > 1 && col % 2 === 1 && rows > 1) {
                baseY = Math.min(waterBottom - 20, baseY + rowStep * 0.28);
            }

            const jitterY = (Math.random() - 0.5) * 6;
            const finalY = Math.max(waterTop + 20, Math.min(waterBottom - 20, baseY + jitterY));

            // Starting X: Column 0 is closest to start line, subsequent columns line up slightly behind
            const colOffsetX = col * 42;
            const jitterX = (Math.random() - 0.5) * 10;
            const startX = this.startLineX - 32 - colOffsetX + jitterX;

            this.ducks.push({
                id: p.id,
                nickname: p.nickname || `Tuyển Thủ ${idx + 1}`,
                team: p.team || 0,
                avatar: p.avatar,
                duck_image: p.duck_image || '',
                x: startX,
                y: finalY,
                baseY: finalY,
                driftSeed: Math.random() * Math.PI * 2,
                baseSpeed: baseSpd,
                nominalSpeed: nominalSpeed,
                speedMultiplier: 1.0,
                tiltAngle: 0,
                effect: null,
                effectDuration: 0,
                spinAngle: 0,
                bobOffset: Math.random() * Math.PI * 2,
                finished: false,
                finishTime: 0,
                rank: null,
                isDrafting: false,
                draftTimer: 0,
                lastBumpTime: 0
            });
        });
    }

    startCountdown() {
        if (this.state === 'racing' || this.state === 'countdown') return;
        if (this.ducks.length < 2) {
            Swal.fire({
                icon: 'warning',
                title: 'Chưa đủ người chơi',
                text: 'Cần tối thiểu 2 tuyển thủ để bắt đầu cuộc đua!',
                ...SWAL_THEME
            });
            return;
        }

        duckAudio.init();
        this.state = 'countdown';

        const overlays = [
            document.getElementById('duck-race-countdown-overlay'),
            document.getElementById('tab-duck-race-countdown-overlay')
        ];
        const countNums = [
            document.getElementById('duck-race-countdown-number'),
            document.getElementById('tab-duck-race-countdown-number')
        ];
        const countSubs = [
            document.getElementById('duck-race-countdown-sub'),
            document.getElementById('tab-duck-race-countdown-sub')
        ];

        overlays.forEach(o => { if (o) o.classList.remove('hidden'); });

        let count = 3;
        countNums.forEach(n => { if (n) n.innerText = count; });
        countSubs.forEach(s => { if (s) s.innerText = 'Chuẩn bị xuất phát...'; });
        duckAudio.playCountdown(false);

        const timer = setInterval(() => {
            count--;
            if (count > 0) {
                countNums.forEach(n => { if (n) n.innerText = count; });
                duckAudio.playCountdown(false);
            } else if (count === 0) {
                countNums.forEach(n => {
                    if (n) {
                        n.innerText = 'GO! 🏁';
                        n.className = 'text-6xl sm:text-7xl font-black font-heading text-emerald-400 drop-shadow-[0_10px_20px_rgba(0,0,0,0.8)] animate-bounce';
                    }
                });
                countSubs.forEach(s => { if (s) s.innerText = 'VỊT XUẤT PHÁT!'; });
                duckAudio.playCountdown(true);
            } else {
                clearInterval(timer);
                overlays.forEach(o => { if (o) o.classList.add('hidden'); });
                countNums.forEach(n => {
                    if (n) {
                        n.className = 'text-7xl sm:text-8xl font-black font-heading text-amber-400 drop-shadow-[0_10px_20px_rgba(0,0,0,0.8)] animate-pulse';
                    }
                });
                this.startRacing();
            }
        }, 850);
    }

    startRacing() {
        this.state = 'racing';
        this.lastTime = performance.now();
        const firstEventDelay = (this.targetDuration / 30) * 1800;
        this.nextEventCheck = performance.now() + firstEventDelay;

        let statusMsg = `Cuộc đua (~${this.targetDuration}s) đang diễn ra! Hãy chú ý các biến cố lật kèo ⚡`;
        if (this.weather === 'thunderstorm') {
            statusMsg = '⛈️ Bão dông đổ bộ! Sấm chớp và mưa rào xuất hiện!';
            this.showToast('⛈️ Bão dông bất ngờ đổ bộ trên sông!', 'bg-slate-900 border-sky-400');
            duckAudio.playThunder();
        } else if (this.weather === 'wind') {
            statusMsg = '🍃 Gió lốc xuôi dòng trợ lực cho toàn bầy vịt!';
            this.showToast('🍃 Gió lốc thổi mạnh xuôi dòng!', 'bg-emerald-600');
        } else if (this.weather === 'sunset') {
            statusMsg = '🌅 Hoàng hôn rực lửa! Chặng đua nghẹt thở bắt đầu!';
            this.showToast('🌅 Hoàng hôn rực rỡ buông xuống!', 'bg-amber-600');
        }

        this.updateStatusText(statusMsg);
        this.loop();
    }

    triggerGlobalEnvironmentEvent() {
        if (this.state !== 'racing') return;
        const total = this.ducks.length;
        if (this.finishOrder.length >= Math.ceil(total * 0.75)) return;

        const activeDucks = this.ducks.filter(d => !d.finished);
        if (activeDucks.length === 0) return;

        const events = [
            { type: 'tidal_surge', label: 'SÓNG THẦN XUÔI DÒNG!', desc: 'Toàn bầy vịt cưỡi sóng lướt nhanh!', icon: '🌊', bg: 'bg-cyan-600' },
            { type: 'gale_winds', label: 'CUỒNG PHONG ĐẨY LƯNG!', desc: 'Gió giật xuôi dòng bứt phá toàn đàn!', icon: '🍃', bg: 'bg-emerald-600' },
            { type: 'undercurrent_drag', label: 'DÒNG XOÁY NGẦM!', desc: 'Dòng nước ngược ghì tốc độ toàn bầy!', icon: '🌀', bg: 'bg-blue-700' },
            { type: 'thunder_squall', label: 'DÔNG BÃO SẤM CHỚP!', desc: 'Mặt sông nhiễm điện làm tê liệt toàn đàn!', icon: '⛈️', bg: 'bg-amber-600' },
            { type: 'rainbow_blessing', label: 'CẦU VỒNG BAN PHƯỚC!', desc: 'Thanh tẩy mặt sông, trợ lực toàn thể vịt đua!', icon: '🌈', bg: 'bg-purple-600' },
            { type: 'river_tremor', label: 'DƯ CHẤN SÔNG NƯỚC!', desc: 'Sóng dập dềnh xáo trộn đường bơi!', icon: '🌊', bg: 'bg-teal-600' }
        ];

        const evt = events[Math.floor(Math.random() * events.length)];
        const durationScale = Math.max(0.6, this.targetDuration / 30);
        const waterTop = 36;
        const waterBottom = this.worldHeight - 36;

        if (evt.type === 'tidal_surge') {
            activeDucks.forEach(d => {
                d.effect = 'boost';
                d.speedMultiplier = 1.75;
                d.effectDuration = 2200 * durationScale;
                this.particles.addWaterSplash(d.x, d.y, 5);
            });
            this.showToast(`${evt.icon} ${evt.label} ${evt.desc}`, evt.bg);
            duckAudio.playSplash();
        } else if (evt.type === 'gale_winds') {
            activeDucks.forEach(d => {
                d.effect = 'tailwind';
                d.speedMultiplier = 1.45;
                d.effectDuration = 2500 * durationScale;
            });
            const viewW = this.logicalWidth / this.zoom;
            this.particles.addWindLeaves(this.cameraX - 40, this.cameraX + viewW + 40, 16);
            this.showToast(`${evt.icon} ${evt.label} ${evt.desc}`, evt.bg);
            duckAudio.playBoost();
        } else if (evt.type === 'undercurrent_drag') {
            activeDucks.forEach(d => {
                d.effect = 'whirlpool';
                d.speedMultiplier = 0.65;
                d.effectDuration = 2000 * durationScale;
                d.spinAngle = (Math.random() - 0.5) * 0.4;
            });
            this.showToast(`${evt.icon} ${evt.label} ${evt.desc}`, evt.bg);
            duckAudio.playWhirlpool();
        } else if (evt.type === 'thunder_squall') {
            this.lightningFlash = 9;
            activeDucks.forEach(d => {
                d.effect = 'shock';
                d.speedMultiplier = 0.72;
                d.effectDuration = 2200 * durationScale;
            });
            duckAudio.playThunder();
            duckAudio.playLightning();
            this.showToast(`${evt.icon} ${evt.label} ${evt.desc}`, evt.bg);
        } else if (evt.type === 'rainbow_blessing') {
            activeDucks.forEach(d => {
                d.effect = 'blessing';
                d.speedMultiplier = 1.40;
                d.effectDuration = 2400 * durationScale;
                this.particles.addConfetti(d.x, d.y, 4);
            });
            this.showToast(`${evt.icon} ${evt.label} ${evt.desc}`, evt.bg);
            duckAudio.playBoost();
        } else if (evt.type === 'river_tremor') {
            activeDucks.forEach(d => {
                const shift = (Math.random() - 0.5) * 36;
                d.baseY = Math.max(waterTop + 20, Math.min(waterBottom - 20, d.baseY + shift));
                this.particles.addWaterSplash(d.x, d.y, 6);
            });
            this.showToast(`${evt.icon} ${evt.label} ${evt.desc}`, evt.bg);
            duckAudio.playSplash();
        }
    }

    triggerSurpriseEvent() {
        this.triggerGlobalEnvironmentEvent();
    }

    showToast(message, bgClass = 'bg-amber-500') {
        const toasts = [
            { t: document.getElementById('duck-race-event-toast'), c: document.getElementById('duck-race-toast-content'), x: document.getElementById('duck-race-toast-text') },
            { t: document.getElementById('tab-duck-race-event-toast'), c: document.getElementById('tab-duck-race-toast-content'), x: document.getElementById('tab-duck-race-toast-text') }
        ];

        toasts.forEach(({ t, c, x }) => {
            if (!t || !c || !x) return;
            x.innerText = message;
            c.className = `px-5 py-2 rounded-2xl ${bgClass} text-slate-950 font-black font-heading text-xs sm:text-sm shadow-2xl flex items-center gap-2 border-2 border-white/80 animate-bounce`;
            t.classList.remove('opacity-0', '-translate-y-4', 'scale-90');
            t.classList.add('opacity-100', 'translate-y-0', 'scale-100');
        });

        if (this.activeToastTimer) clearTimeout(this.activeToastTimer);
        this.activeToastTimer = setTimeout(() => {
            toasts.forEach(({ t }) => {
                if (t) {
                    t.classList.add('opacity-0', '-translate-y-4', 'scale-90');
                    t.classList.remove('opacity-100', 'translate-y-0', 'scale-100');
                }
            });
        }, 2200);
    }

    loop() {
        if (this.state !== 'racing' && this.state !== 'finished') return;

        const now = performance.now();
        const dt = Math.min(40, now - this.lastTime);
        this.lastTime = now;

        this.update(dt, now);
        this.render();

        if (this.state === 'racing' || this.particles.particles.length > 0) {
            this.animFrameId = requestAnimationFrame(() => this.loop());
        }
    }

    update(dt, now) {
        this.waveOffset += 0.05;
        this.particles.update();

        const viewW = this.logicalWidth / this.zoom;

        // Dynamic Weather Particles & Events
        if (this.weather === 'thunderstorm') {
            this.particles.addRainDrops(this.cameraX - 40, this.cameraX + viewW + 120, 3);
            if (now > this.nextLightningTime) {
                this.lightningFlash = 7;
                duckAudio.playThunder();
                this.nextLightningTime = now + 4000 + Math.random() * 5500;
            }
        } else if (this.weather === 'wind') {
            if (Math.random() < 0.22) {
                this.particles.addWindLeaves(this.cameraX - 40, this.cameraX + viewW + 40, 1);
            }
        }
        const weatherSpeedMod = this.weather === 'wind' ? 1.13 : 1.0;

        if (now > this.nextEventCheck && this.state === 'racing') {
            this.triggerGlobalEnvironmentEvent();
            const minInterval = (this.targetDuration / 30) * 2600;
            const randInterval = (this.targetDuration / 30) * 2800;
            this.nextEventCheck = now + minInterval + Math.random() * randInterval;
        }

        const waterTop = 36;
        const waterBottom = this.worldHeight - 36;

        for (const duck of this.ducks) {
            if (duck.finished) continue;

            if (duck.effectDuration > 0) {
                duck.effectDuration -= dt;
                if (duck.effectDuration <= 0) {
                    duck.effect = null;
                    duck.speedMultiplier = 1.0;
                }
            }

            duck.bobOffset += 0.12;
            const strokeRhythm = 1 + Math.sin(duck.bobOffset * 2) * 0.22;
            const noise = (Math.random() - 0.48) * (duck.nominalSpeed * 0.30);

            const draftingMod = duck.isDrafting ? 1.25 : 1.0;
            const speed = (duck.baseSpeed * duck.speedMultiplier * draftingMod * weatherSpeedMod * strokeRhythm + noise) * (dt / 16.6);
            duck.x += Math.max(0.1, speed);

            // Natural undulating sinusoidal swimming drift inside open river
            const drift = Math.sin(now * 0.002 + duck.driftSeed) * 11 + Math.sin(now * 0.004 + duck.driftSeed * 2.3) * 5;
            duck.y = Math.max(waterTop + 20, Math.min(waterBottom - 20, duck.baseY + drift));

            // Tilt forward on boost
            duck.tiltAngle = (duck.speedMultiplier > 1.2 ? 0.14 : (duck.speedMultiplier < 0.8 ? -0.09 : 0));

            if (Math.random() < 0.28) {
                this.particles.addWaterSplash(duck.x - 22, duck.y + Math.sin(duck.bobOffset) * 2.8);
            }

            if (duck.effect === 'boost' || duck.effect === 'tailwind') {
                this.particles.addFireTrail(duck.x - 18, duck.y);
            } else if (duck.effect === 'blessing') {
                if (Math.random() < 0.35) {
                    this.particles.addConfetti(duck.x - 12, duck.y, 1);
                }
            }

            if (duck.effect === 'whirlpool') {
                duck.spinAngle += 0.25;
            } else {
                duck.spinAngle = 0;
            }

            if (duck.x >= this.finishLineX) {
                duck.finished = true;
                duck.finishTime = now;
                duck.rank = this.finishOrder.length + 1;
                this.finishOrder.push(duck);

                this.particles.addConfetti(this.finishLineX, duck.y, 35);

                if (duck.rank === 1) {
                    duckAudio.playFinishFanfare();
                    this.showToast(`🥇 QUÁN QUÂN: ${duck.nickname} về đích ĐẦU TIÊN! 🏆`, 'bg-amber-400');
                } else {
                    duckAudio.playQuack();
                }

                this.updateResultUI();
            }
        }

        // Competitive Rivalry: Elastic Bumping, Slipstream Drafting & Team Sparks
        this.activeClashes = [];
        const activeList = this.ducks.filter(d => !d.finished);
        const activeCount = activeList.length;

        for (let i = 0; i < activeCount; i++) {
            const d1 = activeList[i];
            for (let j = i + 1; j < activeCount; j++) {
                const d2 = activeList[j];
                const dx = Math.abs(d1.x - d2.x);
                const dy = Math.abs(d1.y - d2.y);

                // 1. Elastic Bumping (Quẹt sườn & nảy nhẹ khi quá sát nhau)
                if (dx < 32 && dy < 18) {
                    if (now - d1.lastBumpTime > 450 && now - d2.lastBumpTime > 450) {
                        d1.lastBumpTime = now;
                        d2.lastBumpTime = now;
                        const pushDir = d1.y >= d2.y ? 1 : -1;
                        d1.baseY = Math.min(waterBottom - 20, d1.baseY + pushDir * 9);
                        d2.baseY = Math.max(waterTop + 20, d2.baseY - pushDir * 9);
                        this.particles.addBumpSpark((d1.x + d2.x) / 2, (d1.y + d2.y) / 2, 5);
                        duckAudio.playBump();
                    }
                }

                // 2. Team Rivalry Clash (Khi 2 vịt khác đội so kè ngang hàng)
                if (d1.team !== 0 && d2.team !== 0 && d1.team !== d2.team) {
                    if (dx < 20 && dy < 30) {
                        this.activeClashes.push({
                            x1: d1.x,
                            y1: d1.y,
                            x2: d2.x,
                            y2: d2.y
                        });
                    }
                }
            }

            // 3. Slipstream / Drafting (Núp gió đuôi vịt trước)
            let isTuckedBehind = false;
            for (let k = 0; k < activeCount; k++) {
                if (i === k) continue;
                const frontDuck = activeList[k];
                const gapX = frontDuck.x - d1.x;
                const gapY = Math.abs(frontDuck.y - d1.y);

                if (gapX > 22 && gapX < 62 && gapY < 12) {
                    isTuckedBehind = true;
                    break;
                }
            }

            if (isTuckedBehind) {
                d1.draftTimer += dt;
                if (d1.draftTimer > 320) {
                    d1.isDrafting = true;
                    if (Math.random() < 0.42) {
                        this.particles.addDraftingTrail(d1.x - 16, d1.y);
                    }
                }
            } else {
                d1.draftTimer = Math.max(0, d1.draftTimer - dt * 2);
                if (d1.draftTimer <= 0) {
                    d1.isDrafting = false;
                }
            }
        }

        // Camera Follow Logic (Smooth Tracking Camera Zoomed on the Flock)
        const activeDucks = this.ducks.filter(d => !d.finished);
        if (activeDucks.length > 0) {
            const leaderX = Math.max(...activeDucks.map(d => d.x));
            const avgX = activeDucks.reduce((s, d) => s + d.x, 0) / activeDucks.length;
            const focusX = leaderX * 0.65 + avgX * 0.35;
            let targetX = focusX - viewW * 0.40;
            const maxCamX = Math.max(0, this.finishLineX - viewW * 0.70);
            targetX = Math.max(0, Math.min(targetX, maxCamX));
            this.cameraX += (targetX - this.cameraX) * 0.08;
        } else {
            const targetX = Math.max(0, this.finishLineX - viewW * 0.60);
            this.cameraX += (targetX - this.cameraX) * 0.05;
        }

        if (this.finishOrder.length === this.ducks.length && this.state === 'racing') {
            this.state = 'finished';
            this.onRaceComplete();
        }
    }

    updateResultUI() {
        const containers = [
            document.getElementById('duck-race-result-container'),
            document.getElementById('tab-duck-race-result-container')
        ];
        const countSpans = [
            document.getElementById('duck-finished-count'),
            document.getElementById('tab-duck-finished-count')
        ];
        const totalSpans = [
            document.getElementById('duck-total-count'),
            document.getElementById('tab-duck-total-count')
        ];
        const grids = [
            document.getElementById('duck-race-result-grid'),
            document.getElementById('tab-duck-race-result-grid')
        ];

        containers.forEach(c => { if (c) c.classList.remove('hidden'); });
        countSpans.forEach(s => { if (s) s.innerText = this.finishOrder.length; });
        totalSpans.forEach(s => { if (s) s.innerText = this.ducks.length; });

        grids.forEach(grid => {
            if (!grid) return;
            grid.innerHTML = '';
            this.finishOrder.forEach((duck, idx) => {
                const rank = idx + 1;
                const isBlue = duck.team === 1;
                const isRed = duck.team === 2;
                const teamBadgeClass = isBlue ? 'bg-blue-500/20 text-blue-700 border-blue-500/40' : (isRed ? 'bg-rose-500/20 text-rose-700 border-rose-500/40' : 'bg-slate-100 text-slate-700 border-slate-200');

                let rankBadge = `${rank}.`;
                let rankClass = 'text-slate-600 bg-slate-100 border-slate-200';
                if (rank === 1) {
                    rankBadge = '🥇 #1';
                    rankClass = 'text-amber-800 bg-amber-100 border-amber-300 font-extrabold';
                } else if (rank === 2) {
                    rankBadge = '🥈 #2';
                    rankClass = 'text-slate-700 bg-slate-200 border-slate-300 font-bold';
                } else if (rank === 3) {
                    rankBadge = '🥉 #3';
                    rankClass = 'text-amber-900 bg-amber-200/50 border-amber-400/40 font-bold';
                }

                const card = document.createElement('div');
                card.className = `p-2.5 rounded-xl bg-white border border-slate-200 shadow-xs flex items-center justify-between gap-2 transition hover:border-slate-400 animate-in fade-in zoom-in-95 duration-150`;
                card.innerHTML = `
                    <div class="flex items-center gap-2 truncate">
                        <span class="px-2 py-0.5 rounded-lg border text-[11px] font-black font-heading shrink-0 ${rankClass}">${rankBadge}</span>
                        <span class="font-bold text-xs text-slate-900 truncate max-w-[110px]" title="${duck.nickname}">${duck.nickname}</span>
                    </div>
                    <div class="flex items-center gap-1.5 shrink-0">
                        <span class="text-[9px] px-1.5 py-0.5 rounded border font-semibold ${teamBadgeClass}">
                            ${isBlue ? 'Đội Xanh' : (isRed ? 'Đội Đỏ' : 'Tự Do')}
                        </span>
                        <button type="button" onclick="eliminatePlayerAndContinue('${duck.id}')" title="Loại người này khỏi cuộc đua tiếp theo"
                                class="px-1.5 py-0.5 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-600 border border-rose-200 text-[10px] font-bold transition flex items-center gap-1">
                            <i class="fa-solid fa-xmark text-[9px]"></i> Loại
                        </button>
                    </div>
                `;
                grid.appendChild(card);
            });
        });

        // Update Winner Hero Card in Tab
        if (this.finishOrder.length > 0) {
            const winner = this.finishOrder[0];
            const winnerNameEl = document.getElementById('duck-winner-name');
            if (winnerNameEl) {
                winnerNameEl.innerHTML = `
                    <span class="text-amber-700 font-extrabold text-xl">${winner.nickname}</span>
                    ${winner.team === 1 ? '<span class="ml-2 text-xs px-2.5 py-0.5 rounded-full bg-blue-100 text-blue-800 font-bold">Đội Xanh</span>' : (winner.team === 2 ? '<span class="ml-2 text-xs px-2.5 py-0.5 rounded-full bg-rose-100 text-rose-800 font-bold">Đội Đỏ</span>' : '')}
                `;
            }
            const winnerHero = document.getElementById('duck-winner-hero-card');
            if (winnerHero) winnerHero.classList.remove('hidden');
        }
    }

    onRaceComplete() {
        this.updateStatusText('🏁 Cuộc đua kết thúc! Thứ tự Ban / Pick đã được ấn định.');
        const btnRestarts = [
            document.getElementById('btn-restart-duck-race'),
            document.getElementById('btn-tab-restart-race')
        ];
        btnRestarts.forEach(b => { if (b) b.classList.remove('hidden'); });

        this.particles.addConfetti(this.finishLineX, this.worldHeight / 2, 80);
    }

    render() {
        if (!this.ctx) return;
        const ctx = this.ctx;
        const W = this.logicalWidth;
        const H = this.logicalHeight;
        const worldH = this.worldHeight;

        ctx.clearRect(0, 0, W, H);

        ctx.save();
        // Zoom and follow flock
        ctx.scale(this.zoom, this.zoom);
        ctx.translate(-this.cameraX, -this.cameraY);

        const viewW = W / this.zoom;
        const startDrawX = this.cameraX - 100;
        const endDrawX = this.cameraX + viewW + 100;

        // 1. Draw River & Water Waves (with Dynamic Weather Gradients)
        this.drawRiverEnvironment(startDrawX, endDrawX, worldH);

        // 2. Draw Start Line & Track Props
        this.drawTrackScenery(startDrawX, endDrawX, worldH);

        // 4. Draw Finish Line
        this.drawFinishLine(worldH);

        // 5. Draw Ducks with Z-Ordering (Y-depth sorting)
        const sortedDucks = [...this.ducks].sort((a, b) => a.y - b.y);
        for (const duck of sortedDucks) {
            this.drawDuck(duck);
        }

        // 6. Draw Team Rivalry Clash Sparks (when opposing teams swim neck-and-neck)
        this.drawClashSparks(ctx);

        // 7. Draw Particle System (Splashes, Confetti, Sparks, Rain, Wind Leaves)
        this.particles.draw(ctx);

        // 8. Lightning Flash Overlay (for Thunderstorm Weather)
        if (this.lightningFlash > 0) {
            ctx.fillStyle = `rgba(255, 255, 255, ${this.lightningFlash * 0.08})`;
            ctx.fillRect(startDrawX, 0, endDrawX - startDrawX, worldH);
            this.lightningFlash--;
        }

        ctx.restore();
    }

    renderStatic() {
        this.render();
    }

    drawRiverEnvironment(startDrawX, endDrawX, H) {
        const ctx = this.ctx;

        // River water gradient based on weather
        const riverGrad = ctx.createLinearGradient(0, 0, 0, H);
        if (this.weather === 'sunset') {
            riverGrad.addColorStop(0, '#581c87');    // tím hoàng hôn
            riverGrad.addColorStop(0.45, '#9a3412'); // cam sẫm
            riverGrad.addColorStop(1, '#c2410c');    // cam đỏ rực rỡ
        } else if (this.weather === 'thunderstorm') {
            riverGrad.addColorStop(0, '#09182a');    // xanh đen bão tố
            riverGrad.addColorStop(0.5, '#0f2942');
            riverGrad.addColorStop(1, '#0c1e30');
        } else {
            // sunny & wind
            riverGrad.addColorStop(0, '#0c4a6e');
            riverGrad.addColorStop(0.5, '#075985');
            riverGrad.addColorStop(1, '#0369a1');
        }
        ctx.fillStyle = riverGrad;
        ctx.fillRect(startDrawX, 0, endDrawX - startDrawX, H);

        // Animated wavy surface water lines
        ctx.save();
        ctx.strokeStyle = this.weather === 'sunset' ? 'rgba(254, 215, 170, 0.14)' : 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1.5;

        for (let y = 46; y < H - 46; y += 38) {
            ctx.beginPath();
            const step = 20;
            const startXAligned = Math.floor(startDrawX / step) * step;
            for (let x = startXAligned; x < endDrawX; x += step) {
                const waveY = y + Math.sin((x * 0.02) + this.waveOffset + (y * 0.1)) * 3.5;
                if (x === startXAligned) ctx.moveTo(x, waveY);
                else ctx.lineTo(x, waveY);
            }
            ctx.stroke();
        }
        ctx.restore();

        // Top River Bank (Grass + Stone border + Water foam)
        const grassColor = this.weather === 'sunset' ? '#4d7c0f' : (this.weather === 'thunderstorm' ? '#14532d' : '#15803d');
        ctx.fillStyle = grassColor;
        ctx.fillRect(startDrawX, 0, endDrawX - startDrawX, 26);
        ctx.fillStyle = '#334155';
        ctx.fillRect(startDrawX, 26, endDrawX - startDrawX, 6);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.fillRect(startDrawX, 32, endDrawX - startDrawX, 2.5);

        // Bottom River Bank (Water foam + Stone border + Grass)
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.fillRect(startDrawX, H - 34.5, endDrawX - startDrawX, 2.5);
        ctx.fillStyle = '#334155';
        ctx.fillRect(startDrawX, H - 32, endDrawX - startDrawX, 6);
        ctx.fillStyle = grassColor;
        ctx.fillRect(startDrawX, H - 26, endDrawX - startDrawX, 26);
    }

    drawClashSparks(ctx) {
        if (!this.activeClashes || this.activeClashes.length === 0) return;

        ctx.save();
        for (const c of this.activeClashes) {
            const midX = (c.x1 + c.x2) / 2;
            const midY = (c.y1 + c.y2) / 2;

            ctx.strokeStyle = '#fde047';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(c.x1, c.y1);
            ctx.lineTo(midX + (Math.random() - 0.5) * 12, midY + (Math.random() - 0.5) * 10);
            ctx.lineTo(c.x2, c.y2);
            ctx.stroke();

            ctx.fillStyle = '#fde047';
            ctx.font = 'bold 13px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('⚡', midX, midY - 6);
        }
        ctx.restore();
    }

    drawTrackScenery(startDrawX, endDrawX, H) {
        const ctx = this.ctx;

        // Start Line & Wooden Platform
        if (this.startLineX >= startDrawX - 60 && this.startLineX <= endDrawX + 60) {
            ctx.save();
            ctx.fillStyle = '#78350f';
            ctx.fillRect(this.startLineX - 35, 32, 18, H - 64);
            ctx.strokeStyle = '#b45309';
            ctx.lineWidth = 2;
            ctx.strokeRect(this.startLineX - 35, 32, 18, H - 64);

            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 3;
            ctx.setLineDash([8, 8]);
            ctx.beginPath();
            ctx.moveTo(this.startLineX, 32);
            ctx.lineTo(this.startLineX, H - 32);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = '#38bdf8';
            ctx.font = 'bold 11px font-heading, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('🏁 START', this.startLineX - 26, 22);
            ctx.restore();
        }

        // Distance markers / floating buoys along the river
        for (let dist = 500; dist < this.finishLineX - 200; dist += 500) {
            if (dist >= startDrawX - 40 && dist <= endDrawX + 40) {
                ctx.save();
                // Floating red/white buoy
                ctx.fillStyle = '#ef4444';
                ctx.beginPath();
                ctx.arc(dist, 44, 7, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#ffffff';
                ctx.beginPath();
                ctx.arc(dist, 44, 3.5, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = 'rgba(255, 255, 255, 0.65)';
                ctx.font = 'bold 9px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(`${dist}m`, dist, 60);

                // Bottom buoy
                ctx.fillStyle = '#ef4444';
                ctx.beginPath();
                ctx.arc(dist, H - 44, 7, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#ffffff';
                ctx.beginPath();
                ctx.arc(dist, H - 44, 3.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            }
        }
    }

    drawFinishLine(H) {
        const ctx = this.ctx;
        const x = this.finishLineX;
        const boxSize = 12;

        ctx.save();
        for (let y = 32; y < H - 32; y += boxSize) {
            const isWhite1 = Math.floor(y / boxSize) % 2 === 0;
            ctx.fillStyle = isWhite1 ? '#ffffff' : '#0f172a';
            ctx.fillRect(x, y, boxSize, boxSize);

            ctx.fillStyle = !isWhite1 ? '#ffffff' : '#0f172a';
            ctx.fillRect(x + boxSize, y, boxSize, boxSize);
        }

        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.moveTo(x, 26);
        ctx.lineTo(x, H - 26);
        ctx.stroke();

        ctx.fillStyle = '#f59e0b';
        ctx.fillRect(x - 8, 8, boxSize * 2 + 16, 18);
        ctx.fillStyle = '#0f172a';
        ctx.font = '900 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('FINISH', x + boxSize, 18);
        ctx.restore();
    }

    drawDuck(duck) {
        const ctx = this.ctx;
        const img = this.getDuckImage(duck);
        const size = this.duckSize;
        const bobbingY = Math.sin(duck.bobOffset) * 2.8;

        // 1. Water Contact Shadow
        ctx.save();
        ctx.fillStyle = 'rgba(2, 6, 23, 0.24)';
        ctx.beginPath();
        ctx.ellipse(duck.x, duck.y + bobbingY + size * 0.30, size * 0.40, size * 0.14, 0, 0, Math.PI * 2);
        ctx.fill();

        // Water Wake Waves
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.32)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(duck.x - size * 0.35, duck.y + bobbingY + 2, size * 0.20, Math.PI * 0.6, Math.PI * 1.4);
        ctx.stroke();
        ctx.restore();

        // 2. Standardized 2D Image Duck Sprite
        ctx.save();
        ctx.translate(duck.x, duck.y + bobbingY);

        // Slipstream Drafting Aura Trail (Aerodynamic streamlines)
        if (duck.isDrafting) {
            ctx.save();
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.8)';
            ctx.lineWidth = 2.5;
            ctx.setLineDash([7, 4]);
            ctx.beginPath();
            ctx.moveTo(-size * 0.45, -size * 0.22);
            ctx.lineTo(-size * 0.95, -size * 0.22);
            ctx.moveTo(-size * 0.45, size * 0.22);
            ctx.lineTo(-size * 0.95, size * 0.22);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        if (duck.spinAngle !== 0) {
            ctx.rotate(duck.spinAngle);
        } else if (duck.tiltAngle !== 0) {
            ctx.rotate(duck.tiltAngle);
        }

        // Draw the Duck Image
        ctx.drawImage(img, -size / 2, -size / 2, size, size);

        // Powerup Indicators
        if (duck.effect === 'boost' || duck.effect === 'tailwind') {
            ctx.font = '16px sans-serif';
            ctx.fillText('🚀', -size * 0.5, -size * 0.15);
        } else if (duck.effect === 'shock') {
            ctx.font = '17px sans-serif';
            ctx.fillText('⚡', 0, -size * 0.42);
        } else if (duck.effect === 'whirlpool') {
            ctx.font = '16px sans-serif';
            ctx.fillText('🌀', 0, -size * 0.42);
        }
        ctx.restore();

        // 3. Clean Name Tag (Option B: No team badge, just clear pill with mini avatar and nickname)
        ctx.save();
        const avatarImg = this.avatarImages[duck.id];
        let displayName = duck.nickname;
        if (duck.isDrafting) {
            displayName = '💨 ' + displayName;
        }
        if (displayName.length > 14) {
            displayName = displayName.substring(0, 13) + '…';
        }

        ctx.font = 'bold 9.5px "Plus Jakarta Sans", sans-serif';
        const textMetrics = ctx.measureText(displayName);
        const textW = textMetrics.width;
        const hasAvatar = avatarImg && avatarImg.complete && avatarImg.naturalWidth > 0;
        const tagH = 17;
        const tagW = textW + (hasAvatar ? 23 : 13);
        const tagY = duck.y + bobbingY - size * 0.54;
        const tagX = duck.x - tagW / 2;

        // Dark translucent glassmorphism pill (Cyan glow on drafting)
        ctx.fillStyle = duck.isDrafting ? 'rgba(12, 74, 110, 0.92)' : 'rgba(15, 23, 42, 0.88)';
        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') {
            ctx.roundRect(tagX, tagY - tagH / 2, tagW, tagH, 8.5);
        } else {
            ctx.rect(tagX, tagY - tagH / 2, tagW, tagH);
        }
        ctx.fill();
        ctx.strokeStyle = duck.isDrafting ? '#38bdf8' : (duck.finished && duck.rank === 1 ? '#f59e0b' : 'rgba(255, 255, 255, 0.22)');
        ctx.lineWidth = duck.isDrafting ? 1.6 : 1;
        ctx.stroke();

        let textStartX = tagX + 6;
        if (hasAvatar) {
            const avR = 6;
            const avX = tagX + 8;
            ctx.save();
            ctx.beginPath();
            ctx.arc(avX, tagY, avR, 0, Math.PI * 2);
            ctx.closePath();
            ctx.clip();
            ctx.drawImage(avatarImg, avX - avR, tagY - avR, avR * 2, avR * 2);
            ctx.restore();
            textStartX = tagX + 17;
        }

        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(displayName, textStartX, tagY);

        // Rank Medal next to name if finished
        if (duck.finished && duck.rank) {
            const badgeX = tagX + tagW + 7;
            let medal = `${duck.rank}`;
            let bg = '#64748b';
            if (duck.rank === 1) { medal = '🥇'; bg = '#f59e0b'; }
            else if (duck.rank === 2) { medal = '🥈'; bg = '#94a3b8'; }
            else if (duck.rank === 3) { medal = '🥉'; bg = '#d97706'; }

            ctx.fillStyle = bg;
            ctx.beginPath();
            ctx.arc(badgeX, tagY, 7.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 8.5px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(medal, badgeX, tagY + 0.5);
        }
        ctx.restore();
    }

    updateStatusText(text) {
        const els = [
            document.getElementById('duck-race-status-text'),
            document.getElementById('tab-duck-status-text')
        ];
        els.forEach(el => { if (el) el.innerText = text; });
    }

    reset() {
        if (this.animFrameId) {
            cancelAnimationFrame(this.animFrameId);
            this.animFrameId = null;
        }
        this.state = 'idle';
        this.setupDucks();
        this.renderStatic();

        this.updateStatusText(`Sẵn sàng xuất phát (${this.ducks.length} tuyển thủ, ~${this.targetDuration}s)!`);

        const grids = [
            document.getElementById('duck-race-result-grid'),
            document.getElementById('tab-duck-race-result-grid')
        ];
        grids.forEach(g => { if (g) g.innerHTML = ''; });

        const countSpans = [
            document.getElementById('duck-finished-count'),
            document.getElementById('tab-duck-finished-count')
        ];
        countSpans.forEach(s => { if (s) s.innerText = '0'; });

        const totalSpans = [
            document.getElementById('duck-total-count'),
            document.getElementById('tab-duck-total-count')
        ];
        totalSpans.forEach(s => { if (s) s.innerText = this.ducks.length; });

        const containers = [
            document.getElementById('duck-race-result-container'),
            document.getElementById('tab-duck-race-result-container')
        ];
        containers.forEach(c => { if (c) c.classList.add('hidden'); });

        const btnStarts = [
            document.getElementById('btn-start-duck-race'),
            document.getElementById('btn-tab-start-race')
        ];
        btnStarts.forEach(b => { if (b) b.classList.remove('hidden'); });

        const btnRestarts = [
            document.getElementById('btn-restart-duck-race'),
            document.getElementById('btn-tab-restart-race')
        ];
        btnRestarts.forEach(b => { if (b) b.classList.add('hidden'); });
    }

    destroy() {
        if (this.animFrameId) {
            cancelAnimationFrame(this.animFrameId);
            this.animFrameId = null;
        }
        if (this.boundResize) {
            window.removeEventListener('resize', this.boundResize);
        }
    }
}

// Global instance & Roster manager
window.duckRaceGameInstance = null;

// ==========================================
// ROSTER MANAGEMENT FOR DUCK RACE TAB
// ==========================================
function initDuckRaceTab() {
    if (duckRaceRoster.length === 0) {
        if (currentTeamsResult && currentTeamsResult.team1 && currentTeamsResult.team2) {
            loadTeamsToDuckRaceRoster(false);
        } else if (allPlayers && allPlayers.length > 0) {
            duckRaceRoster = allPlayers.slice(0, 10).map((p, idx) => ({
                id: p.id,
                nickname: p.nickname,
                avatar: p.avatar,
                duck_image: p.duck_image || '',
                team: idx < 5 ? 1 : 2
            }));
        }
    }

    renderDuckRaceRoster();

    // Initialize or resize game canvas
    const canvas = document.getElementById('tab-duck-race-canvas');
    if (canvas) {
        if (window.duckRaceGameInstance) {
            window.duckRaceGameInstance.destroy();
        }
        window.duckRaceGameInstance = new DuckRaceGame('tab-duck-race-canvas', duckRaceRoster);
        window.duckRaceGameInstance.init();
    }
}

function renderDuckRaceRoster() {
    const container = document.getElementById('duck-roster-chips-container');
    const badge = document.getElementById('roster-count-badge');
    if (!container) return;

    if (badge) badge.innerText = `${duckRaceRoster.length} người chơi`;

    if (duckRaceRoster.length === 0) {
        container.innerHTML = `
            <p id="duck-roster-empty-notice" class="text-xs text-slate-400 italic">
                Chưa có ai trong danh sách. Hãy nhập tên ở trên hoặc bấm "Nạp Từ Chia Đội" / "Chọn Ngẫu Nhiên".
            </p>
        `;
    } else {
        container.innerHTML = '';
        duckRaceRoster.forEach(p => {
            const isBlue = p.team === 1;
            const isRed = p.team === 2;
            const borderClass = isBlue ? 'border-blue-300 bg-blue-50/80 text-blue-900' : (isRed ? 'border-rose-300 bg-rose-50/80 text-rose-900' : 'border-amber-300 bg-amber-50/80 text-slate-900');
            const teamDot = isBlue ? 'bg-blue-500' : (isRed ? 'bg-rose-500' : 'bg-amber-500');

            const chip = document.createElement('div');
            chip.className = `inline-flex items-center gap-2 pl-2 pr-1.5 py-1 rounded-xl border text-xs font-semibold shadow-2xs transition hover:shadow-xs ${borderClass}`;
            chip.innerHTML = `
                <span class="w-2 h-2 rounded-full ${teamDot} shrink-0"></span>
                <span class="font-bold truncate max-w-[120px]">${p.nickname}</span>
                <button type="button" onclick="removePlayerFromDuckRaceRoster('${p.id}')" title="Xoá tuyển thủ này"
                        class="w-5 h-5 rounded-lg hover:bg-slate-200/80 text-slate-400 hover:text-rose-600 flex items-center justify-center transition text-xs shrink-0">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            `;
            container.appendChild(chip);
        });
    }

    if (window.duckRaceGameInstance) {
        window.duckRaceGameInstance.setAllPlayers(duckRaceRoster);
    }
}

function handleAddCustomDuckPlayer() {
    const input = document.getElementById('duck-input-player-name');
    if (!input) return;
    const name = input.value.trim();
    if (!name) return;

    // Generate cute avatar URL
    const avatar = `https://api.dicebear.com/7.x/bottts/svg?seed=${encodeURIComponent(name)}`;
    const newPlayer = {
        id: `custom_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
        nickname: name,
        avatar: avatar,
        team: 0
    };

    duckRaceRoster.push(newPlayer);
    input.value = '';
    renderDuckRaceRoster();
}

function removePlayerFromDuckRaceRoster(playerId) {
    duckRaceRoster = duckRaceRoster.filter(p => p.id !== playerId);
    renderDuckRaceRoster();
}

function clearDuckRaceRoster() {
    duckRaceRoster = [];
    renderDuckRaceRoster();
}

function loadTeamsToDuckRaceRoster(showToast = true) {
    if (!currentTeamsResult || !currentTeamsResult.team1 || !currentTeamsResult.team2) {
        if (showToast) {
            Swal.fire({
                icon: 'warning',
                title: 'Chưa có kết quả chia đội',
                text: 'Hãy sang tab "Chia Đội" chọn 10 tuyển thủ và bấm "Chia Đội Tối Ưu" trước.',
                ...SWAL_THEME
            });
        }
        return;
    }

    duckRaceRoster = [
        ...currentTeamsResult.team1.map(p => ({ ...p, duck_image: p.duck_image || '', team: 1 })),
        ...currentTeamsResult.team2.map(p => ({ ...p, duck_image: p.duck_image || '', team: 2 }))
    ];
    renderDuckRaceRoster();

    if (showToast) {
        Swal.fire({
            icon: 'success',
            title: 'Đã nạp 10 tuyển thủ!',
            text: 'Đã nạp thành công 10 tuyển thủ từ kết quả chia đội vào danh sách đua vịt.',
            timer: 1500,
            showConfirmButton: false,
            ...SWAL_THEME
        });
    }
}

function addRandomPlayersToRoster(count = 10) {
    if (!allPlayers || allPlayers.length === 0) {
        Swal.fire({
            icon: 'warning',
            title: 'Chưa tải được danh sách tuyển thủ',
            text: 'Vui lòng kiểm tra kết nối hệ thống.',
            ...SWAL_THEME
        });
        return;
    }

    const unpicked = allPlayers.filter(p => !duckRaceRoster.some(r => r.id === p.id));
    const pool = unpicked.length >= count ? unpicked : allPlayers;
    const shuffled = [...pool].sort(() => Math.random() - 0.5);
    const selected = shuffled.slice(0, count);

    selected.forEach((p, idx) => {
        if (!duckRaceRoster.some(r => r.id === p.id)) {
            duckRaceRoster.push({
                id: p.id,
                nickname: p.nickname,
                avatar: p.avatar,
                duck_image: p.duck_image || '',
                team: idx < Math.floor(count / 2) ? 1 : 2
            });
        }
    });

    renderDuckRaceRoster();
}

// ==========================================
// ELIMINATE WINNER AND CONTINUE (CORE FEATURE)
// ==========================================
function eliminateWinnerAndContinue() {
    if (!window.duckRaceGameInstance || window.duckRaceGameInstance.finishOrder.length === 0) {
        Swal.fire({
            icon: 'info',
            title: 'Chưa xác định người về đích',
            text: 'Hãy bấm Bắt Đầu Đua và chờ vịt cán đích trước.',
            ...SWAL_THEME
        });
        return;
    }

    const winner = window.duckRaceGameInstance.finishOrder[0];
    const winnerName = winner.nickname;

    // Eliminate winner from roster
    duckRaceRoster = duckRaceRoster.filter(p => p.id !== winner.id);
    renderDuckRaceRoster();

    if (window.duckRaceGameInstance) {
        window.duckRaceGameInstance.reset();
    }

    const winnerHero = document.getElementById('duck-winner-hero-card');
    if (winnerHero) winnerHero.classList.add('hidden');

    Swal.fire({
        icon: 'success',
        title: `Đã loại người thắng: ${winnerName}! 🏆`,
        text: `Danh sách còn lại ${duckRaceRoster.length} người chơi. Sẵn sàng đua tiếp cho các thứ hạng tiếp theo!`,
        timer: 2000,
        showConfirmButton: false,
        ...SWAL_THEME
    });
}

function eliminatePlayerAndContinue(playerId) {
    const target = duckRaceRoster.find(p => p.id === playerId);
    const targetName = target ? target.nickname : 'tuyển thủ';

    duckRaceRoster = duckRaceRoster.filter(p => p.id !== playerId);
    renderDuckRaceRoster();

    if (window.duckRaceGameInstance) {
        window.duckRaceGameInstance.reset();
    }

    Swal.fire({
        icon: 'info',
        title: `Đã loại: ${targetName}`,
        text: `Danh sách còn lại ${duckRaceRoster.length} người chơi.`,
        timer: 1500,
        showConfirmButton: false,
        ...SWAL_THEME
    });
}

// System Player Picker Modal
function openSystemPlayerPickerModal() {
    const modal = document.getElementById('modal-duck-player-picker');
    if (!modal) return;
    modal.classList.remove('hidden');
    renderSystemPlayerPickerGrid();
}

function closeSystemPlayerPickerModal() {
    const modal = document.getElementById('modal-duck-player-picker');
    if (modal) modal.classList.add('hidden');
}

function renderSystemPlayerPickerGrid() {
    const grid = document.getElementById('duck-picker-players-grid');
    const search = (document.getElementById('duck-picker-search')?.value || '').toLowerCase().trim();
    if (!grid) return;

    grid.innerHTML = '';
    allPlayers.forEach(p => {
        if (search && !p.nickname.toLowerCase().includes(search) && !p.id.toLowerCase().includes(search)) {
            return;
        }
        const isAdded = duckRaceRoster.some(r => r.id === p.id);
        const card = document.createElement('div');
        card.className = `p-2 rounded-xl border flex items-center gap-2 cursor-pointer transition select-none ${
            isAdded ? 'bg-indigo-50 border-indigo-300 opacity-60' : 'bg-white border-slate-200 hover:border-amber-400 hover:shadow-xs'
        }`;
        card.onclick = () => {
            if (!isAdded) {
                duckRaceRoster.push({ id: p.id, nickname: p.nickname, avatar: p.avatar, duck_image: p.duck_image || '', team: 0 });
                renderDuckRaceRoster();
                renderSystemPlayerPickerGrid();
            }
        };

        card.innerHTML = `
            <img src="${p.avatar}" alt="${p.nickname}" class="w-8 h-8 rounded-lg object-cover bg-slate-100 border border-slate-200 shrink-0">
            <div class="truncate flex-1">
                <h5 class="text-xs font-bold text-slate-900 truncate">${p.nickname}</h5>
                <span class="text-[10px] text-slate-500">${isAdded ? '✓ Đã thêm' : '+ Thêm'}</span>
            </div>
        `;
        grid.appendChild(card);
    });
}

// Global button helpers
function setDuckRaceTeamMode(mode) {
    if (window.duckRaceGameInstance) {
        window.duckRaceGameInstance.setTeamMode(mode);
    }
}

function setDuckRaceDuration(seconds) {
    if (window.duckRaceGameInstance) {
        window.duckRaceGameInstance.setDuration(seconds);
    }
}

function openDuckRace() {
    if (currentTeamsResult && currentTeamsResult.team1 && currentTeamsResult.team2) {
        loadTeamsToDuckRaceRoster(false);
    }
    switchTab('duckrace');
    const el = document.getElementById('tab-content-duckrace');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
}

function closeDuckRace() {
    const overlay = document.getElementById('duck-race-overlay');
    if (overlay) overlay.classList.add('hidden');
    document.body.style.overflow = '';

    if (window.duckRaceGameInstance) {
        window.duckRaceGameInstance.destroy();
        window.duckRaceGameInstance = null;
    }
}

function startDuckRace() {
    const btnStarts = [
        document.getElementById('btn-start-duck-race'),
        document.getElementById('btn-tab-start-race')
    ];
    btnStarts.forEach(b => { if (b) b.classList.add('hidden'); });

    if (window.duckRaceGameInstance) {
        window.duckRaceGameInstance.startCountdown();
    }
}

function restartDuckRace() {
    const btnStarts = [
        document.getElementById('btn-start-duck-race'),
        document.getElementById('btn-tab-start-race')
    ];
    btnStarts.forEach(b => { if (b) b.classList.remove('hidden'); });

    const btnRestarts = [
        document.getElementById('btn-restart-duck-race'),
        document.getElementById('btn-tab-restart-race')
    ];
    btnRestarts.forEach(b => { if (b) b.classList.add('hidden'); });

    if (window.duckRaceGameInstance) {
        window.duckRaceGameInstance.reset();
    }
}
