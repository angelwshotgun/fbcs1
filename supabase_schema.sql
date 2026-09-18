-- ==========================================================
-- FBCS ESPORTS DATABASE SCHEMA FOR SUPABASE (POSTGRESQL)
-- Chạy script này trong Supabase Dashboard > SQL Editor
-- ==========================================================

-- 1. BẢNG HỒ SƠ TUYỂN THỦ (PLAYERS)
CREATE TABLE IF NOT EXISTS public.players (
    id VARCHAR(64) PRIMARY KEY,
    nickname VARCHAR(100) NOT NULL,
    avatar TEXT,
    skill NUMERIC(4, 2) DEFAULT 7.00 CHECK (skill >= 1.0 AND skill <= 10.0),
    champion_pool NUMERIC(4, 2) DEFAULT 7.00 CHECK (champion_pool >= 1.0 AND champion_pool <= 10.0),
    flex_lane NUMERIC(4, 2) DEFAULT 6.50 CHECK (flex_lane >= 1.0 AND flex_lane <= 10.0),
    consistency NUMERIC(4, 2) DEFAULT 7.00 CHECK (consistency >= 1.0 AND consistency <= 10.0),
    stats_ovr NUMERIC(4, 2) DEFAULT 6.90,
    primary_role VARCHAR(30) DEFAULT 'ALL',
    favorite_champions JSONB DEFAULT '[]'::jsonb,
    hidden_elo NUMERIC(8, 2) DEFAULT 1200.00,
    form_score NUMERIC(4, 2) DEFAULT 5.00,
    form_status VARCHAR(30) DEFAULT 'neutral',
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Index tìm kiếm nhanh theo nickname và OVR
CREATE INDEX IF NOT EXISTS idx_players_nickname ON public.players (nickname);
CREATE INDEX IF NOT EXISTS idx_players_stats_ovr ON public.players (stats_ovr DESC);
CREATE INDEX IF NOT EXISTS idx_players_hidden_elo ON public.players (hidden_elo DESC);

-- 2. BẢNG TRẬN ĐẤU (MATCHES)
CREATE TABLE IF NOT EXISTS public.matches (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    match_code VARCHAR(64),
    team1_players JSONB NOT NULL,
    team2_players JSONB NOT NULL,
    team1_power NUMERIC(6, 2) DEFAULT 0.00,
    team2_power NUMERIC(6, 2) DEFAULT 0.00,
    winner VARCHAR(10) NOT NULL CHECK (winner IN ('team1', 'team2')),
    result_code INT NOT NULL CHECK (result_code IN (1, 2)),
    synergies_applied JSONB DEFAULT '{}'::jsonb,
    notes TEXT,
    team1_kills INT DEFAULT 0,
    team2_kills INT DEFAULT 0,
    match_closeness NUMERIC(4, 2) DEFAULT 0.50,
    is_stomp BOOLEAN DEFAULT FALSE,
    balance_rating VARCHAR(20) DEFAULT 'unknown',
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Index sắp xếp lịch sử trận theo thời gian
CREATE INDEX IF NOT EXISTS idx_matches_created_at ON public.matches (created_at ASC);

-- 3. BẢNG CHI TIẾT TUYỂN THỦ THAM GIA TRẬN ĐẤU (MATCH_PARTICIPANTS)
CREATE TABLE IF NOT EXISTS public.match_participants (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    match_id BIGINT NOT NULL REFERENCES public.matches(id) ON DELETE CASCADE,
    player_id VARCHAR(64) NOT NULL REFERENCES public.players(id) ON DELETE CASCADE,
    team INT NOT NULL CHECK (team IN (1, 2)),
    is_winner BOOLEAN NOT NULL,
    champion VARCHAR(50),
    role VARCHAR(30),
    kills INT DEFAULT 0,
    deaths INT DEFAULT 0,
    assists INT DEFAULT 0,
    damage INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_participants_match_id ON public.match_participants (match_id);
CREATE INDEX IF NOT EXISTS idx_participants_player_id ON public.match_participants (player_id);
CREATE INDEX IF NOT EXISTS idx_participants_player_winner ON public.match_participants (player_id, is_winner);

-- 4. TỰ ĐỘNG CẬP NHẬT updated_at TRÊN BẢNG PLAYERS
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc'::text, NOW());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_players_updated_at ON public.players;
CREATE TRIGGER tr_players_updated_at
    BEFORE UPDATE ON public.players
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();

-- 5. PHÂN QUYỀN ROW LEVEL SECURITY (RLS)
-- Cho phép ứng dụng đọc và ghi thông qua anon key / service role key
ALTER TABLE public.players ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.match_participants ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read access to players" ON public.players FOR SELECT USING (true);
CREATE POLICY "Allow service insert access to players" ON public.players FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow service update access to players" ON public.players FOR UPDATE USING (true);
CREATE POLICY "Allow service delete access to players" ON public.players FOR DELETE USING (true);

CREATE POLICY "Allow public read access to matches" ON public.matches FOR SELECT USING (true);
CREATE POLICY "Allow service insert access to matches" ON public.matches FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow service update access to matches" ON public.matches FOR UPDATE USING (true);
CREATE POLICY "Allow service delete access to matches" ON public.matches FOR DELETE USING (true);

CREATE POLICY "Allow public read access to match_participants" ON public.match_participants FOR SELECT USING (true);
CREATE POLICY "Allow service insert access to match_participants" ON public.match_participants FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow service update access to match_participants" ON public.match_participants FOR UPDATE USING (true);
CREATE POLICY "Allow service delete access to match_participants" ON public.match_participants FOR DELETE USING (true);
