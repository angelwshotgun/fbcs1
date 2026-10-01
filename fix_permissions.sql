-- ==========================================================
-- SỬA LỖI HTTP 403 (42501 permission denied for schema public)
-- Copy và dán vào Supabase Dashboard > SQL Editor > Bấm RUN
-- ==========================================================

-- 1. CẤP QUYỀN TRUY CẬP SCHEMA PUBLIC CHO CÁC ROLE SUPABASE
GRANT USAGE ON SCHEMA public TO postgres, anon, authenticated, service_role;
GRANT ALL ON SCHEMA public TO postgres, anon, authenticated, service_role;

-- 2. CẤP QUYỀN TRÊN TOÀN BỘ CÁC BẢNG HIỆN CÓ
GRANT ALL ON ALL TABLES IN SCHEMA public TO postgres, anon, authenticated, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO postgres, anon, authenticated, service_role;
GRANT ALL ON ALL ROUTINES IN SCHEMA public TO postgres, anon, authenticated, service_role;

-- 3. THIẾT LẬP QUYỀN MẶC ĐỊNH CHO CÁC BẢNG ĐƯỢC TẠO SAU NÀY
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON ROUTINES TO postgres, anon, authenticated, service_role;

-- 4. ĐẢM BẢO ROW LEVEL SECURITY (RLS) MỞ TOÀN QUYỀN CHO APP
ALTER TABLE IF EXISTS public.players ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.match_participants ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow all for players" ON public.players;
DROP POLICY IF EXISTS "Allow all for matches" ON public.matches;
DROP POLICY IF EXISTS "Allow all for match_participants" ON public.match_participants;
DROP POLICY IF EXISTS "Allow public read access to players" ON public.players;
DROP POLICY IF EXISTS "Allow service insert access to players" ON public.players;
DROP POLICY IF EXISTS "Allow service update access to players" ON public.players;
DROP POLICY IF EXISTS "Allow service delete access to players" ON public.players;
DROP POLICY IF EXISTS "Allow public read access to matches" ON public.matches;
DROP POLICY IF EXISTS "Allow service insert access to matches" ON public.matches;
DROP POLICY IF EXISTS "Allow service update access to matches" ON public.matches;
DROP POLICY IF EXISTS "Allow service delete access to matches" ON public.matches;
DROP POLICY IF EXISTS "Allow public read access to match_participants" ON public.match_participants;
DROP POLICY IF EXISTS "Allow service insert access to match_participants" ON public.match_participants;
DROP POLICY IF EXISTS "Allow service update access to match_participants" ON public.match_participants;
DROP POLICY IF EXISTS "Allow service delete access to match_participants" ON public.match_participants;

CREATE POLICY "Allow all for players" ON public.players
FOR ALL TO anon, authenticated, service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all for matches" ON public.matches
FOR ALL TO anon, authenticated, service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all for match_participants" ON public.match_participants
FOR ALL TO anon, authenticated, service_role
USING (true)
WITH CHECK (true);
