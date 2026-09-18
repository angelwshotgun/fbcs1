-- ==========================================================
-- FBCS ESPORTS - DATABASE MIGRATION: KILL SCORE & MATCH STATS
-- Script di chuyển (migration) bổ sung các cột lưu trữ thông số điểm hạ gục,
-- độ giằng co trận đấu (match closeness), thế trận một chiều (stomp),
-- đánh giá cân bằng (balance rating) và chỉ số KDA, sát thương của từng tuyển thủ.
-- Chạy script này trong Supabase Dashboard > SQL Editor
-- ==========================================================

-- 1. Bổ sung các cột thống kê điểm hạ gục và đánh giá cân bằng vào bảng matches
ALTER TABLE IF EXISTS public.matches
    ADD COLUMN IF NOT EXISTS team1_kills INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS team2_kills INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS match_closeness NUMERIC(4, 2) DEFAULT 0.50,
    ADD COLUMN IF NOT EXISTS is_stomp BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS balance_rating VARCHAR(20) DEFAULT 'unknown';

-- 2. Bổ sung các cột KDA và sát thương chi tiết vào bảng match_participants
ALTER TABLE IF EXISTS public.match_participants
    ADD COLUMN IF NOT EXISTS kills INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS deaths INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS assists INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS damage INT DEFAULT 0;
