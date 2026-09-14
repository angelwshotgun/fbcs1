import math
from itertools import combinations
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


class EloService:
    def __init__(self, base_elo: float = 1200.0, k_factor: float = 32.0):
        self.base_elo = base_elo
        self.k_factor = k_factor

    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Duyệt qua toàn bộ lịch sử trận đấu để tính:
        1. Elo ẩn (Hidden Elo / MMR)
        2. Phong độ tự động (Form, Streak, Form Score 1-10)
        3. Thống kê trận (Wins, Losses, Matches, Winrate)
        4. Cặp bài trùng (Pair Synergy)
        """
        if df.empty or 'Result' not in df.columns:
            return {
                'elo': {},
                'form': {},
                'stats': {},
                'pair_synergy': {},
                'elo_normalized': {}
            }

        player_cols = [c for c in df.columns if c != 'Result']

        # Khởi tạo Elo và danh sách lịch sử trận
        elo: Dict[str, float] = {p: self.base_elo for p in player_cols}
        match_history: Dict[str, List[str]] = {p: [] for p in player_cols}  # 'W' hoặc 'L'
        stats: Dict[str, Dict[str, int]] = {
            p: {'matches': 0, 'wins': 0, 'losses': 0} for p in player_cols
        }

        pair_stats: Dict[Tuple[str, str], Dict[str, int]] = {}

        # Duyệt qua từng trận đấu theo thứ tự từ đầu đến cuối
        for idx, row in df.iterrows():
            result = row['Result']
            if result not in [1, 2]:
                continue

            team1 = [p for p in player_cols if row[p] == 1]
            team2 = [p for p in player_cols if row[p] == 2]

            if not team1 or not team2:
                continue

            # Tính Elo trung bình 2 đội tại thời điểm trận đấu diễn ra
            avg_elo_1 = sum(elo[p] for p in team1) / len(team1)
            avg_elo_2 = sum(elo[p] for p in team2) / len(team2)

            # Xác suất thắng kỳ vọng theo công thức Logistic Elo
            expected_1 = 1.0 / (1.0 + 10.0 ** ((avg_elo_2 - avg_elo_1) / 400.0))
            expected_2 = 1.0 - expected_1

            actual_1 = 1.0 if result == 1 else 0.0
            actual_2 = 1.0 - actual_1

            delta_1 = self.k_factor * (actual_1 - expected_1)
            delta_2 = self.k_factor * (actual_2 - expected_2)

            # Cập nhật Elo cho Team 1
            for p in team1:
                elo[p] = round(elo[p] + delta_1, 2)
                stats[p]['matches'] += 1
                if result == 1:
                    stats[p]['wins'] += 1
                    match_history[p].append('W')
                else:
                    stats[p]['losses'] += 1
                    match_history[p].append('L')

            # Cập nhật Elo cho Team 2
            for p in team2:
                elo[p] = round(elo[p] + delta_2, 2)
                stats[p]['matches'] += 1
                if result == 2:
                    stats[p]['wins'] += 1
                    match_history[p].append('W')
                else:
                    stats[p]['losses'] += 1
                    match_history[p].append('L')

            # Ghi nhận cặp đồng đội (Team 1)
            for p1, p2 in combinations(sorted(team1), 2):
                pair = (p1, p2)
                if pair not in pair_stats:
                    pair_stats[pair] = {'together_matches': 0, 'together_wins': 0}
                pair_stats[pair]['together_matches'] += 1
                if result == 1:
                    pair_stats[pair]['together_wins'] += 1

            # Ghi nhận cặp đồng đội (Team 2)
            for p1, p2 in combinations(sorted(team2), 2):
                pair = (p1, p2)
                if pair not in pair_stats:
                    pair_stats[pair] = {'together_matches': 0, 'together_wins': 0}
                pair_stats[pair]['together_matches'] += 1
                if result == 2:
                    pair_stats[pair]['together_wins'] += 1

        # ================================
        # TÍNH TOÁN PHONG ĐỘ TỰ ĐỘNG (FORM)
        # ================================
        form_data: Dict[str, Dict[str, Any]] = {}
        for p in player_cols:
            hist = match_history[p]
            recent_matches = hist[-5:] if len(hist) >= 5 else hist
            recent_count = len(recent_matches)

            if recent_count == 0:
                form_data[p] = {
                    'score': 5.0,
                    'multiplier': 1.0,
                    'status': 'neutral',
                    'icon': '🌱',
                    'label': 'Tân binh (Chưa có trận)',
                    'streak': 'N/A',
                    'recent_5': [],
                    'recent_winrate': 0.0
                }
                continue

            recent_wins = recent_matches.count('W')
            recent_losses = recent_matches.count('L')
            recent_winrate = round((recent_wins / recent_count) * 100, 1)

            # Tính Streak hiện tại (W3, L2...)
            current_streak_type = hist[-1]
            streak_count = 0
            for r in reversed(hist):
                if r == current_streak_type:
                    streak_count += 1
                else:
                    break
            streak_str = f"{current_streak_type}{streak_count}"

            # Tính điểm phong độ 1-10
            # Base 5.5
            base_score = 5.5
            diff = recent_wins - recent_losses
            form_score = base_score + (diff * 0.8)

            # Thưởng / phạt thêm dựa trên streak
            if current_streak_type == 'W':
                if streak_count >= 3:
                    form_score += 1.2
                elif streak_count >= 5:
                    form_score += 2.0
            else:
                if streak_count >= 3:
                    form_score -= 1.2
                elif streak_count >= 5:
                    form_score -= 2.0

            form_score = max(1.0, min(10.0, round(form_score, 1)))

            # Phân loại trạng thái & hệ số sức mạnh (Multiplier)
            if form_score >= 8.5 or (current_streak_type == 'W' and streak_count >= 3):
                status = 'on_fire'
                icon = '🔥'
                label = 'Thần phong (On Fire)'
                multiplier = 1.12
            elif form_score >= 6.5:
                status = 'good'
                icon = '⚡'
                label = 'Phong độ cao'
                multiplier = 1.05
            elif form_score >= 4.5:
                status = 'stable'
                icon = '⚖️'
                label = 'Bình ổn'
                multiplier = 1.00
            elif form_score >= 2.5:
                status = 'cold'
                icon = '❄️'
                label = 'Xuống phong độ'
                multiplier = 0.95
            else:
                status = 'slump'
                icon = '💀'
                label = 'Chạm đáy'
                multiplier = 0.88

            form_data[p] = {
                'score': form_score,
                'multiplier': multiplier,
                'status': status,
                'icon': icon,
                'label': label,
                'streak': streak_str,
                'recent_5': recent_matches,
                'recent_winrate': recent_winrate
            }

        # ================================
        # CHUẨN HÓA ELO ẨN SANG THANG 1-10
        # ================================
        elo_values = list(elo.values())
        min_elo = min(elo_values) if elo_values else self.base_elo
        max_elo = max(elo_values) if elo_values else self.base_elo
        elo_range = max(1.0, max_elo - min_elo)

        elo_normalized: Dict[str, float] = {}
        for p, rating in elo.items():
            # Chuẩn hóa về dải 2.0 - 9.5 để tránh cực đoan
            norm = 2.0 + 7.5 * ((rating - min_elo) / elo_range)
            elo_normalized[p] = round(max(1.0, min(10.0, norm)), 1)

        # Tính winrate tổng quan cho từng người
        for p in player_cols:
            m = stats[p]['matches']
            w = stats[p]['wins']
            stats[p]['winrate'] = round((w / m * 100), 1) if m > 0 else 0.0

        # Chuẩn hóa Pair Synergy
        pair_synergy: Dict[str, Dict[str, Any]] = {}
        for (p1, p2), p_data in pair_stats.items():
            tm = p_data['together_matches']
            tw = p_data['together_wins']
            if tm >= 2:
                wr = tw / tm
                # Synergy bonus từ -0.5 đến +0.5
                synergy_score = round((wr - 0.5) * 2.0, 3)
                key = f"{p1}|{p2}"
                pair_synergy[key] = {
                    'p1': p1,
                    'p2': p2,
                    'matches': tm,
                    'wins': tw,
                    'winrate': round(wr * 100, 1),
                    'synergy_score': synergy_score
                }

        return {
            'elo': elo,
            'elo_normalized': elo_normalized,
            'form': form_data,
            'stats': stats,
            'pair_synergy': pair_synergy
        }


elo_service = EloService()
