import math
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class EloService:
    def __init__(self, base_elo: float = 1200.0, k_factor: float = 32.0):
        self.base_elo = base_elo
        self.k_factor = k_factor
    def calc_match_closeness(self, team1_kills: int, team2_kills: int) -> Tuple[float, str, bool]:
        """
        Tính mức độ cân bằng của trận đấu từ tỉ số hạ gục.
        Returns: (closeness, balance_rating, is_stomp)
        - closeness: 0.0 (stomp hoàn toàn) → 1.0 (sát nút)
        - balance_rating: 'perfect' | 'fair' | 'unbalanced' | 'stomp' | 'unknown'
        - is_stomp: True nếu trận lệch hẳn / stomp
        """
        total = team1_kills + team2_kills
        if total == 0:
            return 0.5, 'unknown', False

        max_k = max(team1_kills, team2_kills)
        min_k = min(team1_kills, team2_kills)
        diff = max_k - min_k
        ratio = max_k / max(1, min_k)

        # Tỉ lệ đóng góp kill của đội thắng (50% = cân bằng hoàn toàn, 100% = 1 đội ăn hết)
        kill_share = max_k / total
        # Closeness phi tuyến: 50% share -> 1.0; 60% share (1.5x) -> 0.65; 63% share (1.7x, 44-26) -> 0.55; 70% share (2.3x) -> 0.30
        closeness = round(max(0.0, min(1.0, 1.0 - (kill_share - 0.5) * 3.5)), 2)

        # 1. Trận Siêu Cân Bằng / Sát Nút (Perfect)
        if diff <= 5 and ratio <= 1.25:
            return closeness, 'perfect', False

        # 2. Trận Khá Cân Bằng / Giằng co (Fair)
        elif (diff <= 10 and ratio <= 1.45) or diff <= 6:
            return closeness, 'fair', False

        # 3. Trận Hủy Diệt / Áp Đảo Hoàn Toàn (Stomp)
        elif diff > 18 or (ratio >= 1.85 and diff >= 12) or diff >= 24:
            return closeness, 'stomp', True

        # 4. Trận Lệch Kèo / Mất Cân Bằng (Unbalanced, ví dụ 44 - 26 có diff=18, ratio=1.69)
        else:
            return closeness, 'unbalanced', False

    def _calc_adaptive_k(self, closeness: float, balance_rating: str = 'fair', base_k: float = 32.0) -> float:
        """
        K-factor thích ứng theo mức độ cân bằng thực tế của trận đấu:
        - Trận sát nút ('perfect', closeness >= 0.85): K = 24.0 (Elo ổn định, chia chuẩn)
        - Trận cân bằng ('fair', 0.60 <= closeness < 0.85): K = 28.8
        - Trận lệch kèo ('unbalanced', ví dụ 44-26, closeness 0.40 - 0.60): K = 40.0 (cần kéo dãn Elo nhanh)
        - Trận stomp ('stomp', closeness < 0.40): K = 46.0 - 50.0 (chia sai nhiều, cần sửa dứt khoát)
        """
        if balance_rating == 'perfect' or closeness >= 0.85:
            return round(base_k * 0.75, 1)  # ~24.0
        elif balance_rating == 'fair' or closeness >= 0.65:
            return round(base_k * 0.90, 1)  # ~28.8
        elif balance_rating == 'unbalanced':
            return round(base_k * 1.25, 1)  # ~40.0
        elif balance_rating == 'stomp' or closeness < 0.35:
            stomp_mult = 1.30 + (0.35 - min(0.35, closeness)) * 1.0
            return round(base_k * min(1.56, stomp_mult), 1)  # ~45.0 - 50.0
        else:
            return base_k  # 32.0

    def calculate_all_metrics(
        self,
        df: pd.DataFrame,
        match_details: Optional[Dict[str, Any]] = None,
        players_profiles: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Duyệt qua toàn bộ lịch sử trận đấu để tính:
        1. Elo ẩn (Hidden Elo / MMR, hỗ trợ delta cá nhân hóa AI và điều chỉnh tỷ lệ đóng góp carry vs hưởng ké)
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
                'trio_synergy': {},
                'elo_normalized': {},
                'match_closeness_history': [],
                'h2h_matrix': {}
            }

        player_cols = [c for c in df.columns if c != 'Result']

        # Khởi tạo Elo và danh sách lịch sử trận
        elo: Dict[str, float] = {p: self.base_elo for p in player_cols}
        match_history: Dict[str, List[str]] = {p: [] for p in player_cols}  # 'W' hoặc 'L'
        stats: Dict[str, Dict[str, int]] = {
            p: {'matches': 0, 'wins': 0, 'losses': 0} for p in player_cols
        }

        pair_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
        trio_stats: Dict[Tuple[str, str, str], Dict[str, int]] = {}
        h2h_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Track closeness history for balance report
        match_closeness_history = []

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

            # Kiểm tra xem trận đấu này có lưu delta tùy chỉnh / phân tích cá nhân hóa không
            meta = None
            if match_details:
                meta = match_details.get(str(idx)) or match_details.get(idx)
            custom_deltas = meta.get('player_deltas', {}) if meta else {}

            # Lấy tỉ số hạ gục nếu có
            t1_kills = 0
            t2_kills = 0
            if meta:
                t1_kills = int(meta.get('team1_kills', 0) or 0)
                t2_kills = int(meta.get('team2_kills', 0) or 0)

            # Tính closeness và adaptive K
            if t1_kills > 0 or t2_kills > 0:
                closeness, b_rating, is_stomp = self.calc_match_closeness(t1_kills, t2_kills)
                effective_k = self._calc_adaptive_k(closeness, balance_rating=b_rating, base_k=self.k_factor)
                match_closeness_history.append({
                    'match_idx': idx,
                    'closeness': closeness,
                    'rating': b_rating,
                    'is_stomp': is_stomp,
                    't1_kills': t1_kills,
                    't2_kills': t2_kills,
                    'effective_k': effective_k
                })
            else:
                closeness = 0.5  # Không có dữ liệu kill → dùng K mặc định
                b_rating = 'unknown'
                is_stomp = False
                effective_k = self.k_factor

            delta_1 = effective_k * (actual_1 - expected_1)
            delta_2 = effective_k * (actual_2 - expected_2)

            # Ước tính sức mạnh tương quan để điều chỉnh Elo cho trường hợp hưởng ké (không có ảnh AI)
            def calc_contrib_power(pid: str) -> float:
                p_elo = elo.get(pid, self.base_elo)
                prof = players_profiles.get(pid.lower(), {}) if players_profiles else {}
                p_ovr = float(prof.get('stats_ovr', prof.get('skill', 5.0)))
                return 0.6 * (p_elo / self.base_elo) + 0.4 * (p_ovr / 5.5)

            t1_powers = {p: calc_contrib_power(p) for p in team1}
            t2_powers = {p: calc_contrib_power(p) for p in team2}
            avg_power_1 = sum(t1_powers.values()) / max(1, len(team1))
            avg_power_2 = sum(t2_powers.values()) / max(1, len(team2))

            # Cập nhật Elo cho Team 1
            for p in team1:
                if p in custom_deltas and custom_deltas[p] is not None:
                    p_delta = float(custom_deltas[p])
                else:
                    ratio = t1_powers[p] / max(0.1, avg_power_1)
                    if result == 1:
                        # Thắng: Tuyển thủ có power thấp hơn nhiều đồng đội (hưởng ké) bị giảm điểm thưởng
                        # Tuyển thủ dẫn dắt (power cao) được thưởng thêm
                        contrib_mult = max(0.40, min(1.40, ratio ** 0.85))
                        p_delta = delta_1 * contrib_mult
                    else:
                        # Thua: Tuyển thủ gánh đội được giảm trừ, tuyển thủ kéo đội xuống bị trừ đủ/nhiều hơn
                        loss_mult = max(0.65, min(1.35, (1.0 / max(0.1, ratio)) ** 0.5))
                        p_delta = delta_1 * loss_mult

                elo[p] = round(elo[p] + p_delta, 2)
                stats[p]['matches'] += 1
                if result == 1:
                    stats[p]['wins'] += 1
                    match_history[p].append('W')
                else:
                    stats[p]['losses'] += 1
                    match_history[p].append('L')

            # Cập nhật Elo cho Team 2
            for p in team2:
                if p in custom_deltas and custom_deltas[p] is not None:
                    p_delta = float(custom_deltas[p])
                else:
                    ratio = t2_powers[p] / max(0.1, avg_power_2)
                    if result == 2:
                        contrib_mult = max(0.40, min(1.40, ratio ** 0.85))
                        p_delta = delta_2 * contrib_mult
                    else:
                        loss_mult = max(0.65, min(1.35, (1.0 / max(0.1, ratio)) ** 0.5))
                        p_delta = delta_2 * loss_mult

                elo[p] = round(elo[p] + p_delta, 2)
                stats[p]['matches'] += 1
                if result == 2:
                    stats[p]['wins'] += 1
                    match_history[p].append('W')
                else:
                    stats[p]['losses'] += 1
                    match_history[p].append('L')

            # Ghi nhận cặp đồng đội (Duo - Team 1)
            for p1, p2 in combinations(sorted(team1), 2):
                pair = (p1, p2)
                if pair not in pair_stats:
                    pair_stats[pair] = {'together_matches': 0, 'together_wins': 0}
                pair_stats[pair]['together_matches'] += 1
                if result == 1:
                    pair_stats[pair]['together_wins'] += 1

            # Ghi nhận bộ ba đồng đội (Trio - Team 1)
            for p1, p2, p3 in combinations(sorted(team1), 3):
                trio = (p1, p2, p3)
                if trio not in trio_stats:
                    trio_stats[trio] = {'together_matches': 0, 'together_wins': 0}
                trio_stats[trio]['together_matches'] += 1
                if result == 1:
                    trio_stats[trio]['together_wins'] += 1

            # Ghi nhận cặp đồng đội (Duo - Team 2)
            for p1, p2 in combinations(sorted(team2), 2):
                pair = (p1, p2)
                if pair not in pair_stats:
                    pair_stats[pair] = {'together_matches': 0, 'together_wins': 0}
                pair_stats[pair]['together_matches'] += 1
                if result == 2:
                    pair_stats[pair]['together_wins'] += 1

            # Ghi nhận bộ ba đồng đội (Trio - Team 2)
            for p1, p2, p3 in combinations(sorted(team2), 3):
                trio = (p1, p2, p3)
                if trio not in trio_stats:
                    trio_stats[trio] = {'together_matches': 0, 'together_wins': 0}
                trio_stats[trio]['together_matches'] += 1
                if result == 2:
                    trio_stats[trio]['together_wins'] += 1

            # Ghi nhận đối kháng cá nhân (Head-to-Head - H2H) giữa từng người chơi ở 2 phe
            kill_diff = (t1_kills - t2_kills) if (t1_kills > 0 or t2_kills > 0) else 0
            for p1 in team1:
                for p2 in team2:
                    k1 = (p1, p2)
                    k2 = (p2, p1)
                    if k1 not in h2h_stats:
                        h2h_stats[k1] = {
                            'matches': 0, 'wins': 0, 'losses': 0,
                            'kill_diff_sum': 0, 'stomp_wins': 0, 'stomp_losses': 0,
                            'unbalanced_wins': 0, 'unbalanced_losses': 0
                        }
                    if k2 not in h2h_stats:
                        h2h_stats[k2] = {
                            'matches': 0, 'wins': 0, 'losses': 0,
                            'kill_diff_sum': 0, 'stomp_wins': 0, 'stomp_losses': 0,
                            'unbalanced_wins': 0, 'unbalanced_losses': 0
                        }
                    h2h_stats[k1]['matches'] += 1
                    h2h_stats[k2]['matches'] += 1
                    h2h_stats[k1]['kill_diff_sum'] += kill_diff
                    h2h_stats[k2]['kill_diff_sum'] -= kill_diff
                    if result == 1:
                        h2h_stats[k1]['wins'] += 1
                        h2h_stats[k2]['losses'] += 1
                        if b_rating == 'stomp' or is_stomp:
                            h2h_stats[k1]['stomp_wins'] += 1
                            h2h_stats[k2]['stomp_losses'] += 1
                        elif b_rating == 'unbalanced':
                            h2h_stats[k1]['unbalanced_wins'] += 1
                            h2h_stats[k2]['unbalanced_losses'] += 1
                    elif result == 2:
                        h2h_stats[k2]['wins'] += 1
                        h2h_stats[k1]['losses'] += 1
                        if b_rating == 'stomp' or is_stomp:
                            h2h_stats[k2]['stomp_wins'] += 1
                            h2h_stats[k1]['stomp_losses'] += 1
                        elif b_rating == 'unbalanced':
                            h2h_stats[k2]['unbalanced_wins'] += 1
                            h2h_stats[k1]['unbalanced_losses'] += 1

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

        # Chuẩn hóa Pair Synergy (Duo)
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

        # Chuẩn hóa Trio Synergy (Bộ ba tam tấu - 3 người cùng team)
        trio_synergy: Dict[str, Dict[str, Any]] = {}
        for (p1, p2, p3), t_data in trio_stats.items():
            tm = t_data['together_matches']
            tw = t_data['together_wins']
            if tm >= 2:
                wr = tw / tm
                # Synergy bonus từ -0.6 đến +0.6
                synergy_score = round((wr - 0.5) * 2.4, 3)
                key = f"{p1}|{p2}|{p3}"
                trio_synergy[key] = {
                    'p1': p1,
                    'p2': p2,
                    'p3': p3,
                    'matches': tm,
                    'wins': tw,
                    'winrate': round(wr * 100, 1),
                    'synergy_score': synergy_score
                }

        # Chuẩn hóa ma trận đối đầu trực tiếp (H2H Matrix)
        h2h_matrix: Dict[str, Dict[str, Any]] = {}
        for (p1, p2), h_data in h2h_stats.items():
            key = f"{p1}|{p2}"
            h2h_matrix[key] = {
                'p1': p1,
                'p2': p2,
                **h_data
            }

        return {
            'elo': elo,
            'elo_normalized': elo_normalized,
            'form': form_data,
            'stats': stats,
            'pair_synergy': pair_synergy,
            'trio_synergy': trio_synergy,
            'match_closeness_history': match_closeness_history,
            'h2h_matrix': h2h_matrix
        }

    def evaluate_h2h_matchup(
        self,
        team1: Any,
        team2: Any,
        h2h_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Đánh giá lịch sử đối đầu thực tế giữa 2 đội bất kỳ (dựa trên 25 cặp đối đầu cá nhân).
        Áp dụng linh hoạt cho MỌI tập hợp tuyển thủ (không đòi hỏi 10 người phải giống hệt nhau).
        """
        total_encounters = 0
        t1_wins = 0
        t2_wins = 0
        t1_kill_diff_sum = 0
        t1_stomp_wins = 0
        t2_stomp_wins = 0
        t1_unbalanced_wins = 0
        t2_unbalanced_wins = 0
        rivalries = []

        for p1 in team1:
            for p2 in team2:
                key = f"{str(p1).lower()}|{str(p2).lower()}"
                rec = h2h_map.get(key)
                if rec and rec.get('matches', 0) > 0:
                    m_cnt = rec['matches']
                    w_cnt = rec['wins']
                    l_cnt = rec['losses']
                    kd = rec.get('kill_diff_sum', 0)
                    total_encounters += m_cnt
                    t1_wins += w_cnt
                    t2_wins += l_cnt
                    t1_kill_diff_sum += kd
                    t1_stomp_wins += rec.get('stomp_wins', 0)
                    t2_stomp_wins += rec.get('stomp_losses', 0)
                    t1_unbalanced_wins += rec.get('unbalanced_wins', 0)
                    t2_unbalanced_wins += rec.get('unbalanced_losses', 0)

                    # Lưu các cặp kình địch đáng chú ý (áp đảo hoặc va chạm nhiều)
                    if m_cnt >= 3 and abs(w_cnt - l_cnt) >= 2:
                        rivalries.append({
                            'p1': str(p1),
                            'p2': str(p2),
                            'matches': m_cnt,
                            'w1': w_cnt,
                            'w2': l_cnt,
                            'lead': str(p1) if w_cnt > l_cnt else str(p2),
                            'diff': abs(w_cnt - l_cnt),
                            'avg_kill_diff': round(kd / max(1, m_cnt), 1)
                        })

        if total_encounters == 0:
            return {
                'has_history': False,
                'total_encounters': 0,
                't1_winrate': 50.0,
                't2_winrate': 50.0,
                'avg_kill_diff': 0.0,
                'h2h_bias_elo': 0.0,
                'summary': 'Chưa có dữ liệu đối đầu giữa hai bên.',
                'rivalries': []
            }

        t1_winrate = round((t1_wins / total_encounters) * 100, 1)
        t2_winrate = round(100.0 - t1_winrate, 1)
        avg_kd = round(t1_kill_diff_sum / total_encounters, 1)
        stomp_diff = t1_stomp_wins - t2_stomp_wins
        unbal_diff = t1_unbalanced_wins - t2_unbalanced_wins

        # Quy đổi độ lệch thực chiến ra điểm Elo:
        # - Chênh tỉ lệ thắng đối đầu (+/- 25 Elo)
        # - Chênh lệch mạng trung bình (+/- 25 Elo)
        # - Chênh lệch số trận stomp/unbalanced (+/- 40 Elo)
        h2h_bias_elo = round(
            (t1_winrate - 50.0) * 0.75 +
            avg_kd * 2.0 +
            stomp_diff * 1.5 +
            unbal_diff * 0.8,
            1
        )
        h2h_bias_elo = max(-90.0, min(90.0, h2h_bias_elo))

        # Sắp xếp các kình địch nổi bật nhất
        rivalries.sort(key=lambda x: (x['matches'], x['diff']), reverse=True)

        # Xây dựng câu tóm tắt nhận xét
        if abs(h2h_bias_elo) <= 15.0:
            summary_status = 'Rất cân bằng theo lịch sử đối đầu'
        elif h2h_bias_elo > 15.0:
            summary_status = f'Đội 1 có lợi thế đối đầu lịch sử (+{h2h_bias_elo} Elo)'
        else:
            summary_status = f'Đội 2 có lợi thế đối đầu lịch sử (+{abs(h2h_bias_elo)} Elo)'

        summary = f"{summary_status} qua {total_encounters} lượt chạm trán (Tỉ lệ thắng: {t1_winrate}% - {t2_winrate}%)."

        return {
            'has_history': True,
            'total_encounters': total_encounters,
            't1_wins': t1_wins,
            't2_wins': t2_wins,
            't1_winrate': t1_winrate,
            't2_winrate': t2_winrate,
            'avg_kill_diff': avg_kd,
            'stomp_diff': stomp_diff,
            'unbal_diff': unbal_diff,
            'h2h_bias_elo': h2h_bias_elo,
            'summary': summary,
            'rivalries': rivalries[:4]
        }


elo_service = EloService()

