"""
Hệ thống Đánh Giá Năng Lực & Chia Đội Đối Xứng (RAPM + SVD Rating Engine)
- Thay thế hoàn toàn cơ chế Elo truyền thống, loại bỏ điểm kỹ năng cảm tính (1-10) và vị trí cố định.
- Tính toán 100% khách quan từ dữ liệu kết quả thắng/thua lịch sử bằng Hồi quy Ridge (RAPM - Regularized Adjusted Plus-Minus).
- Phân bậc rõ ràng: Tier S (>=80), Tier A (65-79.9), Tier B (50-64.9), Tier C (<50).
- Chia đội theo thuật toán SVD (Symmetric Value Draft): Phân bổ đối xứng gánh kèo & lót đường, cân bằng điểm thực lực tối đa.
"""

import math
import random
from itertools import combinations
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd


class RatingService:
    def __init__(self, lambda_reg: float = 3.0, half_life: float = 20.0):
        self.lambda_reg = lambda_reg
        self.half_life = half_life
        self._cached_ratings: Optional[Dict[str, Any]] = None

    def invalidate_cache(self):
        """Xóa cache để hệ thống tính toán lại khi có trận mới hoặc cập nhật dữ liệu."""
        self._cached_ratings = None

    def calculate_ratings(self, df: pd.DataFrame, profiles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Tính toán Điểm Thực Lực (Power Score 0-100) và Phân Bậc (Tier S, A, B, C)
        hoàn toàn dựa trên ma trận Ridge Regression RAPM từ kết quả thắng/thua.
        """
        if profiles is None:
            profiles = {}

        profiles_lower = {str(k).lower(): v for k, v in profiles.items()}
        all_profile_ids = list(profiles_lower.keys())

        if df.empty or 'Result' not in df.columns:
            # Trường hợp chưa có dữ liệu trận đấu
            default_ratings = {}
            for pid in all_profile_ids:
                default_ratings[pid] = self._make_default_player_rating(pid, profiles_lower.get(pid, {}))
            return {
                'players': default_ratings,
                'pair_synergy': {},
                'trio_synergy': {},
                'max_power': 50.0,
                'min_power': 50.0
            }

        player_cols = [c for c in df.columns if c != 'Result']
        valid_df = df[df['Result'].isin([1, 2])].copy().reset_index(drop=True)
        N = len(valid_df)

        # Tính trọng số thời gian (Exponential Match Recency Decay) cho từng trận
        # Trận mới nhất (k = N - 1) có weight = 1.0, các trận cũ suy giảm dần theo chu kỳ bán rã half_life.
        # max(0.15, ...) đảm bảo các trận quá khứ vẫn giữ lại giá trị tham chiếu nền tối thiểu.
        if N > 0:
            weights = [max(0.15, (0.5) ** (((N - 1) - k) / self.half_life)) for k in range(N)]
        else:
            weights = []

        # Thống kê số trận, số trận hiệu dụng, số trận thắng, chuỗi trận gần đây
        stats: Dict[str, Dict[str, Any]] = {}
        for p in player_cols:
            p_lower = str(p).lower()
            p_matches = valid_df[valid_df[p].isin([1, 2])]
            m = len(p_matches)
            if m == 0:
                continue

            # Số trận hiệu dụng (tổng trọng số các trận đã tham gia sau khi phân rã theo thời gian)
            eff_m = float(sum(weights[idx] for idx in p_matches.index)) if weights else 0.0

            w1 = len(valid_df[(valid_df[p] == 1) & (valid_df['Result'] == 1)])
            w2 = len(valid_df[(valid_df[p] == 2) & (valid_df['Result'] == 2)])
            w = w1 + w2
            wr = round((w / m) * 100, 1)

            # Lấy 5 trận gần nhất theo thứ tự thời gian
            recent_5 = []
            for _, row in p_matches.tail(5).iterrows():
                side = row[p]
                res = row['Result']
                recent_5.append('W' if side == res else 'L')

            stats[p_lower] = {
                'original_col': p,
                'matches': m,
                'effective_matches': round(eff_m, 1),
                'wins': w,
                'losses': m - w,
                'winrate': wr,
                'recent_5': recent_5,
                'rapm': 0.0,
                'confidence': 1.0
            }

        active_players = list(stats.keys())

        # Nếu có ít nhất 1 trận đấu hợp lệ và có người chơi tham gia
        if N > 0 and len(active_players) > 0:
            X_rows, y_vals = [], []
            for _, row in valid_df.iterrows():
                res = row['Result']
                x_row = []
                for p_id in active_players:
                    orig_col = stats[p_id]['original_col']
                    val = row.get(orig_col, 0)
                    if val == 1:
                        x_row.append(1.0)
                    elif val == 2:
                        x_row.append(-1.0)
                    else:
                        x_row.append(0.0)
                X_rows.append(x_row)
                y_vals.append(1.0 if res == 1 else -1.0)

            X = np.array(X_rows, dtype=float)
            y = np.array(y_vals, dtype=float)

            # Weighted Ridge Regression:
            # W = diag(w_0, ..., w_{N-1})
            # A = X^T W X + lambda * I
            # b = X^T W y
            # beta = A^(-1) b
            # Các trận mới có trọng số cao hơn, phản ánh chính xác phong độ thực tế hiện tại.
            W = np.diag(weights)
            I = np.eye(X.shape[1])
            try:
                A = X.T @ W @ X + self.lambda_reg * I
                b = X.T @ W @ y
                beta = np.linalg.inv(A) @ b
            except Exception:
                beta = np.zeros(X.shape[1])

            for i, p_id in enumerate(active_players):
                stats[p_id]['rapm'] = float(beta[i])

            # Tính toán cực trị beta dương và âm để chuẩn hóa đối xứng qua mốc beta = 0.0 (50.0 điểm)
            max_pos_beta = max(max((s['rapm'] for s in stats.values()), default=0.0), 0.01)
            max_neg_beta = abs(min((s['rapm'] for s in stats.values()), default=0.0))
            if max_neg_beta < 0.01:
                max_neg_beta = 0.01
        else:
            max_pos_beta, max_neg_beta = 1.0, 1.0

        # Tính toán Pair Synergy (Duo) và Trio Synergy thuần kết quả
        pair_synergy: Dict[str, Dict[str, Any]] = {}
        trio_synergy: Dict[str, Dict[str, Any]] = {}

        for p1_idx in range(len(active_players)):
            p1 = active_players[p1_idx]
            orig1 = stats[p1]['original_col']
            for p2_idx in range(p1_idx + 1, len(active_players)):
                p2 = active_players[p2_idx]
                orig2 = stats[p2]['original_col']

                same_t1 = valid_df[(valid_df[orig1] == 1) & (valid_df[orig2] == 1)]
                same_t2 = valid_df[(valid_df[orig1] == 2) & (valid_df[orig2] == 2)]
                together_matches = len(same_t1) + len(same_t2)

                if together_matches >= 2:
                    w1 = len(same_t1[same_t1['Result'] == 1])
                    w2 = len(same_t2[same_t2['Result'] == 2])
                    together_wins = w1 + w2
                    wr = together_wins / together_matches
                    # Synergy bonus từ -2.0 đến +2.0 điểm
                    score = round((wr - 0.5) * 4.0, 2)
                    pair_key = f"{p1}|{p2}"
                    pair_synergy[pair_key] = {
                        'p1': p1,
                        'p2': p2,
                        'matches': together_matches,
                        'wins': together_wins,
                        'winrate': round(wr * 100, 1),
                        'synergy_score': score
                    }

        # Tạo object kết quả đầy đủ cho TẤT CẢ các tuyển thủ (kể cả chưa đấu)
        final_players: Dict[str, Any] = {}
        all_ids = set(active_players) | set(all_profile_ids)

        for pid in sorted(list(all_ids)):
            profile = profiles_lower.get(pid, {})
            nickname = profile.get('nickname') or pid.capitalize()
            avatar = profile.get('avatar') or f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}"

            if pid in stats:
                st = stats[pid]
                m = st['matches']
                eff_m = st['effective_matches']
                w = st['wins']
                l = st['losses']
                wr = st['winrate']
                rec5 = st['recent_5']
                rapm = st['rapm']

                # Chuẩn hóa đối xứng qua trục beta = 0.0 (50.0 điểm thực lực)
                if rapm >= 0:
                    raw_power = 50.0 + 50.0 * (rapm / max_pos_beta)
                else:
                    raw_power = 50.0 - 35.0 * (abs(rapm) / max_neg_beta)

                # Sample Size & Recency Reliability Shrinkage:
                # Tuyển thủ ít trận hoặc chỉ đánh trận từ rất lâu trong quá khứ sẽ có eff_m thấp -> conf thấp -> kéo về 50.0.
                # Tuyển thủ cày ải nhiều trận và phong độ gần đây tốt sẽ có conf tiệm cận 1.0 (phát huy 100% điểm thực lực).
                conf = min(1.0, (eff_m / (eff_m + 3.0)) * 1.25)
                power_score = round(50.0 + (raw_power - 50.0) * conf, 1)
                st['confidence'] = round(conf, 2)
            else:
                # Tân binh chưa có trận đấu
                m, w, l, wr = 0, 0, 0, 0.0
                eff_m = 0.0
                rec5 = []
                rapm = 0.0
                conf = 0.0
                power_score = 50.0

            # Phân Bậc Tier
            tier_info = self._get_tier_info(power_score, m)

            final_players[pid] = {
                'id': pid,
                'nickname': nickname,
                'avatar': avatar,
                'power_score': power_score,
                'rapm': round(rapm, 3),
                'confidence': round(conf, 2),
                'effective_matches': round(eff_m, 1),
                'tier': tier_info['tier'],
                'tier_name': tier_info['name'],
                'tier_icon': tier_info['icon'],
                'tier_stars': tier_info['stars'],
                'tier_badge_class': tier_info['badge_class'],
                'tier_desc': tier_info['desc'],
                'matches': m,
                'wins': w,
                'losses': l,
                'winrate': wr,
                'recent_5': rec5,
                # Giữ tương thích ngược với các hàm gọi elo cũ
                'hidden_elo': round(1000.0 + power_score * 4.0, 1),
                'effective_power': power_score,
                'combined_power': power_score
            }

        return {
            'players': final_players,
            'pair_synergy': pair_synergy,
            'trio_synergy': trio_synergy,
            'min_rapm': -round(max_neg_beta, 3),
            'max_rapm': round(max_pos_beta, 3),
            'max_pos_beta': round(max_pos_beta, 3),
            'max_neg_beta': round(max_neg_beta, 3)
        }

    def _get_tier_info(self, power_score: float, matches: int) -> Dict[str, Any]:
        """Quy định phân bậc 4 Tier tự động dựa trên Điểm Thực Lực."""
        if matches == 0:
            return {
                'tier': 'NEW',
                'name': 'Tân Binh',
                'icon': '🌱',
                'stars': '⭐',
                'badge_class': 'bg-slate-100 text-slate-700 border-slate-300 font-bold',
                'desc': 'Chưa có dữ liệu thi đấu'
            }
        elif power_score >= 80.0:
            return {
                'tier': 'S',
                'name': 'Tier S',
                'icon': '👑',
                'stars': '⭐⭐⭐⭐⭐',
                'badge_class': 'bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 text-slate-900 border-amber-400 font-black shadow-xs',
                'desc': 'Chủ lực gánh kèo đỉnh cao'
            }
        elif power_score >= 65.0:
            return {
                'tier': 'A',
                'name': 'Tier A',
                'icon': '⚔️',
                'stars': '⭐⭐⭐⭐',
                'badge_class': 'bg-purple-100 text-purple-800 border-purple-300 font-bold',
                'desc': 'Chiến tướng độc lập tác chiến'
            }
        elif power_score >= 50.0:
            return {
                'tier': 'B',
                'name': 'Tier B',
                'icon': '🛡️',
                'stars': '⭐⭐⭐',
                'badge_class': 'bg-blue-100 text-blue-800 border-blue-300 font-bold',
                'desc': 'Trụ cột tròn vai vững chắc'
            }
        else:
            return {
                'tier': 'C',
                'name': 'Tier C',
                'icon': '🌱',
                'stars': '⭐⭐',
                'badge_class': 'bg-slate-100 text-slate-600 border-slate-300 font-bold',
                'desc': 'Cần hỗ trợ và cải thiện'
            }

    def _make_default_player_rating(self, pid: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        nickname = profile.get('nickname') or pid.capitalize()
        avatar = profile.get('avatar') or f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}"
        tier_info = self._get_tier_info(50.0, 0)
        return {
            'id': pid,
            'nickname': nickname,
            'avatar': avatar,
            'power_score': 50.0,
            'rapm': 0.0,
            'confidence': 0.0,
            'effective_matches': 0.0,
            'tier': tier_info['tier'],
            'tier_name': tier_info['name'],
            'tier_icon': tier_info['icon'],
            'tier_stars': tier_info['stars'],
            'tier_badge_class': tier_info['badge_class'],
            'tier_desc': tier_info['desc'],
            'matches': 0,
            'wins': 0,
            'losses': 0,
            'winrate': 0.0,
            'recent_5': [],
            'hidden_elo': 1200.0,
            'effective_power': 50.0,
            'combined_power': 50.0
        }

    def get_ratings(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Lấy bảng rating kèm cache in-memory."""
        if self._cached_ratings is None or force_refresh:
            from services.data_manager import data_manager
            df = data_manager.read_matches_df()
            profiles = data_manager.read_players_data()
            self._cached_ratings = self.calculate_ratings(df, profiles)
        return self._cached_ratings

    def get_player(self, player_id: str) -> Optional[Dict[str, Any]]:
        ratings_data = self.get_ratings()
        pid = str(player_id).strip().lower()
        return ratings_data['players'].get(pid)

    def balance_teams_svd(
        self,
        player_ids: List[str],
        allow_rng: bool = True,
        rng_tolerance: float = 3.0
    ) -> Dict[str, Any]:
        """
        Thuật toán SVD (Symmetric Value Draft):
        - Đầu vào: 10 tuyển thủ.
        - Sắp xếp 10 người theo Điểm Thực Lực: P1, P2, ..., P10.
        - Điều kiện 1: Tuyệt đối không cho P1 và P2 (2 người gánh mạnh nhất) cùng đội.
        - Điều kiện 2: Tuyệt đối không cho P9 và P10 (2 người đuối nhất) cùng đội.
        - Điều kiện 3: Tối thiểu hóa độ chênh lệch tổng điểm thực lực |Team1 - Team2|.
        - Hỗ trợ RNG trong phạm vi sai số cực nhỏ để mỗi lần chia có thể xoay đổi linh hoạt.
        """
        if len(player_ids) != 10:
            raise ValueError("Cần chính xác 10 người chơi để chia đội")

        unique_players = sorted(list(set(str(p).lower() for p in player_ids)))
        if len(unique_players) != 10:
            raise ValueError("Danh sách tuyển thủ không được có tên trùng lặp")

        ratings_data = self.get_ratings()
        p_map = {}
        for pid in unique_players:
            if pid in ratings_data['players']:
                p_map[pid] = ratings_data['players'][pid]
            else:
                p_map[pid] = self._make_default_player_rating(pid, {})

        pair_synergy = ratings_data.get('pair_synergy', {})

        # Sắp xếp 10 người theo Điểm Thực Lực giảm dần
        sorted_10 = sorted(unique_players, key=lambda p: p_map[p]['power_score'], reverse=True)
        top1, top2 = sorted_10[0], sorted_10[1]
        bot9, bot10 = sorted_10[-2], sorted_10[-1]

        candidates = []
        for t1_cand in combinations(sorted_10, 5):
            t1_set = set(t1_cand)
            t2_cand = tuple(p for p in sorted_10 if p not in t1_set)

            # Quy tắc 1: Không dồn Top 1 và Top 2 vào cùng một đội
            if top1 in t1_set and top2 in t1_set:
                continue
            if top1 not in t1_set and top2 not in t1_set:
                continue

            # Quy tắc 2: Không dồn Bot 9 và Bot 10 vào cùng một đội
            if bot9 in t1_set and bot10 in t1_set:
                continue
            if bot9 not in t1_set and bot10 not in t1_set:
                continue

            # Tính điểm cơ bản
            base1 = sum(p_map[p]['power_score'] for p in t1_cand)
            base2 = sum(p_map[p]['power_score'] for p in t2_cand)

            # Điểm ăn ý bộ đôi
            syn1 = 0.0
            for a, b in combinations(t1_cand, 2):
                key = f"{a}|{b}" if f"{a}|{b}" in pair_synergy else f"{b}|{a}"
                if key in pair_synergy:
                    syn1 += pair_synergy[key]['synergy_score']

            syn2 = 0.0
            for a, b in combinations(t2_cand, 2):
                key = f"{a}|{b}" if f"{a}|{b}" in pair_synergy else f"{b}|{a}"
                if key in pair_synergy:
                    syn2 += pair_synergy[key]['synergy_score']

            score1 = round(base1 + syn1, 1)
            score2 = round(base2 + syn2, 1)
            diff = round(abs(score1 - score2), 1)

            candidates.append({
                'team1': t1_cand,
                'team2': t2_cand,
                'score1': score1,
                'score2': score2,
                'base1': round(base1, 1),
                'base2': round(base2, 1),
                'diff': diff
            })

        if not candidates:
            # Fallback nếu ràng buộc quá gắt (rất hiếm)
            for t1_cand in combinations(sorted_10, 5):
                t1_set = set(t1_cand)
                t2_cand = tuple(p for p in sorted_10 if p not in t1_set)
                base1 = sum(p_map[p]['power_score'] for p in t1_cand)
                base2 = sum(p_map[p]['power_score'] for p in t2_cand)
                diff = round(abs(base1 - base2), 1)
                candidates.append({
                    'team1': t1_cand,
                    'team2': t2_cand,
                    'score1': base1,
                    'score2': base2,
                    'base1': base1,
                    'base2': base2,
                    'diff': diff
                })

        candidates.sort(key=lambda x: x['diff'])
        min_diff = candidates[0]['diff']

        # Chọn kết quả (RNG mềm)
        if allow_rng and len(candidates) > 1:
            viable = [c for c in candidates if c['diff'] <= (min_diff + rng_tolerance)]
            viable = viable[:6]
            weights = [math.exp(-0.8 * (c['diff'] - min_diff)) for c in viable]
            chosen = random.choices(viable, weights=weights, k=1)[0]
        else:
            chosen = candidates[0]

        best_t1 = chosen['team1']
        best_t2 = chosen['team2']
        score1 = chosen['score1']
        score2 = chosen['score2']
        diff = chosen['diff']

        # Xác suất thắng dựa trên hiệu số điểm thực lực
        diff_power = score1 - score2
        prob1 = round((1.0 / (1.0 + 10.0 ** (-diff_power / 25.0))) * 100, 1)
        prob2 = round(100.0 - prob1, 1)

        return {
            'team1': [p_map[p] for p in best_t1],
            'team2': [p_map[p] for p in best_t2],
            'team1_names': list(best_t1),
            'team2_names': list(best_t2),
            'team1_power': score1,
            'team2_power': score2,
            'team1_base_power': chosen['base1'],
            'team2_base_power': chosen['base2'],
            'power_difference': diff,
            'team1_win_prob': prob1,
            'team2_win_prob': prob2,
            'min_power_diff': min_diff,
            'algorithm': 'SVD (Symmetric Value Draft)'
        }

    def balance_teams_captains_svd(
        self,
        captain1: str,
        captain2: str,
        remaining_players: List[str],
        allow_rng: bool = True,
        rng_tolerance: float = 3.0
    ) -> Dict[str, Any]:
        """Chia đội với 2 Đội trưởng cố định và 8 tuyển thủ còn lại theo SVD."""
        if len(remaining_players) != 8:
            raise ValueError("Cần chính xác 8 người chơi còn lại cho 2 đội trưởng")

        cap1 = str(captain1).strip().lower()
        cap2 = str(captain2).strip().lower()
        all_players = [cap1, cap2] + [str(p).strip().lower() for p in remaining_players]
        if len(set(all_players)) != 10:
            raise ValueError("Đội trưởng và các người chơi không được trùng lặp")

        ratings_data = self.get_ratings()
        p_map = {}
        for pid in all_players:
            if pid in ratings_data['players']:
                p_map[pid] = ratings_data['players'][pid]
            else:
                p_map[pid] = self._make_default_player_rating(pid, {})

        rem = [str(p).strip().lower() for p in remaining_players]
        candidates = []

        for comb in combinations(rem, 4):
            t1 = tuple([cap1] + list(comb))
            t2 = tuple([cap2] + list(set(rem) - set(comb)))

            base1 = sum(p_map[p]['power_score'] for p in t1)
            base2 = sum(p_map[p]['power_score'] for p in t2)
            diff = round(abs(base1 - base2), 1)

            candidates.append({
                'team1': t1,
                'team2': t2,
                'score1': base1,
                'score2': base2,
                'diff': diff
            })

        candidates.sort(key=lambda x: x['diff'])
        min_diff = candidates[0]['diff']

        if allow_rng and len(candidates) > 1:
            viable = [c for c in candidates if c['diff'] <= (min_diff + rng_tolerance)]
            viable = viable[:6]
            weights = [math.exp(-0.8 * (c['diff'] - min_diff)) for c in viable]
            chosen = random.choices(viable, weights=weights, k=1)[0]
        else:
            chosen = candidates[0]

        best_t1 = chosen['team1']
        best_t2 = chosen['team2']
        score1 = chosen['score1']
        score2 = chosen['score2']
        diff = chosen['diff']

        diff_power = score1 - score2
        prob1 = round((1.0 / (1.0 + 10.0 ** (-diff_power / 25.0))) * 100, 1)
        prob2 = round(100.0 - prob1, 1)

        return {
            'team1': [p_map[p] for p in best_t1],
            'team2': [p_map[p] for p in best_t2],
            'captain1': p_map[cap1],
            'captain2': p_map[cap2],
            'team1_names': list(best_t1),
            'team2_names': list(best_t2),
            'team1_power': score1,
            'team2_power': score2,
            'power_difference': diff,
            'team1_win_prob': prob1,
            'team2_win_prob': prob2,
            'min_power_diff': min_diff,
            'algorithm': 'Captains SVD'
        }


rating_service = RatingService()
