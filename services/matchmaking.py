import math
import random
from itertools import combinations
from typing import List, Dict, Any, Tuple
from services.player_service import player_service


class MatchmakingService:
    def _get_player_map(self, player_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        all_players = player_service.get_all_players()
        p_map = {p['id']: p for p in all_players}

        result = {}
        for pid in player_ids:
            if pid in p_map:
                result[pid] = p_map[pid]
            else:
                # Nếu người chơi chưa có hồ sơ
                result[pid] = {
                    'id': pid,
                    'nickname': pid.capitalize(),
                    'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}",
                    'skill': 5.0,
                    'champion_pool': 5.0,
                    'flex_lane': 5.0,
                    'consistency': 5.0,
                    'stats_ovr': 5.0,
                    'hidden_elo': 1200.0,
                    'elo_normalized': 5.5,
                    'effective_power': 5.5,
                    'form': {
                        'score': 5.0,
                        'multiplier': 1.0,
                        'status': 'neutral',
                        'icon': '🌱',
                        'label': 'Tân binh',
                        'streak': 'N/A'
                    },
                    'matches': 0,
                    'winrate': 0.0
                }
        return result

    def _get_pair_synergy(self, p1: str, p2: str, synergy_map: Dict[str, Any]) -> float:
        pair_key1 = f"{p1}|{p2}"
        pair_key2 = f"{p2}|{p1}"
        if pair_key1 in synergy_map:
            return float(synergy_map[pair_key1].get('synergy_score', 0.0))
        if pair_key2 in synergy_map:
            return float(synergy_map[pair_key2].get('synergy_score', 0.0))
        return 0.0

    def _get_trio_synergy(self, p1: str, p2: str, p3: str, trio_synergy_map: Dict[str, Any]) -> float:
        sorted_keys = sorted([p1, p2, p3])
        trio_key = "|".join(sorted_keys)
        if trio_key in trio_synergy_map:
            return float(trio_synergy_map[trio_key].get('synergy_score', 0.0))
        return 0.0

    def _evaluate_team(
        self,
        team_tuple: Tuple[str, ...],
        p_map: Dict[str, Any],
        pair_synergy_map: Dict[str, Any],
        trio_synergy_map: Dict[str, Any],
        balance_mode: str = 'pure_elo'
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Đánh giá sức mạnh của 1 đội 5 người:
        - Mode 'pure_elo': Thuần Elo Ẩn từ lịch sử trận đấu (KHÔNG sử dụng chỉ số stats 1-10).
        - Mode 'composite': Toàn diện (65% Elo ẩn + 35% Stats + Hệ số phong độ + Bổ trợ Stats).
        """
        synergies: List[Dict[str, Any]] = []

        if balance_mode == 'pure_elo':
            # Chế độ THUẦN ELO ẨN: 100% dựa vào Hidden Elo MMR từ kết quả thi đấu, bỏ qua toàn bộ stats
            base_elo = sum(p_map[p]['hidden_elo'] for p in team_tuple)

            # Cặp bài trùng / tam tấu từ lịch sử thi đấu chung (quy đổi ra điểm Elo)
            duo_bonus = 0.0
            for p1, p2 in combinations(team_tuple, 2):
                syn = self._get_pair_synergy(p1, p2, pair_synergy_map)
                if syn > 0.15:
                    bonus_val = round(syn * 25.0, 1)
                    duo_bonus += bonus_val
                    synergies.append({
                        'type': 'duo',
                        'icon': '🤝',
                        'label': 'Cặp bài trùng (Elo)',
                        'players': [p1, p2],
                        'names': f"{p_map[p1]['nickname']} + {p_map[p2]['nickname']}",
                        'bonus': bonus_val
                    })
                elif syn < -0.2:
                    penalty_val = round(syn * 20.0, 1)
                    duo_bonus += penalty_val
                    synergies.append({
                        'type': 'anti_duo',
                        'icon': '💔',
                        'label': 'Khắc khẩu (Elo)',
                        'players': [p1, p2],
                        'names': f"{p_map[p1]['nickname']} + {p_map[p2]['nickname']}",
                        'bonus': penalty_val
                    })

            trio_bonus = 0.0
            for p1, p2, p3 in combinations(team_tuple, 3):
                t_syn = self._get_trio_synergy(p1, p2, p3, trio_synergy_map)
                if t_syn > 0.15:
                    b_val = round(t_syn * 30.0, 1)
                    trio_bonus += b_val
                    synergies.append({
                        'type': 'trio',
                        'icon': '🌟',
                        'label': 'Bộ ba tam tấu (Elo)',
                        'players': [p1, p2, p3],
                        'names': f"{p_map[p1]['nickname']} + {p_map[p2]['nickname']}",
                        'bonus': b_val
                    })
                elif t_syn < -0.2:
                    p_val = round(t_syn * 25.0, 1)
                    trio_bonus += p_val
                    synergies.append({
                        'type': 'anti_trio',
                        'icon': '⚠️',
                        'label': 'Bộ ba xung đột (Elo)',
                        'players': [p1, p2, p3],
                        'names': f"{p_map[p1]['nickname']} + {p_map[p2]['nickname']}",
                        'bonus': p_val
                    })

            # TUYỆT ĐỐI KHÔNG DÙNG CHỈ SỐ STATS (Kỹ năng, Bể tướng, Flex lane, Tính ổn định)
            total_score = round(base_elo + duo_bonus + trio_bonus, 1)
            return total_score, synergies

        # Chế độ Toàn Diện (Composite): Kết hợp Elo ẩn + Stats 1-10 + Phong độ + Bổ trợ chiến thuật
        base_power = sum(p_map[p]['effective_power'] for p in team_tuple)

        # 1. Duo Synergy (Cặp 2 người)
        duo_bonus_sum = 0.0
        for p1, p2 in combinations(team_tuple, 2):
            syn = self._get_pair_synergy(p1, p2, pair_synergy_map)
            if syn > 0.15:
                duo_bonus_sum += syn * 0.25
                synergies.append({
                    'type': 'duo',
                    'icon': '🤝',
                    'label': 'Cặp bài trùng',
                    'players': [p1, p2],
                    'names': f"{p_map[p1]['nickname']} + {p_map[p2]['nickname']}",
                    'bonus': round(syn * 0.25, 2)
                })
            elif syn < -0.2:
                duo_bonus_sum += syn * 0.20
                synergies.append({
                    'type': 'anti_duo',
                    'icon': '💔',
                    'label': 'Khắc khẩu',
                    'players': [p1, p2],
                    'names': f"{p_map[p1]['nickname']} + {p_map[p2]['nickname']}",
                    'bonus': round(syn * 0.20, 2)
                })

        # 2. Trio Synergy (Bộ ba 3 người cùng chơi tốt)
        trio_bonus_sum = 0.0
        for p1, p2, p3 in combinations(team_tuple, 3):
            t_syn = self._get_trio_synergy(p1, p2, p3, trio_synergy_map)
            if t_syn > 0.15:
                trio_bonus_sum += t_syn * 0.35
                synergies.append({
                    'type': 'trio',
                    'icon': '🌟',
                    'label': 'Bộ ba tam tấu',
                    'players': [p1, p2, p3],
                    'names': f"{p_map[p1]['nickname']} + {p_map[p2]['nickname']} + {p_map[p3]['nickname']}",
                    'bonus': round(t_syn * 0.35, 2)
                })
            elif t_syn < -0.2:
                trio_bonus_sum += t_syn * 0.25
                synergies.append({
                    'type': 'anti_trio',
                    'icon': '⚠️',
                    'label': 'Bộ ba xung đột',
                    'players': [p1, p2, p3],
                    'names': f"{p_map[p1]['nickname']} + {p_map[p2]['nickname']} + {p_map[p3]['nickname']}",
                    'bonus': round(t_syn * 0.25, 2)
                })

        # 3. Chemistry Bổ trợ tự nhiên (Stats & Form Heuristics)
        chem_bonus = 0.0
        # Mũi nhọn (Skill >= 7.8) + Trụ cột (Consistency >= 7.5) bù trừ cho nhau
        sharps = [p for p in team_tuple if p_map[p]['skill'] >= 7.8]
        anchors = [p for p in team_tuple if p_map[p]['consistency'] >= 7.5]
        if sharps and anchors:
            chem_bonus += 0.12
            synergies.append({
                'type': 'chemistry',
                'icon': '⚡',
                'label': 'Bổ trợ chiến thuật',
                'names': f"{p_map[sharps[0]]['nickname']} (Mũi nhọn) & {p_map[anchors[0]]['nickname']} (Trụ cột)",
                'bonus': 0.12
            })

        # Bộ ba linh hoạt flex lane (>= 3 người có flex_lane >= 7.0)
        flex_players = [p for p in team_tuple if p_map[p]['flex_lane'] >= 7.0]
        if len(flex_players) >= 3:
            chem_bonus += 0.10
            synergies.append({
                'type': 'chemistry',
                'icon': '🔄',
                'label': 'Đa năng biến ảo',
                'names': f"{len(flex_players)} người chơi flex lane cao",
                'bonus': 0.10
            })

        # Cộng hưởng phong độ (Form resonance)
        hot_count = sum(1 for p in team_tuple if p_map[p].get('form', {}).get('status') in ['on_fire', 'good'])
        cold_count = sum(1 for p in team_tuple if p_map[p].get('form', {}).get('status') in ['cold', 'slump'])
        if hot_count >= 3:
            chem_bonus += 0.15
            synergies.append({
                'type': 'chemistry',
                'icon': '🔥',
                'label': 'Cộng hưởng hưng phấn',
                'names': f"{hot_count} tuyển thủ đang phong độ cao",
                'bonus': 0.15
            })
        elif cold_count >= 3:
            chem_bonus -= 0.15
            synergies.append({
                'type': 'chemistry',
                'icon': '❄️',
                'label': 'Áp lực tâm lý',
                'names': f"{cold_count} tuyển thủ đang xuống phong độ",
                'bonus': -0.15
            })

        total_score = round(base_power + duo_bonus_sum + trio_bonus_sum + chem_bonus, 2)
        return total_score, synergies

    def _select_candidate_with_rng(
        self,
        candidates: List[Dict[str, Any]],
        allow_rng: bool = True,
        rng_tolerance: float = 0.6,
        balance_mode: str = 'pure_elo'
    ) -> Tuple[Dict[str, Any], float, int]:
        """
        Chọn kết quả chia đội có tính toán ngẫu nhiên (RNG) trong ngưỡng cân bằng:
        - Sắp xếp các phương án chia theo độ chênh lệch tăng dần.
        - Lọc các phương án có diff <= min_diff + rng_tolerance.
        - Chọn ngẫu nhiên có trọng số (weighted random), đảm bảo kết quả luôn cân bằng cao
          nhưng không bị cố định một mẫu rập khuôn mỗi lần bấm.
        """
        candidates.sort(key=lambda x: x['diff'])
        min_diff = candidates[0]['diff']

        if not allow_rng or len(candidates) == 1:
            return candidates[0], min_diff, 1

        if balance_mode == 'pure_elo':
            tolerance = max(8.0, min(50.0, float(rng_tolerance) * 40.0))
            viable = [
                c for c in candidates
                if c['diff'] <= (min_diff + tolerance) and c['diff'] <= 60.0
            ]
            decay = 2.2 / 25.0
        else:
            tolerance = max(0.2, min(1.2, float(rng_tolerance)))
            viable = [
                c for c in candidates
                if c['diff'] <= (min_diff + tolerance) and c['diff'] <= 1.3
            ]
            decay = 2.2

        viable = viable[:6]
        if not viable:
            viable = [candidates[0]]

        # Trọng số mềm (phương án càng gần min_diff càng có xác suất cao)
        weights = [math.exp(-decay * (c['diff'] - min_diff)) for c in viable]
        chosen = random.choices(viable, weights=weights, k=1)[0]
        return chosen, min_diff, len(viable)

    def create_balanced_teams(
        self,
        player_ids: List[str],
        allow_rng: bool = True,
        rng_tolerance: float = 0.6,
        balance_mode: str = 'pure_elo'
    ) -> Dict[str, Any]:
        """Chia 10 người chơi thành 2 đội 5-5 với tính toán cân bằng và sai số RNG thông minh."""
        if len(player_ids) != 10:
            raise ValueError("Cần chọn chính xác 10 người chơi để chia đội")

        unique_players = sorted(list(set(player_ids)))
        if len(unique_players) != 10:
            raise ValueError("Danh sách người chơi không được có tên trùng lặp")

        p_map = self._get_player_map(unique_players)
        metrics = player_service.get_metrics()
        pair_synergy = metrics.get('pair_synergy', {})
        trio_synergy = metrics.get('trio_synergy', {})

        comb_5 = list(combinations(unique_players, 5))
        evaluated_pairs = set()
        candidates: List[Dict[str, Any]] = []

        for t1_tuple in comb_5:
            t1_set = set(t1_tuple)
            t2_tuple = tuple(sorted(list(set(unique_players) - t1_set)))

            # Tránh lặp lại (Team 1 vs Team 2 tương đương Team 2 vs Team 1)
            pair_signature = tuple(sorted([t1_tuple, t2_tuple]))
            if pair_signature in evaluated_pairs:
                continue
            evaluated_pairs.add(pair_signature)

            score_1, syns_1 = self._evaluate_team(t1_tuple, p_map, pair_synergy, trio_synergy, balance_mode=balance_mode)
            score_2, syns_2 = self._evaluate_team(t2_tuple, p_map, pair_synergy, trio_synergy, balance_mode=balance_mode)
            diff = abs(score_1 - score_2)

            candidates.append({
                'team1': t1_tuple,
                'team2': t2_tuple,
                'score1': score_1,
                'score2': score_2,
                'diff': diff,
                'syns1': syns_1,
                'syns2': syns_2
            })

        chosen, min_diff, pool_count = self._select_candidate_with_rng(
            candidates,
            allow_rng=allow_rng,
            rng_tolerance=rng_tolerance,
            balance_mode=balance_mode
        )

        best_team1 = chosen['team1']
        best_team2 = chosen['team2']
        best_score1 = chosen['score1']
        best_score2 = chosen['score2']
        diff = chosen['diff']

        # Tính tỷ lệ thắng dự đoán
        if balance_mode == 'pure_elo':
            diff_elo = (best_score1 - best_score2) / 5.0
            prob_1 = round((1.0 / (1.0 + 10.0 ** (-diff_elo / 400.0))) * 100, 1)
            prob_2 = round(100.0 - prob_1, 1)
        else:
            diff_power = best_score1 - best_score2
            prob_1 = round((1.0 / (1.0 + 10.0 ** (-diff_power / 4.0))) * 100, 1)
            prob_2 = round(100.0 - prob_1, 1)

        t1_total_elo = round(sum(p_map[p]['hidden_elo'] for p in best_team1))
        t2_total_elo = round(sum(p_map[p]['hidden_elo'] for p in best_team2))

        return {
            'team1': [p_map[p] for p in best_team1],
            'team2': [p_map[p] for p in best_team2],
            'team1_names': list(best_team1),
            'team2_names': list(best_team2),
            'team1_power': round(best_score1, 1 if balance_mode == 'pure_elo' else 2),
            'team2_power': round(best_score2, 1 if balance_mode == 'pure_elo' else 2),
            'team1_total_elo': t1_total_elo,
            'team2_total_elo': t2_total_elo,
            'team1_avg_elo': round(t1_total_elo / 5.0, 1),
            'team2_avg_elo': round(t2_total_elo / 5.0, 1),
            'power_difference': round(diff, 1 if balance_mode == 'pure_elo' else 2),
            'team1_win_prob': prob_1,
            'team2_win_prob': prob_2,
            'team1_synergies': chosen['syns1'],
            'team2_synergies': chosen['syns2'],
            'rng_applied': allow_rng,
            'rng_tolerance': rng_tolerance,
            'min_power_diff': round(min_diff, 1 if balance_mode == 'pure_elo' else 2),
            'pool_candidates_count': pool_count,
            'balance_mode': balance_mode
        }

    def create_teams_with_captains(
        self,
        captain1: str,
        captain2: str,
        remaining_players: List[str],
        allow_rng: bool = True,
        rng_tolerance: float = 0.6,
        balance_mode: str = 'pure_elo'
    ) -> Dict[str, Any]:
        """Chia đội với 2 Đội trưởng cố định và 8 thành viên còn lại (hỗ trợ RNG cân bằng)."""
        if len(remaining_players) != 8:
            raise ValueError("Cần chính xác 8 người chơi còn lại cho 2 đội trưởng")

        all_players = [captain1, captain2] + remaining_players
        if len(set(all_players)) != 10:
            raise ValueError("Đội trưởng và các người chơi không được trùng lặp")

        p_map = self._get_player_map(all_players)
        metrics = player_service.get_metrics()
        pair_synergy = metrics.get('pair_synergy', {})
        trio_synergy = metrics.get('trio_synergy', {})

        comb_4 = list(combinations(remaining_players, 4))
        candidates: List[Dict[str, Any]] = []

        for comb in comb_4:
            t1 = tuple([captain1] + list(comb))
            t2 = tuple([captain2] + list(set(remaining_players) - set(comb)))

            score_1, syns_1 = self._evaluate_team(t1, p_map, pair_synergy, trio_synergy, balance_mode=balance_mode)
            score_2, syns_2 = self._evaluate_team(t2, p_map, pair_synergy, trio_synergy, balance_mode=balance_mode)
            diff = abs(score_1 - score_2)

            candidates.append({
                'team1': t1,
                'team2': t2,
                'score1': score_1,
                'score2': score_2,
                'diff': diff,
                'syns1': syns_1,
                'syns2': syns_2
            })

        chosen, min_diff, pool_count = self._select_candidate_with_rng(
            candidates,
            allow_rng=allow_rng,
            rng_tolerance=rng_tolerance,
            balance_mode=balance_mode
        )

        best_team1 = chosen['team1']
        best_team2 = chosen['team2']
        best_score1 = chosen['score1']
        best_score2 = chosen['score2']
        diff = chosen['diff']

        if balance_mode == 'pure_elo':
            diff_elo = (best_score1 - best_score2) / 5.0
            prob_1 = round((1.0 / (1.0 + 10.0 ** (-diff_elo / 400.0))) * 100, 1)
            prob_2 = round(100.0 - prob_1, 1)
        else:
            diff_power = best_score1 - best_score2
            prob_1 = round((1.0 / (1.0 + 10.0 ** (-diff_power / 4.0))) * 100, 1)
            prob_2 = round(100.0 - prob_1, 1)

        t1_total_elo = round(sum(p_map[p]['hidden_elo'] for p in best_team1))
        t2_total_elo = round(sum(p_map[p]['hidden_elo'] for p in best_team2))

        return {
            'team1': [p_map[p] for p in best_team1],
            'team2': [p_map[p] for p in best_team2],
            'captain1': p_map[captain1],
            'captain2': p_map[captain2],
            'team1_names': list(best_team1),
            'team2_names': list(best_team2),
            'team1_power': round(best_score1, 1 if balance_mode == 'pure_elo' else 2),
            'team2_power': round(best_score2, 1 if balance_mode == 'pure_elo' else 2),
            'team1_total_elo': t1_total_elo,
            'team2_total_elo': t2_total_elo,
            'team1_avg_elo': round(t1_total_elo / 5.0, 1),
            'team2_avg_elo': round(t2_total_elo / 5.0, 1),
            'power_difference': round(diff, 1 if balance_mode == 'pure_elo' else 2),
            'team1_win_prob': prob_1,
            'team2_win_prob': prob_2,
            'team1_synergies': chosen['syns1'],
            'team2_synergies': chosen['syns2'],
            'rng_applied': allow_rng,
            'rng_tolerance': rng_tolerance,
            'min_power_diff': round(min_diff, 1 if balance_mode == 'pure_elo' else 2),
            'pool_candidates_count': pool_count,
            'balance_mode': balance_mode
        }


matchmaking_service = MatchmakingService()
