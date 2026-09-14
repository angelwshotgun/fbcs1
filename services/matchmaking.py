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
                    'skill': 7.0,
                    'champion_pool': 7.0,
                    'flex_lane': 6.5,
                    'consistency': 7.0,
                    'stats_ovr': 6.9,
                    'hidden_elo': 1200.0,
                    'elo_normalized': 5.5,
                    'effective_power': 6.0,
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

    def create_balanced_teams(self, player_ids: List[str]) -> Dict[str, Any]:
        """Chia 10 người chơi thành 2 đội 5-5 cân bằng nhất."""
        if len(player_ids) != 10:
            raise ValueError("Cần chọn chính xác 10 người chơi để chia đội")

        unique_players = sorted(list(set(player_ids)))
        if len(unique_players) != 10:
            raise ValueError("Danh sách người chơi không được có tên trùng lặp")

        p_map = self._get_player_map(unique_players)
        metrics = player_service.get_metrics()
        synergy_map = metrics.get('pair_synergy', {})

        comb_5 = list(combinations(unique_players, 5))
        best_team1, best_team2 = None, None
        min_diff = float('inf')
        best_score1, best_score2 = 0.0, 0.0

        # C(10, 5) = 252. Chia đôi là 126 cặp đối xứng
        evaluated_pairs = set()

        for t1_tuple in comb_5:
            t1_set = set(t1_tuple)
            t2_tuple = tuple(sorted(list(set(unique_players) - t1_set)))

            # Tránh lặp lại (Team 1 vs Team 2 tương đương Team 2 vs Team 1)
            pair_signature = tuple(sorted([t1_tuple, t2_tuple]))
            if pair_signature in evaluated_pairs:
                continue
            evaluated_pairs.add(pair_signature)

            # Tính điểm sức mạnh Team 1
            power_1 = sum(p_map[p]['effective_power'] for p in t1_tuple)
            syn_1 = sum(self._get_pair_synergy(p1, p2, synergy_map) for p1, p2 in combinations(t1_tuple, 2))
            total_1 = power_1 + (syn_1 * 0.25)

            # Tính điểm sức mạnh Team 2
            power_2 = sum(p_map[p]['effective_power'] for p in t2_tuple)
            syn_2 = sum(self._get_pair_synergy(p1, p2, synergy_map) for p1, p2 in combinations(t2_tuple, 2))
            total_2 = power_2 + (syn_2 * 0.25)

            diff = abs(total_1 - total_2)
            if diff < min_diff:
                min_diff = diff
                best_team1 = t1_tuple
                best_team2 = t2_tuple
                best_score1 = total_1
                best_score2 = total_2

        # Tính tỷ lệ thắng dự đoán
        diff_power = best_score1 - best_score2
        # Logistic curve: scale 4.0
        prob_1 = round((1.0 / (1.0 + 10.0 ** (-diff_power / 4.0))) * 100, 1)
        prob_2 = round(100.0 - prob_1, 1)

        # Lấy thông tin chi tiết các cặp đôi ăn ý trong mỗi team
        t1_synergies = []
        for p1, p2 in combinations(best_team1, 2):
            syn = self._get_pair_synergy(p1, p2, synergy_map)
            if syn > 0.15:
                t1_synergies.append({
                    'p1': p_map[p1]['nickname'],
                    'p2': p_map[p2]['nickname'],
                    'bonus': syn
                })

        t2_synergies = []
        for p1, p2 in combinations(best_team2, 2):
            syn = self._get_pair_synergy(p1, p2, synergy_map)
            if syn > 0.15:
                t2_synergies.append({
                    'p1': p_map[p1]['nickname'],
                    'p2': p_map[p2]['nickname'],
                    'bonus': syn
                })

        return {
            'team1': [p_map[p] for p in best_team1],
            'team2': [p_map[p] for p in best_team2],
            'team1_names': list(best_team1),
            'team2_names': list(best_team2),
            'team1_power': round(best_score1, 2),
            'team2_power': round(best_score2, 2),
            'power_difference': round(min_diff, 2),
            'team1_win_prob': prob_1,
            'team2_win_prob': prob_2,
            'team1_synergies': t1_synergies,
            'team2_synergies': t2_synergies
        }

    def create_teams_with_captains(self, captain1: str, captain2: str, remaining_players: List[str]) -> Dict[str, Any]:
        """Chia đội với 2 Đội trưởng cố định và 8 thành viên còn lại."""
        if len(remaining_players) != 8:
            raise ValueError("Cần chính xác 8 người chơi còn lại cho 2 đội trưởng")

        all_players = [captain1, captain2] + remaining_players
        if len(set(all_players)) != 10:
            raise ValueError("Đội trưởng và các người chơi không được trùng lặp")

        p_map = self._get_player_map(all_players)
        metrics = player_service.get_metrics()
        synergy_map = metrics.get('pair_synergy', {})

        comb_4 = list(combinations(remaining_players, 4))
        best_team1, best_team2 = None, None
        min_diff = float('inf')
        best_score1, best_score2 = 0.0, 0.0

        for comb in comb_4:
            t1 = tuple([captain1] + list(comb))
            t2 = tuple([captain2] + list(set(remaining_players) - set(comb)))

            power_1 = sum(p_map[p]['effective_power'] for p in t1)
            syn_1 = sum(self._get_pair_synergy(p1, p2, synergy_map) for p1, p2 in combinations(t1, 2))
            total_1 = power_1 + (syn_1 * 0.25)

            power_2 = sum(p_map[p]['effective_power'] for p in t2)
            syn_2 = sum(self._get_pair_synergy(p1, p2, synergy_map) for p1, p2 in combinations(t2, 2))
            total_2 = power_2 + (syn_2 * 0.25)

            diff = abs(total_1 - total_2)
            if diff < min_diff:
                min_diff = diff
                best_team1 = t1
                best_team2 = t2
                best_score1 = total_1
                best_score2 = total_2

        diff_power = best_score1 - best_score2
        prob_1 = round((1.0 / (1.0 + 10.0 ** (-diff_power / 4.0))) * 100, 1)
        prob_2 = round(100.0 - prob_1, 1)

        return {
            'team1': [p_map[p] for p in best_team1],
            'team2': [p_map[p] for p in best_team2],
            'captain1': p_map[captain1],
            'captain2': p_map[captain2],
            'team1_names': list(best_team1),
            'team2_names': list(best_team2),
            'team1_power': round(best_score1, 2),
            'team2_power': round(best_score2, 2),
            'power_difference': round(min_diff, 2),
            'team1_win_prob': prob_1,
            'team2_win_prob': prob_2
        }


matchmaking_service = MatchmakingService()
