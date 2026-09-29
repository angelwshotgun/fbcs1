"""
Dịch vụ Chia Đội Cân Bằng (Matchmaking Service)
- Áp dụng thuật toán SVD (Symmetric Value Draft) từ RatingService.
- Không sử dụng các chỉ số kỹ năng 1-10 hay hạn chế vị trí (lane).
- Đảm bảo đối xứng: 2 tuyển thủ mạnh nhất luôn chia đôi về 2 phe, 2 tuyển thủ yếu nhất luôn chia đôi về 2 phe.
- Tối thiểu hóa độ lệch tổng Điểm Thực Lực (Power Difference).
"""

from typing import List, Dict, Any, Tuple
from itertools import combinations
from services.player_service import player_service
from services.rating_service import rating_service


class MatchmakingService:
    def create_balanced_teams(
        self,
        player_ids: List[str],
        allow_rng: bool = True,
        rng_tolerance: float = 0.6,
        balance_mode: str = 'svd'
    ) -> Dict[str, Any]:
        """Chia 10 tuyển thủ thành 2 đội 5-5 theo thuật toán SVD (Symmetric Value Draft)."""
        if len(player_ids) != 10:
            raise ValueError("Cần chọn chính xác 10 người chơi để chia đội")

        unique_players = sorted(list(set(str(p).strip().lower() for p in player_ids)))
        if len(unique_players) != 10:
            raise ValueError("Danh sách tuyển thủ không được có tên trùng lặp")

        # Map rng_tolerance từ client (0.1 - 1.0) sang thang điểm thực lực (1.0 - 5.0)
        tolerance_pts = max(1.5, min(6.0, float(rng_tolerance) * 4.0))

        svd_result = rating_service.balance_teams_svd(
            player_ids=unique_players,
            allow_rng=allow_rng,
            rng_tolerance=tolerance_pts
        )

        all_players_map = {p['id'].lower(): p for p in player_service.get_all_players()}
        team1_objs = [all_players_map.get(pid, {'id': pid, 'nickname': pid.capitalize(), 'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}", 'power_score': 50.0, 'tier_name': 'Tier B'}) for pid in svd_result['team1_names']]
        team2_objs = [all_players_map.get(pid, {'id': pid, 'nickname': pid.capitalize(), 'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}", 'power_score': 50.0, 'tier_name': 'Tier B'}) for pid in svd_result['team2_names']]

        # Tính toán danh sách điểm ăn ý (synergies) cho giao diện hiển thị
        ratings_data = rating_service.get_ratings()
        pair_synergy = ratings_data.get('pair_synergy', {})

        team1_syns = self._extract_team_synergies(svd_result['team1_names'], pair_synergy, all_players_map)
        team2_syns = self._extract_team_synergies(svd_result['team2_names'], pair_synergy, all_players_map)

        t1_power = svd_result['team1_power']
        t2_power = svd_result['team2_power']

        return {
            'team1': team1_objs,
            'team2': team2_objs,
            'team1_names': svd_result['team1_names'],
            'team2_names': svd_result['team2_names'],
            'team1_power': t1_power,
            'team2_power': t2_power,
            'team1_total_elo': t1_power,
            'team2_total_elo': t2_power,
            'team1_avg_elo': round(t1_power / 5.0, 1),
            'team2_avg_elo': round(t2_power / 5.0, 1),
            'power_difference': svd_result['power_difference'],
            'team1_win_prob': svd_result['team1_win_prob'],
            'team2_win_prob': svd_result['team2_win_prob'],
            'team1_synergies': team1_syns,
            'team2_synergies': team2_syns,
            'rng_applied': allow_rng,
            'rng_tolerance': rng_tolerance,
            'min_power_diff': svd_result.get('min_power_diff', svd_result['power_difference']),
            'algorithm': 'SVD (Symmetric Value Draft)'
        }

    def create_teams_with_captains(
        self,
        captain1: str,
        captain2: str,
        remaining_players: List[str],
        allow_rng: bool = True,
        rng_tolerance: float = 0.6,
        balance_mode: str = 'svd'
    ) -> Dict[str, Any]:
        """Chia đội với 2 Đội trưởng cố định và 8 tuyển thủ còn lại."""
        if len(remaining_players) != 8:
            raise ValueError("Cần chính xác 8 người chơi còn lại cho 2 đội trưởng")

        tolerance_pts = max(1.5, min(6.0, float(rng_tolerance) * 4.0))

        svd_result = rating_service.balance_teams_captains_svd(
            captain1=captain1,
            captain2=captain2,
            remaining_players=remaining_players,
            allow_rng=allow_rng,
            rng_tolerance=tolerance_pts
        )

        all_players_map = {p['id'].lower(): p for p in player_service.get_all_players()}
        team1_objs = [all_players_map.get(pid, {'id': pid, 'nickname': pid.capitalize(), 'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}", 'power_score': 50.0, 'tier_name': 'Tier B'}) for pid in svd_result['team1_names']]
        team2_objs = [all_players_map.get(pid, {'id': pid, 'nickname': pid.capitalize(), 'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}", 'power_score': 50.0, 'tier_name': 'Tier B'}) for pid in svd_result['team2_names']]

        ratings_data = rating_service.get_ratings()
        pair_synergy = ratings_data.get('pair_synergy', {})

        team1_syns = self._extract_team_synergies(svd_result['team1_names'], pair_synergy, all_players_map)
        team2_syns = self._extract_team_synergies(svd_result['team2_names'], pair_synergy, all_players_map)

        t1_power = svd_result['team1_power']
        t2_power = svd_result['team2_power']

        return {
            'team1': team1_objs,
            'team2': team2_objs,
            'captain1': all_players_map.get(str(captain1).lower()),
            'captain2': all_players_map.get(str(captain2).lower()),
            'team1_names': svd_result['team1_names'],
            'team2_names': svd_result['team2_names'],
            'team1_power': t1_power,
            'team2_power': t2_power,
            'team1_total_elo': t1_power,
            'team2_total_elo': t2_power,
            'team1_avg_elo': round(t1_power / 5.0, 1),
            'team2_avg_elo': round(t2_power / 5.0, 1),
            'power_difference': svd_result['power_difference'],
            'team1_win_prob': svd_result['team1_win_prob'],
            'team2_win_prob': svd_result['team2_win_prob'],
            'team1_synergies': team1_syns,
            'team2_synergies': team2_syns,
            'rng_applied': allow_rng,
            'rng_tolerance': rng_tolerance,
            'min_power_diff': svd_result.get('min_power_diff', svd_result['power_difference']),
            'algorithm': 'Captains SVD'
        }

    def _extract_team_synergies(
        self,
        team_pids: List[str],
        pair_synergy: Dict[str, Any],
        p_map: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Trích xuất danh sách cặp bài trùng trong đội hình."""
        syns = []
        for a, b in combinations(team_pids, 2):
            key = f"{a}|{b}" if f"{a}|{b}" in pair_synergy else f"{b}|{a}"
            if key in pair_synergy:
                item = pair_synergy[key]
                score = item.get('synergy_score', 0.0)
                nick_a = p_map.get(a, {}).get('nickname', a)
                nick_b = p_map.get(b, {}).get('nickname', b)
                if score > 0.5:
                    syns.append({
                        'type': 'duo',
                        'icon': '🤝',
                        'label': 'Cặp bài trùng',
                        'names': f"{nick_a} + {nick_b} ({item.get('winrate')}%)",
                        'bonus': score
                    })
                elif score < -0.8:
                    syns.append({
                        'type': 'anti_duo',
                        'icon': '💔',
                        'label': 'Ít ăn ý',
                        'names': f"{nick_a} + {nick_b} ({item.get('winrate')}%)",
                        'bonus': score
                    })
        return syns


matchmaking_service = MatchmakingService()
