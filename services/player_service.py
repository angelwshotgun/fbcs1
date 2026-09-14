from typing import Dict, List, Any, Optional, Tuple
from services.data_manager import data_manager
from services.elo_service import elo_service


class PlayerService:
    def __init__(self):
        self._cached_metrics = None

    def refresh_metrics(self) -> Dict[str, Any]:
        """Tính toán lại toàn bộ metrics từ match_data.csv."""
        df = data_manager.read_matches_df()
        self._cached_metrics = elo_service.calculate_all_metrics(df)
        return self._cached_metrics

    def get_metrics(self) -> Dict[str, Any]:
        if self._cached_metrics is None:
            self.refresh_metrics()
        return self._cached_metrics

    def get_all_players(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả người chơi kết hợp giữa:
        - Hồ sơ stats trong players.json (Kỹ năng, Bể tướng, Flex lane, Tính ổn định)
        - Elo ẩn, Phong độ tự động và thống kê từ lịch sử trận đấu (elo_service)
        """
        metrics = self.get_metrics()
        elo_map = metrics.get('elo', {})
        elo_norm_map = metrics.get('elo_normalized', {})
        form_map = metrics.get('form', {})
        stats_map = metrics.get('stats', {})

        profiles = data_manager.read_players_data()

        # Chuẩn hóa toàn bộ map về key chữ thường để tránh trùng lặp do lệch hoa/thường
        profiles_lower = {k.lower(): v for k, v in profiles.items()}
        elo_lower = {k.lower(): v for k, v in elo_map.items()}
        elo_norm_lower = {k.lower(): v for k, v in elo_norm_map.items()}
        form_lower = {k.lower(): v for k, v in form_map.items()}
        stats_lower = {k.lower(): v for k, v in stats_map.items()}

        all_ids = sorted(list(set(profiles_lower.keys()).union(set(elo_lower.keys()))))

        results = []
        for pid in all_ids:
            profile = profiles_lower.get(pid, {})

            # Stats cơ bản (thang điểm 1 - 10)
            skill = float(profile.get('skill', 7.0))
            champ_pool = float(profile.get('champion_pool', 7.0))
            flex_lane = float(profile.get('flex_lane', 6.5))
            consistency = float(profile.get('consistency', 7.0))

            # Tính điểm Stats OVR tổng hợp
            stats_ovr = round(
                0.45 * skill + 0.20 * flex_lane + 0.15 * champ_pool + 0.20 * consistency,
                1
            )

            # Elo ẩn & Phong độ từ lịch sử đấu
            hidden_elo = elo_lower.get(pid, 1200.0)
            elo_norm = elo_norm_lower.get(pid, 5.5)
            form_info = form_lower.get(pid, {
                'score': 5.0,
                'multiplier': 1.0,
                'status': 'neutral',
                'icon': '🌱',
                'label': 'Tân binh',
                'streak': 'N/A',
                'recent_5': [],
                'recent_winrate': 0.0
            })

            # Thống kê số trận
            p_stats = stats_lower.get(pid, {'matches': 0, 'wins': 0, 'losses': 0, 'winrate': 0.0})

            has_profile = pid in profiles_lower

            # Điểm thực chiến tổng hợp (Power Rating 1-10)
            if p_stats['matches'] > 0 and has_profile:
                combined_power = round(0.65 * elo_norm + 0.35 * stats_ovr, 2)
            elif p_stats['matches'] > 0:
                combined_power = round(elo_norm, 2)
            else:
                combined_power = round(stats_ovr, 2)

            effective_power = round(combined_power * form_info.get('multiplier', 1.0), 2)

            player_obj = {
                'id': pid,
                'nickname': profile.get('nickname', pid.capitalize()),
                'avatar': profile.get('avatar', f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}"),
                'has_profile': has_profile,
                'skill': skill,
                'champion_pool': champ_pool,
                'flex_lane': flex_lane,
                'consistency': consistency,
                'stats_ovr': stats_ovr,
                'favorite_champions': profile.get('favorite_champions', []),
                'primary_role': profile.get('primary_role', 'ALL'),
                'hidden_elo': hidden_elo,
                'elo_normalized': elo_norm,
                'combined_power': combined_power,
                'effective_power': effective_power,
                'form': form_info,
                'matches': p_stats.get('matches', 0),
                'wins': p_stats.get('wins', 0),
                'losses': p_stats.get('losses', 0),
                'winrate': p_stats.get('winrate', 0.0)
            }
            results.append(player_obj)

        results.sort(key=lambda x: (x['hidden_elo'], x['effective_power']), reverse=True)
        return results

    def get_player(self, player_id: str) -> Optional[Dict[str, Any]]:
        pid = player_id.strip().lower()
        all_players = self.get_all_players()
        for p in all_players:
            if p['id'].lower() == pid:
                return p
        return None

    def create_player(self, data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        player_id = data.get('id', '').strip().lower()
        if not player_id:
            return False, "ID người chơi không được để trống", None

        # Kiểm tra trùng lặp không phân biệt hoa thường
        profiles = data_manager.read_players_data()
        if any(k.lower() == player_id for k in profiles.keys()):
            return False, f"Người chơi '{player_id}' đã tồn tại", None

        # Khởi tạo thông số chuẩn hóa thang 1-10
        nickname = data.get('nickname', '').strip() or player_id.capitalize()
        avatar = data.get('avatar', '').strip() or f"https://api.dicebear.com/7.x/bottts/svg?seed={player_id}"
        skill = max(1.0, min(10.0, float(data.get('skill', 7.0))))
        champ_pool = max(1.0, min(10.0, float(data.get('champion_pool', 7.0))))
        flex_lane = max(1.0, min(10.0, float(data.get('flex_lane', 6.5))))
        consistency = max(1.0, min(10.0, float(data.get('consistency', 7.0))))
        fav_champs = data.get('favorite_champions', [])
        primary_role = data.get('primary_role', 'MID')

        new_player = {
            'id': player_id,
            'nickname': nickname,
            'avatar': avatar,
            'skill': skill,
            'champion_pool': champ_pool,
            'flex_lane': flex_lane,
            'consistency': consistency,
            'favorite_champions': fav_champs,
            'primary_role': primary_role
        }

        # Lưu đơn lẻ nhanh chóng
        success = data_manager.save_single_player(new_player)
        if success:
            return True, "Thêm người chơi thành công", self.get_player(player_id)
        return False, "Lỗi khi lưu người chơi vào cơ sở dữ liệu", None

    def update_player(self, player_id: str, data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        pid = player_id.strip().lower()
        profiles = data_manager.read_players_data()

        target_key = None
        for k in profiles.keys():
            if k.lower() == pid:
                target_key = k
                break

        if target_key is not None:
            cur = profiles[target_key].copy()
            # Nếu key cũ khác với pid chuẩn hóa (ví dụ sniperTX vs snipertx), xóa key cũ để không bị 2 bản ghi
            if target_key != pid:
                data_manager.delete_single_player(target_key)
        else:
            cur = {'id': pid}

        cur['id'] = pid
        if 'nickname' in data and data['nickname']:
            cur['nickname'] = str(data['nickname']).strip()
        if 'avatar' in data and data['avatar']:
            cur['avatar'] = str(data['avatar']).strip()
        if 'skill' in data:
            cur['skill'] = max(1.0, min(10.0, float(data['skill'])))
        if 'champion_pool' in data:
            cur['champion_pool'] = max(1.0, min(10.0, float(data['champion_pool'])))
        if 'flex_lane' in data:
            cur['flex_lane'] = max(1.0, min(10.0, float(data['flex_lane'])))
        if 'consistency' in data:
            cur['consistency'] = max(1.0, min(10.0, float(data['consistency'])))
        if 'favorite_champions' in data:
            cur['favorite_champions'] = data['favorite_champions']
        if 'primary_role' in data:
            cur['primary_role'] = str(data['primary_role']).strip().upper()

        # Lưu đơn lẻ nhanh chóng
        success = data_manager.save_single_player(cur)
        if success:
            return True, "Cập nhật thông tin thành công", self.get_player(pid)
        return False, "Lỗi khi cập nhật dữ liệu", None

    def delete_player(self, player_id: str) -> Tuple[bool, str]:
        pid = player_id.strip().lower()
        profiles = data_manager.read_players_data()
        target_keys = [k for k in profiles.keys() if k.lower() == pid]
        if target_keys:
            for k in target_keys:
                data_manager.delete_single_player(k)
            return True, f"Đã xóa hồ sơ người chơi '{pid}'"
        return False, f"Không tìm thấy người chơi '{pid}' trong danh sách hồ sơ"


player_service = PlayerService()
