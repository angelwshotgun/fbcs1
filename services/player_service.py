from typing import Dict, List, Any, Optional, Tuple
from services.data_manager import data_manager
from services.rating_service import rating_service


class PlayerService:
    def __init__(self):
        self._cached_metrics = None

    def refresh_metrics(self) -> Dict[str, Any]:
        """Làm mới cache và tính toán lại bảng năng lực RAPM từ dữ liệu trận đấu."""
        rating_service.invalidate_cache()
        self._cached_metrics = rating_service.get_ratings(force_refresh=True)
        return self._cached_metrics

    def get_metrics(self) -> Dict[str, Any]:
        if self._cached_metrics is None:
            self.refresh_metrics()
        return self._cached_metrics

    def get_all_players(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tuyển thủ với Điểm Thực Lực (0 - 100), Phân Bậc Tier (S, A, B, C)
        và thống kê thuần túy từ kết quả thi đấu thực tế (không có chỉ số cảm tính hay AI).
        """
        ratings_data = rating_service.get_ratings()
        players_map = ratings_data.get('players', {})
        profiles = data_manager.read_players_data()
        profiles_lower = {str(k).lower(): v for k, v in profiles.items()}

        all_ids = sorted(list(profiles_lower.keys()))
        results = []

        for pid in all_ids:
            profile = profiles_lower.get(pid, {})
            p_rating = players_map.get(pid)

            if not p_rating:
                p_rating = rating_service._make_default_player_rating(pid, profile)

            power_score = p_rating.get('power_score', 50.0)
            m_count = p_rating.get('matches', 0)
            wins = p_rating.get('wins', 0)
            losses = p_rating.get('losses', 0)
            winrate = p_rating.get('winrate', 0.0)
            recent_5 = p_rating.get('recent_5', [])
            tier = p_rating.get('tier', 'B')
            tier_name = p_rating.get('tier_name', 'Tier B')
            tier_icon = p_rating.get('tier_icon', '🛡️')
            tier_stars = p_rating.get('tier_stars', '⭐⭐⭐')
            tier_badge_class = p_rating.get('tier_badge_class', '')
            tier_desc = p_rating.get('tier_desc', '')

            # Tính win streak / loss streak từ recent_5
            cur_streak = 0
            streak_type = 'W'
            if recent_5:
                streak_type = recent_5[-1]
                for r in reversed(recent_5):
                    if r == streak_type:
                        cur_streak += 1
                    else:
                        break

            # Tạo Badges thành tích thuần túy số liệu thi đấu (hoàn toàn khách quan)
            badges = []

            # 1. Badge số trận (Kinh nghiệm)
            if m_count >= 15:
                badges.append({
                    'key': 'veteran',
                    'label': f'Lão Làng ({m_count} trận)',
                    'icon': '🎖️',
                    'badge_class': 'bg-slate-100 text-slate-800 border-slate-300 font-bold',
                    'desc': f'Đã thi đấu {m_count} trận — kỳ cựu của phòng đấu'
                })
            elif m_count >= 8:
                badges.append({
                    'key': 'experienced',
                    'label': f'Dày Dạn ({m_count} trận)',
                    'icon': '⚔️',
                    'badge_class': 'bg-slate-100 text-slate-700 border-slate-200 font-bold',
                    'desc': f'Đã tham gia {m_count} trận đấu'
                })

            # 2. Badge Tỷ lệ thắng cao
            if m_count >= 4:
                if winrate >= 66.7:
                    badges.append({
                        'key': 'high_wr',
                        'label': f'Bất Bại ({winrate}%)',
                        'icon': '💎',
                        'badge_class': 'bg-emerald-100 text-emerald-800 border-emerald-300 font-bold',
                        'desc': f'Tỷ lệ thắng áp đảo {winrate}% qua {m_count} trận'
                    })
                elif winrate <= 30.0:
                    badges.append({
                        'key': 'underdog',
                        'label': f'Cần Bứt Phá ({winrate}%)',
                        'icon': '🔥',
                        'badge_class': 'bg-amber-100 text-amber-800 border-amber-300 font-bold',
                        'desc': f'Tỷ lệ thắng {winrate}%, chờ cơ hội phục thù'
                    })

            # 3. Badge Chuỗi Thắng (Streak)
            if streak_type == 'W' and cur_streak >= 3:
                badges.append({
                    'key': 'streak',
                    'label': f'Thắng Liên Tiếp x{cur_streak}',
                    'icon': '🔥',
                    'badge_class': 'bg-rose-100 text-rose-800 border-rose-300 font-bold',
                    'desc': f'Đang có chuỗi thắng {cur_streak} trận liên tiếp!'
                })

            # 4. Badge Ít thi đấu gần đây (áp dụng cho người có trận quá khứ nhưng lâu không đánh)
            eff_matches = p_rating.get('effective_matches', 0.0)
            if m_count >= 1 and eff_matches < 1.8:
                badges.append({
                    'key': 'inactive',
                    'label': f'Hao Mòn Thời Gian ({eff_matches:.1f} trận H.D)',
                    'icon': '⏳',
                    'badge_class': 'bg-amber-50 text-amber-700 border-amber-200 font-medium',
                    'desc': f'Ít thi đấu các trận gần đây (số trận hiệu dụng: {eff_matches:.1f}), điểm thực lực được co cụm về mức chuẩn'
                })

            form_info = {
                'score': round(power_score / 10.0, 1),
                'multiplier': 1.0,
                'status': 'on_fire' if (streak_type == 'W' and cur_streak >= 2) else ('cold' if (streak_type == 'L' and cur_streak >= 2) else 'neutral'),
                'icon': tier_icon,
                'label': tier_name,
                'streak': f"{streak_type}{cur_streak}" if cur_streak > 0 else 'N/A',
                'recent_5': recent_5,
                'recent_winrate': round(sum(1 for r in recent_5 if r == 'W') / len(recent_5) * 100, 1) if recent_5 else 0.0
            }

            impact_role = {
                'key': tier.lower(),
                'label': tier_name,
                'icon': tier_icon,
                'badge': tier_badge_class,
                'desc': tier_desc
            }

            player_obj = {
                'id': pid,
                'nickname': profile.get('nickname', pid.capitalize()),
                'avatar': profile.get('avatar', f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}"),
                'power_score': power_score,
                'tier': tier,
                'tier_name': tier_name,
                'tier_icon': tier_icon,
                'tier_stars': tier_stars,
                'tier_badge_class': tier_badge_class,
                'tier_desc': tier_desc,
                'rapm': p_rating.get('rapm', 0.0),
                'confidence': p_rating.get('confidence', 0.0),
                'effective_matches': eff_matches,
                'matches': m_count,
                'wins': wins,
                'losses': losses,
                'winrate': winrate,
                'form': form_info,
                'badges': badges,
                'impact_role': impact_role,
                # Khả năng tương thích ngược
                'hidden_elo': power_score,
                'elo_normalized': round(power_score / 10.0, 1),
                'combined_power': power_score,
                'effective_power': power_score,
                'stats_ovr': round(power_score / 10.0, 1),
                'skill': round(power_score / 10.0, 1),
                'champion_pool': 5.0,
                'flex_lane': 5.0,
                'consistency': 5.0,
                'primary_role': 'ALL'
            }
            results.append(player_obj)

        # Sắp xếp: Tuyển thủ có trận đấu xếp theo Điểm Thực Lực giảm dần, sau đó đến người chưa đấu
        results.sort(
            key=lambda x: (
                1 if x['matches'] > 0 else 0,
                x['power_score'],
                x['winrate'],
                x['matches']
            ),
            reverse=True
        )

        # Gán thứ hạng Rank và danh hiệu Top 3
        active_rank = 1
        for p in results:
            if p.get('matches', 0) > 0:
                if active_rank == 1:
                    p['badges'].insert(0, {
                        'key': 'top1',
                        'label': '👑 Quán Quân Server',
                        'icon': '👑',
                        'badge_class': 'bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 text-slate-900 border-amber-400 font-black shadow-2xs',
                        'desc': f'Số 1 toàn server với {p["power_score"]} Điểm Thực Lực'
                    })
                elif active_rank == 2:
                    p['badges'].insert(0, {
                        'key': 'top2',
                        'label': '🥈 Á Quân Server',
                        'icon': '🥈',
                        'badge_class': 'bg-slate-200 text-slate-800 border-slate-300 font-bold',
                        'desc': 'Số 2 toàn server'
                    })
                elif active_rank == 3:
                    p['badges'].insert(0, {
                        'key': 'top3',
                        'label': '🥉 Hạng Ba Server',
                        'icon': '🥉',
                        'badge_class': 'bg-amber-100 text-amber-800 border-amber-300 font-bold',
                        'desc': 'Top 3 server'
                    })
                p['rank'] = active_rank
                active_rank += 1
            else:
                p['rank'] = '-'

        return results

    def get_player(self, player_id: str) -> Optional[Dict[str, Any]]:
        pid = str(player_id).strip().lower()
        for p in self.get_all_players():
            if p['id'].lower() == pid:
                return p
        return None

    def create_player(self, data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Thêm tuyển thủ mới (chỉ cần ID, Nickname, Avatar)."""
        player_id = data.get('id', '').strip().lower()
        if not player_id:
            return False, "ID người chơi không được để trống", None

        profiles = data_manager.read_players_data()
        if any(k.lower() == player_id for k in profiles.keys()):
            return False, f"Người chơi '{player_id}' đã tồn tại", None

        nickname = data.get('nickname', '').strip() or player_id.capitalize()
        avatar = data.get('avatar', '').strip() or f"https://api.dicebear.com/7.x/bottts/svg?seed={player_id}"

        new_player = {
            'id': player_id,
            'nickname': nickname,
            'avatar': avatar
        }

        success = data_manager.save_single_player(new_player)
        if success:
            self.refresh_metrics()
            return True, "Thêm người chơi thành công", self.get_player(player_id)
        return False, "Lỗi khi lưu người chơi vào cơ sở dữ liệu", None

    def update_player(self, player_id: str, data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Cập nhật thông tin cơ bản của tuyển thủ (Nickname, Avatar)."""
        pid = str(player_id).strip().lower()
        profiles = data_manager.read_players_data()

        target_key = None
        for k in profiles.keys():
            if k.lower() == pid:
                target_key = k
                break

        if target_key is not None:
            cur = profiles[target_key].copy()
            if target_key != pid:
                data_manager.delete_single_player(target_key)
        else:
            cur = {'id': pid}

        cur['id'] = pid
        if 'nickname' in data and data['nickname']:
            cur['nickname'] = str(data['nickname']).strip()
        if 'avatar' in data and data['avatar']:
            cur['avatar'] = str(data['avatar']).strip()

        # Loại bỏ các trường kỹ năng cảm tính cũ nếu có
        for old_field in ['skill', 'champion_pool', 'flex_lane', 'consistency', 'primary_role', 'favorite_champions']:
            if old_field in cur:
                del cur[old_field]

        success = data_manager.save_single_player(cur)
        if success:
            self.refresh_metrics()
            return True, "Cập nhật thông tin thành công", self.get_player(pid)
        return False, "Lỗi khi cập nhật dữ liệu", None

    def delete_player(self, player_id: str) -> Tuple[bool, str]:
        """Xóa tuyển thủ."""
        pid = str(player_id).strip().lower()
        profiles = data_manager.read_players_data()
        target_keys = [k for k in profiles.keys() if k.lower() == pid]
        if target_keys:
            for k in target_keys:
                data_manager.delete_single_player(k)
            self.refresh_metrics()
            return True, f"Đã xóa hồ sơ người chơi '{pid}'"
        return False, f"Không tìm thấy người chơi '{pid}' trong danh sách hồ sơ"


player_service = PlayerService()
