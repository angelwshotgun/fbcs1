from typing import Dict, List, Any, Optional, Tuple
from services.data_manager import data_manager
from services.elo_service import elo_service


class PlayerService:
    def __init__(self):
        self._cached_metrics = None

    def refresh_metrics(self) -> Dict[str, Any]:
        """Tính toán lại toàn bộ metrics từ match_data.csv và match_details.json."""
        df = data_manager.read_matches_df()
        match_details = data_manager.read_match_details()
        profiles = data_manager.read_players_data()
        self._cached_metrics = elo_service.calculate_all_metrics(
            df,
            match_details=match_details,
            players_profiles=profiles
        )
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
        match_details = data_manager.read_match_details()

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

            # Stats cơ bản (thang điểm 1 - 10, mặc định 5.0)
            skill = float(profile.get('skill', 5.0))
            champ_pool = float(profile.get('champion_pool', 5.0))
            flex_lane = float(profile.get('flex_lane', 5.0))
            consistency = float(profile.get('consistency', 5.0))

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

            # Tính toán vai trò thực chiến và thu thập dữ liệu chi tiết từ lịch sử trận
            m_count = p_stats.get('matches', 0)
            wr_val = p_stats.get('winrate', 0.0)
            passenger_tags_count = 0
            carry_tags_count = 0
            mvp_count = 0
            svp_count = 0
            clutch_wins = 0
            stomp_wins = 0

            if match_details:
                for m_rec in match_details.values():
                    t1_players = [str(x).lower() for x in m_rec.get('team1', [])]
                    t2_players = [str(x).lower() for x in m_rec.get('team2', [])]
                    winner = m_rec.get('winner')
                    is_winner = (winner == 'team1' and pid in t1_players) or (winner == 'team2' and pid in t2_players)

                    if is_winner:
                        is_stomp = m_rec.get('is_stomp', False)
                        closeness = float(m_rec.get('match_closeness', 0.5))
                        t1k = int(m_rec.get('team1_kills', 0))
                        t2k = int(m_rec.get('team2_kills', 0))
                        diff = abs(t1k - t2k) if (t1k > 0 or t2k > 0) else None

                        if is_stomp or (diff is not None and diff >= 15):
                            stomp_wins += 1
                        elif closeness >= 0.75 or (diff is not None and diff <= 6):
                            clutch_wins += 1

                    perfs = m_rec.get('player_performances', [])
                    if isinstance(perfs, list):
                        for pf in perfs:
                            if str(pf.get('player_id', '')).lower() == pid:
                                tag = str(pf.get('performance_tag', '')).upper()
                                if tag in ['PASSENGER', 'CARRIED']:
                                    passenger_tags_count += 1
                                elif tag in ['MVP', 'CARRY']:
                                    carry_tags_count += 1
                                if tag == 'MVP':
                                    mvp_count += 1
                                elif tag == 'SVP':
                                    svp_count += 1

            if m_count < 2:
                role_key = 'newbie'
                role_label = 'Tân binh'
                role_icon = '🌱'
                role_badge = 'bg-slate-100 text-slate-600 border-slate-200'
                role_desc = 'Chưa đủ số trận để đánh giá vai trò'
            elif passenger_tags_count >= 2 or (wr_val >= 60.0 and stats_ovr < 4.8) or (passenger_tags_count >= 1 and stats_ovr < 5.0):
                role_key = 'passenger'
                role_label = 'Hưởng ké'
                role_icon = '🎒'
                role_badge = 'bg-amber-100 text-amber-800 border-amber-300'
                role_desc = 'Thắng nhiều nhờ được đồng đội gánh, đóng góp cá nhân hạn chế'
            elif carry_tags_count >= 2 or (wr_val >= 60.0 and stats_ovr >= 6.5) or (carry_tags_count >= 1 and stats_ovr >= 7.0):
                role_key = 'carry'
                role_label = 'Chủ lực'
                role_icon = '👑'
                role_badge = 'bg-purple-100 text-purple-800 border-purple-300'
                role_desc = 'Người dẫn dắt lối chơi, đóng góp trực tiếp vào chiến thắng'
            elif wr_val <= 35.0 and stats_ovr >= 6.8:
                role_key = 'unlucky'
                role_label = 'Gánh tạ'
                role_icon = '🛡️'
                role_badge = 'bg-rose-100 text-rose-800 border-rose-300'
                role_desc = 'Kỹ năng cao nhưng thường xuyên gánh đồng đội yếu thế'
            else:
                role_key = 'core'
                role_label = 'Trụ cột'
                role_icon = '⚖️'
                role_badge = 'bg-slate-100 text-slate-700 border-slate-200'
                role_desc = 'Đóng góp ổn định, tròn vai trong các trận đấu'

            impact_role = {
                'key': role_key,
                'label': role_label,
                'icon': role_icon,
                'badge': role_badge,
                'desc': role_desc,
                'passenger_count': passenger_tags_count,
                'carry_count': carry_tags_count
            }

            # Hệ thống Badges đa dạng
            badges = []

            # 1. Danh hiệu Thực chiến
            if mvp_count >= 1:
                badges.append({
                    'key': 'mvp',
                    'label': f'MVP x{mvp_count}' if mvp_count > 1 else 'MVP',
                    'icon': '🏆',
                    'badge_class': 'bg-amber-100 text-amber-800 border-amber-300 font-bold',
                    'desc': f'Tỏa sáng rực rỡ và đạt danh hiệu MVP {mvp_count} lần'
                })

            if svp_count >= 1:
                badges.append({
                    'key': 'svp',
                    'label': f'SVP x{svp_count}' if svp_count > 1 else 'SVP',
                    'icon': '🛡️',
                    'badge_class': 'bg-rose-100 text-rose-800 border-rose-300 font-bold',
                    'desc': f'Chiến binh đơn độc gánh đội thua {svp_count} lần'
                })

            if clutch_wins >= 1:
                badges.append({
                    'key': 'clutch',
                    'label': f'Lội Ngược Dòng x{clutch_wins}' if clutch_wins > 1 else 'Lội Ngược Dòng',
                    'icon': '🥊',
                    'badge_class': 'bg-emerald-100 text-emerald-800 border-emerald-300 font-bold',
                    'desc': f'Bản lĩnh giành chiến thắng trong {clutch_wins} trận sát nút căng thẳng'
                })

            if stomp_wins >= 1:
                badges.append({
                    'key': 'stomp',
                    'label': f'Hủy Diệt x{stomp_wins}' if stomp_wins > 1 else 'Hủy Diệt',
                    'icon': '💥',
                    'badge_class': 'bg-red-100 text-red-800 border-red-300 font-bold',
                    'desc': f'Đè bẹp đối thủ áp đảo trong {stomp_wins} trận Stomp'
                })

            # 2. Danh hiệu Chuỗi Thắng / Thua & Phong độ
            streak_str = form_info.get('streak', '')
            if streak_str.startswith('W'):
                try:
                    s_num = int(streak_str[1:])
                    if s_num >= 4:
                        badges.append({
                            'key': 'streak_god',
                            'label': f'Bất Bại {s_num}W',
                            'icon': '⚡',
                            'badge_class': 'bg-gradient-to-r from-amber-500 to-orange-500 text-white border-amber-600 shadow-2xs font-black',
                            'desc': f'Đang giữ chuỗi toàn thắng {s_num} trận liên tiếp'
                        })
                    elif s_num >= 2:
                        badges.append({
                            'key': 'streak_w',
                            'label': f'Chuỗi {s_num}W',
                            'icon': '🔥',
                            'badge_class': 'bg-amber-50 text-amber-700 border-amber-200 font-bold',
                            'desc': f'Hưng phấn với chuỗi thắng {s_num} trận'
                        })
                except Exception:
                    pass
            elif streak_str.startswith('L'):
                try:
                    s_num = int(streak_str[1:])
                    if s_num >= 3:
                        badges.append({
                            'key': 'streak_l',
                            'label': f'Giải Hạn {s_num}L',
                            'icon': '🧊',
                            'badge_class': 'bg-slate-100 text-slate-600 border-slate-300',
                            'desc': f'Chuỗi thua {s_num} trận không may, cần người kéo lại'
                        })
                except Exception:
                    pass

            # 3. Danh hiệu Tố chất & Kỹ năng
            if skill >= 8.5:
                badges.append({
                    'key': 'mech_god',
                    'label': 'Tay To',
                    'icon': '🎯',
                    'badge_class': 'bg-teal-100 text-teal-800 border-teal-300 font-bold',
                    'desc': 'Kỹ năng cá nhân vượt trội (Skill >= 8.5/10)'
                })

            if champ_pool >= 8.0:
                badges.append({
                    'key': 'deep_pool',
                    'label': 'Kho Tướng',
                    'icon': '📚',
                    'badge_class': 'bg-violet-100 text-violet-800 border-violet-300 font-bold',
                    'desc': 'Bể tướng rất rộng (Champion Pool >= 8.0/10)'
                })

            if flex_lane >= 8.0:
                badges.append({
                    'key': 'flex_god',
                    'label': 'Tắc Kè Hoa',
                    'icon': '🔄',
                    'badge_class': 'bg-cyan-100 text-cyan-800 border-cyan-300 font-bold',
                    'desc': 'Linh hoạt mọi làn đường (Flex Lane >= 8.0/10)'
                })

            if consistency >= 8.0:
                badges.append({
                    'key': 'iron_anchor',
                    'label': 'Hòn Đá Tảng',
                    'icon': '⚓',
                    'badge_class': 'bg-slate-100 text-slate-800 border-slate-300 font-bold',
                    'desc': 'Thi đấu ổn định tuyệt đối (Consistency >= 8.0/10)'
                })

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
                'impact_role': impact_role,
                'badges': badges,
                'matches': p_stats.get('matches', 0),
                'wins': p_stats.get('wins', 0),
                'losses': p_stats.get('losses', 0),
                'winrate': p_stats.get('winrate', 0.0)
            }
            results.append(player_obj)

        results.sort(key=lambda x: (x['hidden_elo'], x['effective_power']), reverse=True)

        # Gán danh hiệu Xếp Hạng dựa trên thứ tự BXH
        active_rank = 1
        for p in results:
            if p.get('matches', 0) > 0:
                if active_rank == 1:
                    p['badges'].insert(0, {
                        'key': 'top1',
                        'label': 'Bá Chủ BXH',
                        'icon': '👑',
                        'badge_class': 'bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 text-slate-900 border-amber-400 font-black shadow-2xs',
                        'desc': 'Đang nắm giữ vị trí Số 1 trên Bảng Xếp Hạng Elo toàn máy chủ'
                    })
                elif active_rank in [2, 3]:
                    p['badges'].insert(0, {
                        'key': 'podium',
                        'label': f'Top {active_rank}',
                        'icon': '🥈' if active_rank == 2 else '🥉',
                        'badge_class': 'bg-indigo-100 text-indigo-800 border-indigo-300 font-bold',
                        'desc': f'Tuyển thủ thuộc Top {active_rank} máy chủ'
                    })
                elif p.get('matches', 0) <= 2 and (p.get('winrate', 0) >= 60.0 or p.get('hidden_elo', 0) >= 1230.0):
                    p['badges'].insert(0, {
                        'key': 'rising_star',
                        'label': 'Tân Binh Quái Kiệt',
                        'icon': '⭐',
                        'badge_class': 'bg-sky-100 text-sky-800 border-sky-300 font-bold',
                        'desc': 'Tân binh khởi đầu xuất sắc với tỷ lệ thắng và Elo cao'
                    })
                p['rank'] = active_rank
                active_rank += 1

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

        # Khởi tạo thông số chuẩn hóa thang 1-10 (mặc định 5.0)
        nickname = data.get('nickname', '').strip() or player_id.capitalize()
        avatar = data.get('avatar', '').strip() or f"https://api.dicebear.com/7.x/bottts/svg?seed={player_id}"
        skill = max(1.0, min(10.0, float(data.get('skill', 5.0))))
        champ_pool = max(1.0, min(10.0, float(data.get('champion_pool', 5.0))))
        flex_lane = max(1.0, min(10.0, float(data.get('flex_lane', 5.0))))
        consistency = max(1.0, min(10.0, float(data.get('consistency', 5.0))))
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
