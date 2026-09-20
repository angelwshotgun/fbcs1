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

        # Chỉ lấy các tuyển thủ thực tế đang tồn tại trong danh sách hồ sơ (profiles)
        # Tuyệt đối không tự động phục sinh các ID đã xóa chỉ vì có dữ liệu trong lịch sử đấu
        all_ids = sorted(list(profiles_lower.keys()))

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
            # === Thêm metrics mới ===
            feeder_count = 0
            great_count = 0
            solid_count = 0
            total_deaths = 0
            kda_games = 0
            stomp_losses = 0       # thua trong trận bị đè
            clutch_losses = 0     # thua trong trận sát nút
            total_delta_gained = 0.0
            total_delta_lost = 0.0
            big_win_delta_count = 0    # lần delta >= +20
            big_loss_delta_count = 0   # lần delta <= -18
            always_solid_or_better = True  # track nếu chưa bao giờ FEEDER/PASSENGER

            if match_details:
                for m_rec in match_details.values():
                    t1_players = [str(x).lower() for x in m_rec.get('team1', [])]
                    t2_players = [str(x).lower() for x in m_rec.get('team2', [])]
                    winner = m_rec.get('winner')
                    is_winner = (winner == 'team1' and pid in t1_players) or (winner == 'team2' and pid in t2_players)
                    in_match = pid in t1_players or pid in t2_players

                    if in_match:
                        is_stomp_m = m_rec.get('is_stomp', False)
                        closeness = float(m_rec.get('match_closeness', 0.5))
                        t1k = int(m_rec.get('team1_kills', 0))
                        t2k = int(m_rec.get('team2_kills', 0))
                        diff = abs(t1k - t2k) if (t1k > 0 or t2k > 0) else None

                        if is_winner:
                            if is_stomp_m or (diff is not None and diff >= 15):
                                stomp_wins += 1
                            elif closeness >= 0.75 or (diff is not None and diff <= 6):
                                clutch_wins += 1
                        else:
                            if is_stomp_m or (diff is not None and diff >= 15):
                                stomp_losses += 1
                            elif closeness >= 0.75 or (diff is not None and diff <= 6):
                                clutch_losses += 1

                    perfs = m_rec.get('player_performances', [])
                    if isinstance(perfs, list):
                        for pf in perfs:
                            if str(pf.get('player_id', '')).lower() == pid:
                                tag = str(pf.get('performance_tag', '')).upper()
                                delta_val = float(pf.get('recommended_delta', 0) or 0)

                                if tag in ['PASSENGER', 'CARRIED', 'FEEDER']:
                                    always_solid_or_better = False
                                if tag in ['PASSENGER', 'CARRIED']:
                                    passenger_tags_count += 1
                                elif tag in ['MVP', 'CARRY']:
                                    carry_tags_count += 1
                                if tag == 'MVP':
                                    mvp_count += 1
                                elif tag == 'SVP':
                                    svp_count += 1
                                elif tag == 'FEEDER':
                                    feeder_count += 1
                                elif tag == 'GREAT':
                                    great_count += 1
                                elif tag == 'SOLID':
                                    solid_count += 1

                                if delta_val > 0:
                                    total_delta_gained += delta_val
                                    if delta_val >= 20:
                                        big_win_delta_count += 1
                                elif delta_val < 0:
                                    total_delta_lost += abs(delta_val)
                                    if abs(delta_val) >= 18:
                                        big_loss_delta_count += 1

                                # Đọc số deaths từ KDA
                                kda_str = str(pf.get('kda', '') or '')
                                if '/' in kda_str:
                                    parts = kda_str.replace(' ', '').split('/')
                                    try:
                                        total_deaths += int(parts[1])
                                        kda_games += 1
                                    except Exception:
                                        pass

            # Tính trung bình deaths mỗi trận
            avg_deaths = round(total_deaths / kda_games, 1) if kda_games > 0 else None
            # Win streak / Loss streak từ recent_5
            recent_5 = form_info.get('recent_5', [])
            cur_win_streak = 0
            for r in reversed(recent_5):
                if r == 'W':
                    cur_win_streak += 1
                else:
                    break
            cur_loss_streak = 0
            for r in reversed(recent_5):
                if r == 'L':
                    cur_loss_streak += 1
                else:
                    break

            if m_count < 2:
                role_key = 'newbie'
                role_label = 'Mới Ra Lò'
                role_icon = '🐣'
                role_badge = 'bg-slate-100 text-slate-600 border-slate-200'
                role_desc = 'Tân binh đang bước vào đấu trường — ai cũng có lần đầu!'
            elif passenger_tags_count >= 2 or (wr_val >= 60.0 and stats_ovr < 4.8) or (passenger_tags_count >= 1 and stats_ovr < 5.0):
                role_key = 'passenger'
                role_label = 'Đi Nhờ Xe'
                role_icon = '🎒'
                role_badge = 'bg-amber-100 text-amber-800 border-amber-300'
                role_desc = 'Thắng nhiều nhờ đồng đội gánh, ngồi hưởng thành quả chiến thắng là chủ yếu'
            elif carry_tags_count >= 2 or (wr_val >= 60.0 and stats_ovr >= 6.5) or (carry_tags_count >= 1 and stats_ovr >= 7.0):
                role_key = 'carry'
                role_label = 'Gánh Team'
                role_icon = '💪'
                role_badge = 'bg-purple-100 text-purple-800 border-purple-300'
                role_desc = 'Người thực sự tạo ra chiến thắng, không có họ đội chịu chết'
            elif wr_val <= 35.0 and stats_ovr >= 6.8:
                role_key = 'unlucky'
                role_label = 'Gánh Không Nổi'
                role_icon = '😤'
                role_badge = 'bg-rose-100 text-rose-800 border-rose-300'
                role_desc = 'Chơi hay mà vẫn thua — đồng đội drag down quá nặng'
            else:
                role_key = 'core'
                role_label = 'Tròn Vai'
                role_icon = '⚖️'
                role_badge = 'bg-slate-100 text-slate-700 border-slate-200'
                role_desc = 'Không nổi bật nhưng đáng tin cậy — làm đủ việc, ít drama'

            impact_role = {
                'key': role_key,
                'label': role_label,
                'icon': role_icon,
                'badge': role_badge,
                'desc': role_desc,
                'passenger_count': passenger_tags_count,
                'carry_count': carry_tags_count
            }

            # Hệ thống Badges thực chiến (hoàn toàn dựa vào dữ liệu thi đấu thực tế, không dùng stats tĩnh)
            badges = []

            # 1. Danh hiệu Thực chiến cá nhân (AI Scoreboard / Trình diễn)
            if mvp_count >= 1:
                if mvp_count >= 5:
                    mvp_label = f'Huyền Thoại MVP x{mvp_count}'
                    mvp_desc = f'Đạt MVP {mvp_count} lần — đẳng cấp không cần bàn cãi'
                elif mvp_count >= 3:
                    mvp_label = f'Máy Gánh x{mvp_count} MVP'
                    mvp_desc = f'Gánh đội đến {mvp_count} lần rồi đấy bạn ơi'
                elif mvp_count == 2:
                    mvp_label = 'Siêu Sao x2 MVP'
                    mvp_desc = 'Đạt MVP 2 lần — không phải ngẫu nhiên đâu nhé'
                else:
                    mvp_label = '⭐ MVP'
                    mvp_desc = 'Từng tỏa sáng rực rỡ, một mình cân cả trận'
                badges.append({
                    'key': 'mvp',
                    'label': mvp_label,
                    'icon': '🏆',
                    'badge_class': 'bg-amber-100 text-amber-800 border-amber-300 font-bold',
                    'desc': mvp_desc
                })

            if svp_count >= 1:
                if svp_count >= 3:
                    svp_label = f'Chiến Binh Cô Đơn x{svp_count}'
                    svp_desc = f'Gánh tua mà vẫn thua {svp_count} lần — đồng đội nợ họ nhiều lắm'
                elif svp_count == 2:
                    svp_label = 'Anh Hùng Lỡ Vận x2'
                    svp_desc = 'Chơi hay nhưng đồng đội không đủ sức theo'
                else:
                    svp_label = 'Anh Hùng Thất Thế'
                    svp_desc = 'Cố gánh hết mình nhưng không cứu được — thôi thì lần sau'
                badges.append({
                    'key': 'svp',
                    'label': svp_label,
                    'icon': '🛡️',
                    'badge_class': 'bg-rose-100 text-rose-800 border-rose-300 font-bold',
                    'desc': svp_desc
                })

            # 2. Danh hiệu tương quan thế trận & tỉ số hạ gục (Kill Score & Match Closeness)
            if clutch_wins >= 1:
                if clutch_wins >= 4:
                    clutch_label = f'Vua Lật Kèo x{clutch_wins}'
                    clutch_desc = f'Lật ngược thế cờ {clutch_wins} lần — không bao giờ bỏ cuộc'
                elif clutch_wins >= 2:
                    clutch_label = f'Lội Ngược Dòng x{clutch_wins}'
                    clutch_desc = f'Giành chiến thắng sát nút {clutch_wins} lần, gan lì có tiếng'
                else:
                    clutch_label = 'Tim Đập Không Ngừng'
                    clutch_desc = 'Thắng trong trận sát nút rụng tim — đẳng cấp chịu áp lực'
                badges.append({
                    'key': 'clutch',
                    'label': clutch_label,
                    'icon': '🥊',
                    'badge_class': 'bg-emerald-100 text-emerald-800 border-emerald-300 font-bold',
                    'desc': clutch_desc
                })

            if stomp_wins >= 1:
                if stomp_wins >= 4:
                    stomp_label = f'Xe Lu Nghiền Nát x{stomp_wins}'
                    stomp_desc = f'Nghiền nát đối thủ {stomp_wins} lần, không cho cơ hội thở'
                elif stomp_wins >= 2:
                    stomp_label = f'Thảm Họa x{stomp_wins}'
                    stomp_desc = f'Đè bẹp đối thủ không thương tiếc {stomp_wins} trận liên tiếp'
                else:
                    stomp_label = 'Ăn Hiếp Bully'
                    stomp_desc = 'Thắng trận Stomp quá dễ — nhìn cứ như ăn hiếp con nít'
                badges.append({
                    'key': 'stomp',
                    'label': stomp_label,
                    'icon': '💥',
                    'badge_class': 'bg-red-100 text-red-800 border-red-300 font-bold',
                    'desc': stomp_desc
                })

            # 3. Danh hiệu Tỷ lệ thắng & Cống hiến thực chiến
            if m_count >= 3 and wr_val >= 70.0:
                if wr_val >= 85.0:
                    wr_label = f'Cheat Code ({round(wr_val)}%)'
                    wr_desc = f'Win rate {round(wr_val)}% — đây là người hay hay là bug vậy?'
                elif wr_val >= 75.0:
                    wr_label = f'Hard Carry ({round(wr_val)}% WR)'
                    wr_desc = f'Duy trì {round(wr_val)}% tỷ lệ thắng — khỏi phải nói nhiều'
                else:
                    wr_label = f'Win Machine ({round(wr_val)}%)'
                    wr_desc = f'Đăng ký thắng {round(wr_val)}% thời gian, ổn định đáng gờm'
                badges.append({
                    'key': 'high_wr',
                    'label': wr_label,
                    'icon': '💎',
                    'badge_class': 'bg-blue-100 text-blue-800 border-blue-300 font-bold',
                    'desc': wr_desc
                })
            elif m_count >= 6:
                if m_count >= 15:
                    vet_label = f'Lão Làng ({m_count} trận)'
                    vet_desc = f'Đã {m_count} trận — biết hết mặt anh em trong server rồi'
                elif m_count >= 10:
                    vet_label = f'Chiến Tướng ({m_count} trận)'
                    vet_desc = f'{m_count} trận chinh chiến, đã qua lửa nhiều lần'
                else:
                    vet_label = f'Quen Mặt ({m_count} trận)'
                    vet_desc = f'{m_count} trận rồi đó — đã bắt đầu biết mùi chiến trường'
                badges.append({
                    'key': 'veteran',
                    'label': vet_label,
                    'icon': '⚔️',
                    'badge_class': 'bg-slate-100 text-slate-800 border-slate-300 font-bold',
                    'desc': vet_desc
                })

            # 4. Danh hiệu Phong độ hiện tại (Streak)
            if cur_win_streak >= 3:
                if cur_win_streak >= 5:
                    badges.append({
                        'key': 'on_fire',
                        'label': f'Không Thể Cản Phá 🔥x{cur_win_streak}',
                        'icon': '🔥',
                        'badge_class': 'bg-orange-100 text-orange-800 border-orange-300 font-bold',
                        'desc': f'Thắng {cur_win_streak} trận liên tiếp — có ai dám đứng đối diện không?'
                    })
                else:
                    badges.append({
                        'key': 'on_fire',
                        'label': f'Đang Bùng Cháy 🔥x{cur_win_streak}',
                        'icon': '🔥',
                        'badge_class': 'bg-orange-100 text-orange-800 border-orange-300 font-bold',
                        'desc': f'Chuỗi thắng {cur_win_streak} trận — đang vào form đỉnh cao'
                    })
            elif cur_loss_streak >= 3:
                if cur_loss_streak >= 5:
                    badges.append({
                        'key': 'cold_streak',
                        'label': f'Thua Không Biết Chán ❄️x{cur_loss_streak}',
                        'icon': '❄️',
                        'badge_class': 'bg-sky-100 text-sky-800 border-sky-300 font-bold',
                        'desc': f'Thua {cur_loss_streak} trận liên tiếp — thôi nghỉ ra ngoài hít thở đi bạn ơi'
                    })
                else:
                    badges.append({
                        'key': 'cold_streak',
                        'label': f'Đang Đóng Băng ❄️x{cur_loss_streak}',
                        'icon': '❄️',
                        'badge_class': 'bg-sky-100 text-sky-800 border-sky-300 font-bold',
                        'desc': f'Chuỗi thua {cur_loss_streak} trận — cần nghỉ ngơi hoặc đổi đồng đội'
                    })

            # 5. Danh hiệu Số Deaths cá nhân
            if avg_deaths is not None and kda_games >= 3:
                if avg_deaths <= 1.5:
                    badges.append({
                        'key': 'unkillable',
                        'label': f'Khó Chết ({avg_deaths} deaths/trận)',
                        'icon': '🧊',
                        'badge_class': 'bg-cyan-100 text-cyan-800 border-cyan-300 font-bold',
                        'desc': f'Chỉ chết trung bình {avg_deaths} lần/trận — bản năng sống sót siêu phàm'
                    })
                elif avg_deaths >= 7.0:
                    badges.append({
                        'key': 'inted',
                        'label': f'Quà Tặng Địch ({avg_deaths} deaths/trận)',
                        'icon': '🎁',
                        'badge_class': 'bg-pink-100 text-pink-800 border-pink-300 font-bold',
                        'desc': f'Chết trung bình {avg_deaths} lần/trận — phát lì xì cho địch hơi nhiều rồi đó'
                    })

            # 6. Danh hiệu chuyên cho mạng (FEEDER)
            if feeder_count >= 2:
                if feeder_count >= 4:
                    badges.append({
                        'key': 'feeder',
                        'label': f'Nhà Tài Trợ Địch x{feeder_count}',
                        'icon': '💸',
                        'badge_class': 'bg-red-100 text-red-900 border-red-400 font-bold',
                        'desc': f'Bị tag FEEDER {feeder_count} lần — địch cảm ơn nhiều lắm'
                    })
                else:
                    badges.append({
                        'key': 'feeder',
                        'label': f'Chia Mạng Hào Phóng x{feeder_count}',
                        'icon': '💸',
                        'badge_class': 'bg-red-100 text-red-800 border-red-300 font-bold',
                        'desc': f'FEEDER {feeder_count} lần — cần xem lại bản đồ mini map một chút'
                    })

            # 7. Danh hiệu Tuyển thủ toàn diện (GREAT liên tiếp)
            if great_count >= 3 and mvp_count == 0:
                badges.append({
                    'key': 'allrounder',
                    'label': f'Tuyển Thủ Toàn Diện ({great_count} GREAT)',
                    'icon': '🌟',
                    'badge_class': 'bg-violet-100 text-violet-800 border-violet-300 font-bold',
                    'desc': f'GREAT {great_count} lần — không flashy nhưng luôn đóng góp đỉnh cao'
                })

            # 8. Danh hiệu Đáng Tin Cậy (chưa bao giờ FEEDER/PASSENGER)
            if m_count >= 4 and always_solid_or_better:
                badges.append({
                    'key': 'reliable',
                    'label': 'Không Bao Giờ Thọt',
                    'icon': '🎯',
                    'badge_class': 'bg-teal-100 text-teal-800 border-teal-300 font-bold',
                    'desc': f'Chưa một lần bị tag FEEDER hay PASSENGER trong {m_count} trận — đáng tin cậy nhất server'
                })

            # 9. Danh hiệu Elo Turbulence (delta lên xuống dữ dội)
            if big_win_delta_count >= 2 and big_loss_delta_count >= 2:
                badges.append({
                    'key': 'volatile',
                    'label': 'Elo Tàu Lượn Siêu Tốc',
                    'icon': '🎢',
                    'badge_class': 'bg-fuchsia-100 text-fuchsia-800 border-fuchsia-300 font-bold',
                    'desc': f'Elo lúc lên +20, lúc xuống -18 — theo dõi bảng xếp hạng của người này là thú vị nhất'
                })

            # 10. Danh hiệu Nạn Nhân Stomp (bị đè nhiều lần)
            if stomp_losses >= 3:
                badges.append({
                    'key': 'stomped',
                    'label': f'Bị Bắt Nạt x{stomp_losses}',
                    'icon': '🫠',
                    'badge_class': 'bg-slate-200 text-slate-700 border-slate-400 font-bold',
                    'desc': f'Thua trận Stomp {stomp_losses} lần — đồng đội toàn kéo vào trận đấu một chiều'
                })

            # 11. Danh hiệu Kẻ Lỡ Tay (thua sát nút nhiều lần)
            if clutch_losses >= 3:
                badges.append({
                    'key': 'heartbreak',
                    'label': f'Hay Lỡ x{clutch_losses}',
                    'icon': '💔',
                    'badge_class': 'bg-rose-50 text-rose-700 border-rose-300 font-bold',
                    'desc': f'Thua {clutch_losses} trận sát nút — cứ thấy có cơ hội là lại hụt, nghe đau lắm'
                })

            # 12. Danh hiệu người gánh Elo nhiều nhất
            if total_delta_gained >= 80 and m_count >= 4:
                badges.append({
                    'key': 'elo_farmer',
                    'label': f'Cày Elo +{round(total_delta_gained)}',
                    'icon': '📈',
                    'badge_class': 'bg-green-100 text-green-800 border-green-300 font-bold',
                    'desc': f'Tổng cộng cày được +{round(total_delta_gained)} Elo — máy in tiền thực sự'
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
                        'label': '👑 Trùm Cuối',
                        'icon': '👑',
                        'badge_class': 'bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 text-slate-900 border-amber-400 font-black shadow-2xs',
                        'desc': 'Đứng đầu bảng Elo toàn server — ai muốn đánh thì cứ lên!'
                    })
                elif active_rank == 2:
                    p['badges'].insert(0, {
                        'key': 'podium',
                        'label': '🥈 Á Quân',
                        'icon': '🥈',
                        'badge_class': 'bg-indigo-100 text-indigo-800 border-indigo-300 font-bold',
                        'desc': 'Số 2 toàn server — gần tới đỉnh lắm rồi, cố thêm tí!'
                    })
                elif active_rank == 3:
                    p['badges'].insert(0, {
                        'key': 'podium',
                        'label': '🥉 Hạng Ba',
                        'icon': '🥉',
                        'badge_class': 'bg-amber-50 text-amber-700 border-amber-300 font-bold',
                        'desc': 'Top 3 server — vào podium rồi, còn ai dám nói gì nào!'
                    })
                elif p.get('matches', 0) <= 2 and (p.get('winrate', 0) >= 60.0 or p.get('hidden_elo', 0) >= 1230.0):
                    p['badges'].insert(0, {
                        'key': 'rising_star',
                        'label': '⚡ Ngôi Sao Mới',
                        'icon': '⚡',
                        'badge_class': 'bg-sky-100 text-sky-800 border-sky-300 font-bold',
                        'desc': 'Mới vào đã nổi rồi — dân này có số không đùa được đâu'
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
            self._cached_metrics = None
            self.refresh_metrics()
            return True, f"Đã xóa hồ sơ người chơi '{pid}'"
        return False, f"Không tìm thấy người chơi '{pid}' trong danh sách hồ sơ"


player_service = PlayerService()
