import os
import json
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
                'rank_change': p_rating.get('rank_change', 0),
                'active_rank_change': p_rating.get('active_rank_change', 0),
                'prev_global_rank': p_rating.get('prev_global_rank'),
                'prev_active_rank': p_rating.get('prev_active_rank'),
                'power_delta': p_rating.get('power_delta', 0.0),
                'prev_power_score': p_rating.get('prev_power_score'),
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

    def get_player_details(self, player_id: str, force_ai_refresh: bool = False, custom_key: str = None) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin chi tiết toàn diện của tuyển thủ:
        - Hồ sơ năng lực (Power Score, RAPM, Rank, Badges)
        - Toàn bộ lịch sử các trận đấu đã tham gia
        - Thống kê phe Xanh / Đỏ, chuỗi thắng/thua, stomp, trận căng thẳng
        - Cặp đôi ăn ý (Top Synergies) & Đối thủ duyên nợ (Nemesis)
        - Điểm đa chiều cho biểu đồ Radar & Trend line
        - Hồ sơ AI cá nhân hóa (Gemini AI analysis)
        """
        pid = str(player_id).strip().lower()
        player = self.get_player(pid)
        if not player:
            return None

        # Lấy thông tin mở rộng từ hồ sơ gốc (nếu có role, tướng yêu thích)
        profiles = data_manager.read_players_data()
        raw_profile = {}
        for k, v in profiles.items():
            if str(k).strip().lower() == pid:
                raw_profile = v
                break

        primary_role = raw_profile.get('primary_role', 'ALL')
        favorite_champions = raw_profile.get('favorite_champions', [])

        all_players = self.get_all_players()
        players_map = {str(p['id']).strip().lower(): p for p in all_players}

        all_matches = data_manager.get_matches_history()

        player_matches = []
        teammate_stats = {}
        rival_stats = {}

        blue_matches = 0
        blue_wins = 0
        red_matches = 0
        red_wins = 0

        for m in all_matches:
            t1 = [str(x).strip().lower() for x in m.get('team1_players', [])]
            t2 = [str(x).strip().lower() for x in m.get('team2_players', [])]

            in_t1 = pid in t1
            in_t2 = pid in t2

            if not in_t1 and not in_t2:
                continue

            player_side = 'team1' if in_t1 else 'team2'
            winner = m.get('winner', 'team1')
            is_winner = (winner == player_side)

            if player_side == 'team1':
                blue_matches += 1
                if is_winner:
                    blue_wins += 1
                teammate_ids = [p for p in t1 if p != pid]
                opponent_ids = t2
            else:
                red_matches += 1
                if is_winner:
                    red_wins += 1
                teammate_ids = [p for p in t2 if p != pid]
                opponent_ids = t1

            for tm_id in teammate_ids:
                if tm_id not in teammate_stats:
                    teammate_stats[tm_id] = {'matches': 0, 'wins': 0, 'losses': 0}
                teammate_stats[tm_id]['matches'] += 1
                if is_winner:
                    teammate_stats[tm_id]['wins'] += 1
                else:
                    teammate_stats[tm_id]['losses'] += 1

            for op_id in opponent_ids:
                if op_id not in rival_stats:
                    rival_stats[op_id] = {'matches': 0, 'wins_against': 0, 'losses_against': 0}
                rival_stats[op_id]['matches'] += 1
                if is_winner:
                    rival_stats[op_id]['wins_against'] += 1
                else:
                    rival_stats[op_id]['losses_against'] += 1

            teammate_objs = [
                players_map.get(tid, {
                    'id': tid,
                    'nickname': tid.capitalize(),
                    'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={tid}",
                    'tier_name': 'Tier B',
                    'tier_icon': '🛡️',
                    'power_score': 50.0
                }) for tid in teammate_ids
            ]
            opponent_objs = [
                players_map.get(oid, {
                    'id': oid,
                    'nickname': oid.capitalize(),
                    'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={oid}",
                    'tier_name': 'Tier B',
                    'tier_icon': '🛡️',
                    'power_score': 50.0
                }) for oid in opponent_ids
            ]

            t1_k = m.get('team1_kills', 0)
            t2_k = m.get('team2_kills', 0)
            player_kills = t1_k if player_side == 'team1' else t2_k
            opponent_kills = t2_k if player_side == 'team1' else t1_k

            player_matches.append({
                'id': m.get('id'),
                'match_code': m.get('match_code', f"M-{m.get('id')}"),
                'created_at': m.get('created_at', ''),
                'player_side': player_side,
                'winner': winner,
                'is_winner': is_winner,
                'player_kills': player_kills,
                'opponent_kills': opponent_kills,
                'team1_kills': t1_k,
                'team2_kills': t2_k,
                'team1_power': m.get('team1_power', 0.0),
                'team2_power': m.get('team2_power', 0.0),
                'match_closeness': m.get('match_closeness', 0.5),
                'is_stomp': m.get('is_stomp', False),
                'balance_rating': m.get('balance_rating', 'unknown'),
                'ai_summary': m.get('ai_summary', ''),
                'notes': m.get('notes', ''),
                'teammates': teammate_objs,
                'opponents': opponent_objs
            })

        total_matches = len(player_matches)
        wins = sum(1 for m in player_matches if m['is_winner'])
        losses = total_matches - wins
        winrate = round((wins / total_matches) * 100, 1) if total_matches > 0 else 0.0

        # Chuỗi thắng/thua hiện tại
        cur_streak_type = 'W'
        cur_streak_count = 0
        if player_matches:
            cur_streak_type = 'W' if player_matches[0]['is_winner'] else 'L'
            for m in player_matches:
                if (m['is_winner'] and cur_streak_type == 'W') or (not m['is_winner'] and cur_streak_type == 'L'):
                    cur_streak_count += 1
                else:
                    break

        # Chuỗi thắng dài nhất & chuỗi thua dài nhất
        longest_win_streak = 0
        longest_loss_streak = 0
        temp_w = 0
        temp_l = 0
        for m in reversed(player_matches):
            if m['is_winner']:
                temp_w += 1
                temp_l = 0
                longest_win_streak = max(longest_win_streak, temp_w)
            else:
                temp_l += 1
                temp_w = 0
                longest_loss_streak = max(longest_loss_streak, temp_l)

        # Danh sách Đồng Đội ăn ý
        best_teammates = []
        for tid, s in teammate_stats.items():
            tm_obj = players_map.get(tid, {'id': tid, 'nickname': tid.capitalize(), 'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={tid}"})
            wr = round((s['wins'] / s['matches']) * 100, 1) if s['matches'] > 0 else 0.0
            best_teammates.append({
                'id': tid,
                'nickname': tm_obj.get('nickname', tid.capitalize()),
                'avatar': tm_obj.get('avatar', f"https://api.dicebear.com/7.x/bottts/svg?seed={tid}"),
                'tier_name': tm_obj.get('tier_name', 'Tier B'),
                'tier_icon': tm_obj.get('tier_icon', '🛡️'),
                'matches': s['matches'],
                'wins': s['wins'],
                'losses': s['losses'],
                'winrate': wr
            })
        best_teammates.sort(key=lambda x: (1 if x['matches'] >= 2 else 0, x['winrate'], x['matches']), reverse=True)

        # Danh sách Đối Thủ Duyên Nợ (Rivals)
        rivals = []
        for oid, s in rival_stats.items():
            op_obj = players_map.get(oid, {'id': oid, 'nickname': oid.capitalize(), 'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={oid}"})
            wr_against = round((s['wins_against'] / s['matches']) * 100, 1) if s['matches'] > 0 else 0.0
            rivals.append({
                'id': oid,
                'nickname': op_obj.get('nickname', oid.capitalize()),
                'avatar': op_obj.get('avatar', f"https://api.dicebear.com/7.x/bottts/svg?seed={oid}"),
                'tier_name': op_obj.get('tier_name', 'Tier B'),
                'tier_icon': op_obj.get('tier_icon', '🛡️'),
                'matches': s['matches'],
                'wins_against': s['wins_against'],
                'losses_against': s['losses_against'],
                'winrate_against': wr_against
            })
        rivals.sort(key=lambda x: (x['matches'], 100 - x['winrate_against']), reverse=True)

        # Trend điểm & tỷ lệ thắng qua tối đa 15 trận gần nhất (theo thứ tự thời gian từ cũ -> mới)
        trend_matches = list(reversed(player_matches[:15]))
        trend_data = []
        running_w = 0
        for idx, tm in enumerate(trend_matches):
            if tm['is_winner']:
                running_w += 1
            running_wr = round((running_w / (idx + 1)) * 100, 1)
            trend_data.append({
                'match_index': idx + 1,
                'match_code': tm['match_code'],
                'result': 'W' if tm['is_winner'] else 'L',
                'running_winrate': running_wr,
                'created_at': tm['created_at']
            })

        # Radar Chart Metrics (chuẩn hóa 0-100)
        power_score = float(player.get('power_score', 50.0))
        recent_5 = player.get('recent_5', [])
        form_points = sum(20 for r in recent_5 if r == 'W')
        if cur_streak_type == 'W':
            form_points = min(100, form_points + min(15, cur_streak_count * 5))
        elif cur_streak_type == 'L':
            form_points = max(10, form_points - min(15, cur_streak_count * 5))
        form_score = max(15, min(100, form_points))

        exp_score = min(100.0, round(total_matches * 4.2, 1))
        impact_score = max(15.0, min(100.0, round(50.0 + (float(player.get('rapm', 0.0)) * 14.0) + (winrate - 50.0) * 0.4, 1)))

        synergy_pool = [t['winrate'] for t in best_teammates if t['matches'] >= 2]
        synergy_score = round(sum(synergy_pool) / len(synergy_pool), 1) if synergy_pool else winrate

        radar_scores = {
            'power': power_score,
            'winrate': float(winrate),
            'form': float(form_score),
            'experience': float(exp_score),
            'impact': float(impact_score),
            'synergy': float(synergy_score)
        }

        stats_breakdown = {
            'total_matches': total_matches,
            'wins': wins,
            'losses': losses,
            'winrate': winrate,
            'side_stats': {
                'team1': {
                    'name': 'Đội Xanh (Team 1)',
                    'matches': blue_matches,
                    'wins': blue_wins,
                    'losses': blue_matches - blue_wins,
                    'winrate': round((blue_wins / blue_matches) * 100, 1) if blue_matches > 0 else 0.0
                },
                'team2': {
                    'name': 'Đội Đỏ (Team 2)',
                    'matches': red_matches,
                    'wins': red_wins,
                    'losses': red_matches - red_wins,
                    'winrate': round((red_wins / red_matches) * 100, 1) if red_matches > 0 else 0.0
                }
            },
            'current_streak': {
                'type': cur_streak_type,
                'count': cur_streak_count
            },
            'longest_win_streak': longest_win_streak,
            'longest_loss_streak': longest_loss_streak,
            'stomp_stats': {
                'wins': sum(1 for m in player_matches if m['is_winner'] and m['is_stomp']),
                'losses': sum(1 for m in player_matches if not m['is_winner'] and m['is_stomp'])
            },
            'close_matches': {
                'total': sum(1 for m in player_matches if m.get('match_closeness', 0.5) >= 0.7),
                'wins': sum(1 for m in player_matches if m['is_winner'] and m.get('match_closeness', 0.5) >= 0.7)
            },
            'best_teammates': best_teammates,
            'rivals': rivals,
            'radar_scores': radar_scores,
            'trend_data': trend_data,
            'primary_role': primary_role,
            'favorite_champions': favorite_champions
        }

        # Quản lý bộ đệm AI Profile
        ai_profiles = _load_ai_profiles()
        ai_analysis = ai_profiles.get(pid)

        if not ai_analysis or force_ai_refresh:
            from services.gemini_service import gemini_service
            ai_analysis = gemini_service.analyze_player_profile(
                player_data=player,
                stats_data=stats_breakdown,
                custom_key=custom_key
            )
            _save_ai_profile(pid, ai_analysis)

        return {
            'player': player,
            'stats': stats_breakdown,
            'matches': player_matches,
            'ai_analysis': ai_analysis
        }


AI_PROFILES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data',
    'ai_player_profiles.json'
)


def _load_ai_profiles() -> Dict[str, Any]:
    if os.path.exists(AI_PROFILES_FILE):
        try:
            with open(AI_PROFILES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_ai_profile(player_id: str, profile_data: Dict[str, Any]):
    try:
        profiles = _load_ai_profiles()
        profiles[str(player_id).strip().lower()] = profile_data
        with open(AI_PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[PlayerService] Error saving AI profile: {e}")


player_service = PlayerService()

