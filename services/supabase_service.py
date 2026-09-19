import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


class SupabaseService:
    def __init__(self):
        self._reload_config()

    def _reload_config(self):
        load_dotenv()
        self.url = (os.getenv('SUPABASE_URL') or os.getenv('NEXT_PUBLIC_SUPABASE_URL') or '').strip().rstrip('/')
        self.key = (
            os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            or os.getenv('SUPABASE_KEY')
            or os.getenv('SUPABASE_ANON_KEY')
            or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
            or ''
        ).strip()

    def is_configured(self) -> bool:
        if not (self.url and self.key):
            self._reload_config()
        return bool(self.url and self.key)

    def _get_headers(self, prefer: Optional[str] = None) -> Dict[str, str]:
        if not self.key:
            self._reload_config()
        headers = {
            'apikey': self.key,
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if prefer:
            headers['Prefer'] = prefer
        return headers

    def _request(
        self,
        endpoint: str,
        method: str = 'GET',
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        prefer: Optional[str] = None,
        timeout: int = 10
    ) -> Tuple[bool, Any]:
        """Thực hiện HTTP request tới Supabase PostgREST API."""
        if not self.is_configured():
            return False, "Supabase chưa được cấu hình (thiếu URL hoặc KEY)"

        url = f"{self.url}/rest/v1/{endpoint.lstrip('/')}"
        if params:
            query_str = urllib.parse.urlencode(params)
            url = f"{url}?{query_str}"

        headers = self._get_headers(prefer)
        body = None
        if data is not None:
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                resp_data = response.read().decode('utf-8')
                if resp_data:
                    try:
                        parsed = json.loads(resp_data)
                        return True, parsed
                    except json.JSONDecodeError:
                        return True, resp_data
                return True, {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            return False, f"HTTP {e.code}: {err_body}"
        except urllib.error.URLError as e:
            return False, f"Network Error: {str(e.reason)}"
        except Exception as e:
            return False, str(e)

    # ==========================================
    # KIỂM TRA TRẠNG THÁI KẾT NỐI
    # ==========================================
    def test_connection(self) -> Dict[str, Any]:
        """Kiểm tra kết nối và số lượng bản ghi trong các bảng Supabase."""
        if not self.is_configured():
            return {
                'connected': False,
                'message': 'Chưa cấu hình SUPABASE_URL hoặc SUPABASE_KEY trong file .env',
                'players_count': 0,
                'matches_count': 0
            }

        # 1. Kiểm tra bảng players
        ok_p, res_p = self._request('players', method='GET', params={'select': 'id', 'limit': '1000'})
        if not ok_p:
            return {
                'connected': False,
                'message': f"Lỗi kết nối bảng players: {res_p}",
                'players_count': 0,
                'matches_count': 0
            }

        players_count = len(res_p) if isinstance(res_p, list) else 0

        # 2. Kiểm tra bảng matches
        ok_m, res_m = self._request('matches', method='GET', params={'select': 'id', 'limit': '1000'})
        matches_count = len(res_m) if ok_m and isinstance(res_m, list) else 0

        return {
            'connected': True,
            'message': 'Đã kết nối thành công tới Supabase PostgreSQL!',
            'players_count': players_count,
            'matches_count': matches_count,
            'url': self.url
        }

    # ==========================================
    # QUẢN LÝ TUYỂN THỦ (PLAYERS CRUD)
    # ==========================================
    def get_all_players(self) -> Optional[Dict[str, Any]]:
        """Lấy tất cả hồ sơ tuyển thủ từ Supabase dưới dạng Dict {id: player_obj}."""
        if not self.is_configured():
            return None

        ok, res = self._request('players', method='GET', params={'select': '*', 'order': 'id.asc'})
        if not ok or not isinstance(res, list):
            return None

        players_map = {}
        for row in res:
            pid = row['id']
            players_map[pid] = {
                'id': pid,
                'nickname': row.get('nickname', pid.capitalize()),
                'avatar': row.get('avatar') or f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}",
                'skill': float(row.get('skill', 7.0)),
                'champion_pool': float(row.get('champion_pool', 7.0)),
                'flex_lane': float(row.get('flex_lane', 6.5)),
                'consistency': float(row.get('consistency', 7.0)),
                'stats_ovr': float(row.get('stats_ovr', 6.9)),
                'primary_role': row.get('primary_role', 'ALL'),
                'favorite_champions': row.get('favorite_champions', []),
                'hidden_elo': float(row.get('hidden_elo', 1200.0)),
                'form_score': float(row.get('form_score', 5.0)),
                'form_status': row.get('form_status', 'neutral')
            }
        return players_map

    def upsert_player(self, p_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Thêm mới hoặc cập nhật một tuyển thủ trên Supabase."""
        if not self.is_configured():
            return False, "Chưa cấu hình Supabase"

        pid = p_data.get('id', '').strip().lower()
        if not pid:
            return False, "Thiếu ID tuyển thủ"

        row = {
            'id': pid,
            'nickname': p_data.get('nickname', pid.capitalize()),
            'avatar': p_data.get('avatar', ''),
            'skill': float(p_data.get('skill', 7.0)),
            'champion_pool': float(p_data.get('champion_pool', 7.0)),
            'flex_lane': float(p_data.get('flex_lane', 6.5)),
            'consistency': float(p_data.get('consistency', 7.0)),
            'stats_ovr': float(p_data.get('stats_ovr', 6.9)),
            'primary_role': p_data.get('primary_role', 'ALL'),
            'favorite_champions': p_data.get('favorite_champions', [])
        }

        ok, res = self._request(
            'players',
            method='POST',
            data=row,
            prefer='resolution=merge-duplicates,return=representation'
        )
        if ok:
            return True, "Đã lưu tuyển thủ lên Supabase thành công"
        return False, str(res)

    def delete_player(self, player_id: str) -> Tuple[bool, str]:
        """Xóa tuyển thủ khỏi bảng Supabase."""
        if not self.is_configured():
            return False, "Chưa cấu hình Supabase"

        ok, res = self._request(f"players?id=eq.{player_id}", method='DELETE')
        if ok:
            return True, "Đã xóa tuyển thủ trên Supabase"
        return False, str(res)

    def sync_players_from_local(self, local_players: Dict[str, Any]) -> Tuple[bool, str, int]:
        """Đồng bộ toàn bộ danh sách tuyển thủ từ local (players.json) lên Supabase."""
        if not self.is_configured():
            return False, "Chưa cấu hình Supabase", 0

        if not local_players:
            return False, "Không có dữ liệu tuyển thủ cục bộ để đồng bộ", 0

        rows = []
        for pid, p in local_players.items():
            rows.append({
                'id': pid,
                'nickname': p.get('nickname', pid.capitalize()),
                'avatar': p.get('avatar', f"https://api.dicebear.com/7.x/bottts/svg?seed={pid}"),
                'skill': float(p.get('skill', 7.0)),
                'champion_pool': float(p.get('champion_pool', 7.0)),
                'flex_lane': float(p.get('flex_lane', 6.5)),
                'consistency': float(p.get('consistency', 7.0)),
                'stats_ovr': float(p.get('stats_ovr', 6.9)),
                'primary_role': p.get('primary_role', 'ALL'),
                'favorite_champions': p.get('favorite_champions', [])
            })

        ok, res = self._request(
            'players',
            method='POST',
            data=rows,
            prefer='resolution=merge-duplicates,return=representation'
        )
        if ok:
            return True, f"Đã đồng bộ {len(rows)} tuyển thủ lên Supabase!", len(rows)
        return False, f"Lỗi đồng bộ Supabase: {res}", 0

    # ==========================================
    # QUẢN LÝ TRẬN ĐẤU ĐA CHIỀU (MATCHES)
    # ==========================================
    def get_matches_df(self, all_player_ids: List[str]) -> Optional[pd.DataFrame]:
        """
        Lấy danh sách các trận đấu từ Supabase và chuyển đổi thành DataFrame
        tương thích hoàn toàn với EloService (các cột người chơi 0/1/2 + cột Result).
        """
        if not self.is_configured():
            return None

        ok, matches = self._request('matches', method='GET', params={'select': '*', 'order': 'created_at.asc'})
        if not ok or not isinstance(matches, list):
            return None

        if len(matches) == 0:
            cols = list(all_player_ids) + ['Result']
            return pd.DataFrame(columns=cols)

        # Xây dựng ma trận phân tích từ bảng matches
        all_cols = set(all_player_ids)
        rows_data = []

        for m in matches:
            t1 = m.get('team1_players', [])
            t2 = m.get('team2_players', [])
            res = m.get('result_code', 1 if m.get('winner') == 'team1' else 2)

            all_cols.update(t1)
            all_cols.update(t2)

            row = {p: 0 for p in all_cols}
            for p in t1:
                row[p] = 1
            for p in t2:
                row[p] = 2
            row['Result'] = res
            rows_data.append(row)

        df = pd.DataFrame(rows_data)
        # Đảm bảo cột Result nằm ở cuối
        cols = [c for c in df.columns if c != 'Result'] + ['Result']
        return df[cols]

    @staticmethod
    def _parse_kda(kda_val: Any) -> Tuple[int, int, int]:
        if not kda_val:
            return 0, 0, 0
        try:
            parts = str(kda_val).split('/')
            if len(parts) == 3:
                return int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
        except Exception:
            pass
        return 0, 0, 0

    @staticmethod
    def _parse_damage(dmg_val: Any) -> int:
        if not dmg_val:
            return 0
        try:
            s = str(dmg_val).strip().lower()
            if s.endswith('k'):
                return int(float(s[:-1]) * 1000)
            return int(float(s))
        except Exception:
            return 0

    def insert_match(
        self,
        team1: List[str],
        team2: List[str],
        winner: str,
        team1_power: float = 0.0,
        team2_power: float = 0.0,
        synergies: Optional[Dict[str, Any]] = None,
        notes: str = '',
        player_deltas: Optional[Dict[str, float]] = None,
        player_performances: Optional[List[Dict[str, Any]]] = None,
        ai_summary: str = '',
        team1_kills: int = 0,
        team2_kills: int = 0,
        match_closeness: float = 0.5,
        is_stomp: bool = False,
        balance_rating: str = 'unknown'
    ) -> Tuple[bool, Any]:
        """
        Ghi nhận trận đấu đa chiều vào Supabase:
        1. Tạo bản ghi trong bảng `matches` kèm siêu dữ liệu AI phân tích (KDA, Elo cá nhân hóa)
        2. Tạo 10 bản ghi chi tiết trong `match_participants`
        """
        if not self.is_configured():
            return False, "Chưa cấu hình Supabase"

        result_code = 1 if winner == 'team1' else 2
        match_code = f"M-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

        syn_payload = dict(synergies or {})
        if player_deltas or player_performances or ai_summary:
            syn_payload['metadata'] = {
                'player_deltas': player_deltas or {},
                'player_performances': player_performances or [],
                'ai_summary': ai_summary or ''
            }

        match_row = {
            'match_code': match_code,
            'team1_players': team1,
            'team2_players': team2,
            'team1_power': float(team1_power),
            'team2_power': float(team2_power),
            'winner': winner,
            'result_code': result_code,
            'synergies_applied': syn_payload,
            'notes': notes,
            'team1_kills': int(team1_kills),
            'team2_kills': int(team2_kills),
            'match_closeness': float(match_closeness),
            'is_stomp': bool(is_stomp),
            'balance_rating': str(balance_rating)
        }

        # 1. Ghi vào bảng matches
        ok, res = self._request(
            'matches',
            method='POST',
            data=match_row,
            prefer='return=representation'
        )

        # Fallback tự động nếu Supabase chưa chạy migration (chưa có cột team1_kills)
        if not ok and isinstance(res, str) and 'column' in res and 'does not exist' in res:
            print("[SupabaseService] Notice: New kill score columns not found on Supabase. Falling back to metadata...")
            fallback_row = {
                'match_code': match_code,
                'team1_players': team1,
                'team2_players': team2,
                'team1_power': float(team1_power),
                'team2_power': float(team2_power),
                'winner': winner,
                'result_code': result_code,
                'synergies_applied': syn_payload,
                'notes': notes
            }
            # Lưu kill info vào metadata trong synergies_applied
            if 'metadata' not in fallback_row['synergies_applied']:
                fallback_row['synergies_applied']['metadata'] = {}
            fallback_row['synergies_applied']['metadata'].update({
                'team1_kills': int(team1_kills),
                'team2_kills': int(team2_kills),
                'match_closeness': float(match_closeness),
                'is_stomp': bool(is_stomp),
                'balance_rating': str(balance_rating)
            })
            ok, res = self._request(
                'matches',
                method='POST',
                data=fallback_row,
                prefer='return=representation'
            )

        if not ok:
            return False, f"Lỗi tạo trận đấu: {res}"

        match_id = None
        if isinstance(res, list) and len(res) > 0 and 'id' in res[0]:
            match_id = res[0]['id']
        # 2. Ghi chi tiết 10 tuyển thủ vào bảng match_participants nếu có match_id
        if match_id:
            participants = []
            perf_map = {str(p.get('player_id', '')).lower(): p for p in (player_performances or [])}
            for p in team1:
                p_perf = perf_map.get(str(p).lower(), {})
                k, d, a = self._parse_kda(p_perf.get('kda'))
                dmg = self._parse_damage(p_perf.get('damage'))
                participants.append({
                    'match_id': match_id,
                    'player_id': p,
                    'team': 1,
                    'is_winner': (winner == 'team1'),
                    'champion': p_perf.get('champion') if p_perf.get('champion') and p_perf.get('champion') != '-' else None,
                    'role': p_perf.get('performance_tag'),
                    'kills': k,
                    'deaths': d,
                    'assists': a,
                    'damage': dmg
                })
            for p in team2:
                p_perf = perf_map.get(str(p).lower(), {})
                k, d, a = self._parse_kda(p_perf.get('kda'))
                dmg = self._parse_damage(p_perf.get('damage'))
                participants.append({
                    'match_id': match_id,
                    'player_id': p,
                    'team': 2,
                    'is_winner': (winner == 'team2'),
                    'champion': p_perf.get('champion') if p_perf.get('champion') and p_perf.get('champion') != '-' else None,
                    'role': p_perf.get('performance_tag'),
                    'kills': k,
                    'deaths': d,
                    'assists': a,
                    'damage': dmg
                })

            self._request(
                'match_participants',
                method='POST',
                data=participants,
                prefer='return=minimal'
            )

        return True, match_id

    def get_all_matches_raw(self, order: str = 'created_at.desc') -> List[Dict[str, Any]]:
        """Lấy danh sách các trận đấu nguyên bản từ Supabase."""
        if not self.is_configured():
            return []
        ok, res = self._request('matches', method='GET', params={'select': '*', 'order': order})
        if ok and isinstance(res, list):
            return res
        return []

    def update_match(self, match_id: int, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Cập nhật trận đấu và danh sách tuyển thủ tham gia trong match_participants."""
        if not self.is_configured():
            return False, "Chưa cấu hình Supabase"

        team1 = data.get('team1', [])
        team2 = data.get('team2', [])
        winner = data.get('winner', 'team1')
        result_code = 1 if winner == 'team1' else 2

        match_update: Dict[str, Any] = {
            'team1_players': team1,
            'team2_players': team2,
            'winner': winner,
            'result_code': result_code
        }
        if 'notes' in data:
            match_update['notes'] = data['notes']
        if 'team1_power' in data:
            match_update['team1_power'] = float(data['team1_power'])
        if 'team2_power' in data:
            match_update['team2_power'] = float(data['team2_power'])
        if 'synergies_applied' in data:
            match_update['synergies_applied'] = data['synergies_applied']
        if 'team1_kills' in data:
            match_update['team1_kills'] = int(data['team1_kills'])
        if 'team2_kills' in data:
            match_update['team2_kills'] = int(data['team2_kills'])
        if 'match_closeness' in data:
            match_update['match_closeness'] = float(data['match_closeness'])
        if 'is_stomp' in data:
            match_update['is_stomp'] = bool(data['is_stomp'])
        if 'balance_rating' in data:
            match_update['balance_rating'] = str(data['balance_rating'])

        ok, res = self._request(
            f"matches?id=eq.{match_id}",
            method='PATCH',
            data=match_update,
            prefer='return=representation'
        )
        if not ok and isinstance(res, str) and 'column' in res and 'does not exist' in res:
            print("[SupabaseService] Notice: New kill score columns not found on Supabase. Falling back to metadata...")
            for k in ['team1_kills', 'team2_kills', 'match_closeness', 'is_stomp', 'balance_rating']:
                match_update.pop(k, None)
            ok, res = self._request(
                f"matches?id=eq.{match_id}",
                method='PATCH',
                data=match_update,
                prefer='return=representation'
            )

        if not ok:
            return False, f"Lỗi cập nhật bảng matches: {res}"

        # Cập nhật match_participants: xóa cũ và tạo mới
        self._request(f"match_participants?match_id=eq.{match_id}", method='DELETE')

        syn = data.get('synergies_applied') or {}
        meta = syn.get('metadata') if isinstance(syn, dict) else {}
        perfs = meta.get('player_performances', []) if meta else []
        perf_map = {str(p.get('player_id', '')).lower(): p for p in perfs}

        participants = []
        for p in team1:
            p_perf = perf_map.get(str(p).lower(), {})
            k, d, a = self._parse_kda(p_perf.get('kda'))
            dmg = self._parse_damage(p_perf.get('damage'))
            participants.append({
                'match_id': match_id,
                'player_id': p,
                'team': 1,
                'is_winner': (winner == 'team1'),
                'champion': p_perf.get('champion') if p_perf.get('champion') and p_perf.get('champion') != '-' else None,
                'role': p_perf.get('performance_tag'),
                'kills': k,
                'deaths': d,
                'assists': a,
                'damage': dmg
            })
        for p in team2:
            p_perf = perf_map.get(str(p).lower(), {})
            k, d, a = self._parse_kda(p_perf.get('kda'))
            dmg = self._parse_damage(p_perf.get('damage'))
            participants.append({
                'match_id': match_id,
                'player_id': p,
                'team': 2,
                'is_winner': (winner == 'team2'),
                'champion': p_perf.get('champion') if p_perf.get('champion') and p_perf.get('champion') != '-' else None,
                'role': p_perf.get('performance_tag'),
                'kills': k,
                'deaths': d,
                'assists': a,
                'damage': dmg
            })

        self._request('match_participants', method='POST', data=participants, prefer='return=minimal')
        return True, "Đã cập nhật trận đấu thành công"

    def delete_match(self, match_id: int) -> Tuple[bool, str]:
        """Xóa một trận đấu khỏi Supabase."""
        if not self.is_configured():
            return False, "Chưa cấu hình Supabase"

        self._request(f"match_participants?match_id=eq.{match_id}", method='DELETE')
        ok, res = self._request(f"matches?id=eq.{match_id}", method='DELETE')
        if ok:
            return True, "Đã xóa trận đấu thành công"
        return False, f"Lỗi xóa trận đấu: {res}"


supabase_service = SupabaseService()
