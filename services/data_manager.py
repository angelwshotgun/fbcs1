import json
import os
import threading
import time
from io import StringIO
from typing import Any, Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
from services.supabase_service import supabase_service

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOCAL_MATCH_FILE = os.path.join(DATA_DIR, 'match_data.csv')
LOCAL_PLAYERS_FILE = os.path.join(DATA_DIR, 'players.json')
LOCAL_MATCH_DETAILS_FILE = os.path.join(DATA_DIR, 'match_details.json')

# GitHub configurations
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = os.getenv('REPO_NAME', 'angelwshotgun/fbcs')
MATCH_FILE_PATH = 'match_data.csv'
PLAYERS_FILE_PATH = 'players.json'


class DataManager:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._players_cache: Optional[Dict[str, Any]] = None
        self._players_cache_time: float = 0.0
        self._matches_df_cache: Optional[pd.DataFrame] = None
        self._matches_cache_time: float = 0.0
        self.github_repo = None
        self._init_github()
        self._check_and_seed_supabase()

    def _init_github(self):
        if GITHUB_TOKEN:
            try:
                from github import Github
                g = Github(GITHUB_TOKEN)
                self.github_repo = g.get_repo(REPO_NAME)
            except Exception as e:
                print(f"[DataManager] GitHub init error: {e}")
                self.github_repo = None

    def _check_and_seed_supabase(self):
        """Nếu Supabase đã cấu hình nhưng chưa có dữ liệu tuyển thủ, tự động seed từ players.json."""
        if supabase_service.is_configured():
            try:
                status = supabase_service.test_connection()
                if status.get('connected') and status.get('players_count', 0) == 0:
                    local_p = self.read_local_players_data()
                    if local_p:
                        print(f"[DataManager] Seeding {len(local_p)} players to Supabase...")
                        supabase_service.sync_players_from_local(local_p)
            except Exception as e:
                print(f"[DataManager] Supabase auto-seed notice: {e}")

    def get_storage_mode(self) -> str:
        if supabase_service.is_configured():
            return "supabase"
        if self.github_repo is not None:
            return "github"
        return "local"

    # ==========================================
    # PLAYERS STATS OPERATIONS (SUPABASE + LOCAL)
    # ==========================================
    def _backup_players_github(self, players_data: Dict[str, Any]):
        if not self.github_repo:
            return
        try:
            json_content = json.dumps(players_data, ensure_ascii=False, indent=2)
            try:
                file_item = self.github_repo.get_contents(PLAYERS_FILE_PATH)
                self.github_repo.update_file(PLAYERS_FILE_PATH, "Update players.json", json_content, file_item.sha)
            except Exception:
                self.github_repo.create_file(PLAYERS_FILE_PATH, "Create players.json", json_content)
        except Exception as e:
            print(f"[DataManager] Save GitHub players.json background error: {e}")

    def _backup_matches_github(self, df: pd.DataFrame):
        if not self.github_repo:
            return
        try:
            csv_content = df.to_csv(index=False)
            try:
                file_item = self.github_repo.get_contents(MATCH_FILE_PATH)
                self.github_repo.update_file(MATCH_FILE_PATH, "Update match_data.csv", csv_content, file_item.sha)
            except Exception:
                self.github_repo.create_file(MATCH_FILE_PATH, "Create match_data.csv", csv_content)
        except Exception as e:
            print(f"[DataManager] Save GitHub CSV background error: {e}")

    # ==========================================
    # PLAYERS STATS OPERATIONS (SUPABASE + LOCAL)
    # ==========================================
    def read_local_players_data(self) -> Dict[str, Any]:
        """Đọc hồ sơ tuyển thủ từ file JSON cục bộ."""
        if os.path.exists(LOCAL_PLAYERS_FILE):
            try:
                with open(LOCAL_PLAYERS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DataManager] Read local players.json error: {e}")
        return {}

    def read_players_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Đọc hồ sơ tuyển thủ:
        1. Sử dụng in-memory cache nếu còn hạn (TTL = 30s) và không ép buộc refresh.
        2. Thử đọc từ Supabase PostgreSQL nếu khả dụng.
        3. Fallback đọc từ local file hoặc GitHub.
        """
        now = time.time()
        if not force_refresh and self._players_cache and (now - self._players_cache_time < 30):
            return self._players_cache

        # 1. Thử đọc từ Supabase trước
        if supabase_service.is_configured():
            try:
                sb_players = supabase_service.get_all_players()
                if sb_players is not None and len(sb_players) > 0:
                    with self._lock:
                        self._players_cache = sb_players
                        self._players_cache_time = now
                        try:
                            with open(LOCAL_PLAYERS_FILE, 'w', encoding='utf-8') as f:
                                json.dump(sb_players, f, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                    return sb_players
            except Exception as e:
                print(f"[DataManager] Supabase read players error: {e}")

        # 2. Fallback đọc từ local file
        local_data = self.read_local_players_data()
        if local_data:
            with self._lock:
                self._players_cache = local_data
                self._players_cache_time = now
            return local_data

        # 3. Fallback đọc từ GitHub
        if self.github_repo:
            try:
                file_content = self.github_repo.get_contents(PLAYERS_FILE_PATH)
                file_data = file_content.decoded_content.decode('utf-8')
                data = json.loads(file_data)
                with self._lock:
                    self._players_cache = data
                    self._players_cache_time = now
                    try:
                        with open(LOCAL_PLAYERS_FILE, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                return data
            except Exception:
                pass

        return {}

    def save_players_data(self, players_data: Dict[str, Any]) -> bool:
        """Lưu thông tin tuyển thủ đồng thời vào Supabase và file cục bộ."""
        saved_local = False

        with self._lock:
            try:
                with open(LOCAL_PLAYERS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(players_data, f, ensure_ascii=False, indent=2)
                saved_local = True
                self._players_cache = players_data
                self._players_cache_time = time.time()
            except Exception as e:
                print(f"[DataManager] Save local players.json error: {e}")

        # Đồng bộ lên Supabase nếu có
        if supabase_service.is_configured():
            try:
                supabase_service.sync_players_from_local(players_data)
            except Exception as e:
                print(f"[DataManager] Supabase sync error: {e}")

        # Đồng bộ lên GitHub non-blocking
        if self.github_repo:
            threading.Thread(target=self._backup_players_github, args=(players_data,), daemon=True).start()

        return saved_local

    def save_single_player(self, player_obj: Dict[str, Any]) -> bool:
        """Thêm mới hoặc cập nhật 1 tuyển thủ (Supabase + Local)."""
        pid = player_obj.get('id', '').strip().lower()
        if not pid:
            return False
        player_obj['id'] = pid

        with self._lock:
            # 1. Cập nhật local JSON (xóa triệt để các key trùng lệch hoa/thường)
            players_data = self.read_local_players_data()
            for k in list(players_data.keys()):
                if k.lower() == pid:
                    del players_data[k]
            players_data[pid] = player_obj
            try:
                with open(LOCAL_PLAYERS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(players_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[DataManager] Save single player local error: {e}")

            # Cập nhật cache tức thì
            self._players_cache = players_data
            self._players_cache_time = time.time()

        # 2. Cập nhật Supabase (chỉ upsert đúng 1 tuyển thủ)
        if supabase_service.is_configured():
            try:
                supabase_service.upsert_player(player_obj)
            except Exception as e:
                print(f"[DataManager] Supabase upsert player error: {e}")

        # 3. GitHub backup non-blocking
        if self.github_repo:
            threading.Thread(target=self._backup_players_github, args=(players_data,), daemon=True).start()

        return True

    def delete_single_player(self, player_id: str) -> bool:
        """Xóa 1 tuyển thủ (Supabase + Local)."""
        pid = player_id.strip().lower()
        keys_to_delete = []

        with self._lock:
            players_data = self.read_local_players_data()
            keys_to_delete = [k for k in players_data.keys() if k.lower() == pid]
            for k in keys_to_delete:
                del players_data[k]
            try:
                with open(LOCAL_PLAYERS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(players_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[DataManager] Delete single player local error: {e}")

            self._players_cache = players_data
            self._players_cache_time = time.time()

        # Xóa trên Supabase
        if supabase_service.is_configured():
            try:
                supabase_service.delete_player(pid)
                for k in keys_to_delete:
                    if k != pid:
                        supabase_service.delete_player(k)
            except Exception as e:
                print(f"[DataManager] Supabase delete player error: {e}")

        return True

    # ==========================================
    # MATCHES OPERATIONS (SUPABASE + LOCAL)
    # ==========================================
    def read_matches_df(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Đọc danh sách trận đấu:
        1. Sử dụng in-memory cache nếu có.
        2. Thử lấy từ Supabase (truy vấn bảng matches đa chiều và chuyển đổi thành ma trận Elo).
        3. Nếu Supabase chưa có hoặc lỗi, đọc từ local file match_data.csv.
        """
        now = time.time()
        if not force_refresh and self._matches_df_cache is not None and (now - self._matches_cache_time < 30):
            return self._matches_df_cache.copy()

        # 1. Thử đọc từ Supabase
        if supabase_service.is_configured():
            try:
                players_map = self.read_players_data()
                all_pids = list(players_map.keys())
                df = supabase_service.get_matches_df(all_pids)
                if df is not None and not df.empty:
                    with self._lock:
                        self._matches_df_cache = df
                        self._matches_cache_time = now
                        try:
                            df.to_csv(LOCAL_MATCH_FILE, index=False)
                        except Exception:
                            pass
                    return df.copy()
            except Exception as e:
                print(f"[DataManager] Supabase read matches error: {e}")

        # 2. Đọc từ local CSV
        if os.path.exists(LOCAL_MATCH_FILE):
            try:
                df = pd.read_csv(LOCAL_MATCH_FILE)
                with self._lock:
                    self._matches_df_cache = df
                    self._matches_cache_time = now
                return df.copy()
            except Exception as e:
                print(f"[DataManager] Read local CSV error: {e}")

        # 3. Đọc từ GitHub nếu có
        if self.github_repo:
            try:
                file_content = self.github_repo.get_contents(MATCH_FILE_PATH)
                file_data = file_content.decoded_content.decode('utf-8')
                df = pd.read_csv(StringIO(file_data))
                with self._lock:
                    self._matches_df_cache = df
                    self._matches_cache_time = now
                    try:
                        df.to_csv(LOCAL_MATCH_FILE, index=False)
                    except Exception:
                        pass
                return df.copy()
            except Exception as e:
                print(f"[DataManager] Read GitHub CSV failed: {e}")

        return pd.DataFrame(columns=['Result'])

    def save_matches_df(self, df: pd.DataFrame) -> bool:
        """Lưu DataFrame trận đấu vào local file và GitHub."""
        saved_local = False
        with self._lock:
            try:
                df.to_csv(LOCAL_MATCH_FILE, index=False)
                saved_local = True
                self._matches_df_cache = df.copy()
                self._matches_cache_time = time.time()
            except Exception as e:
                print(f"[DataManager] Save local CSV error: {e}")

        if self.github_repo:
            threading.Thread(target=self._backup_matches_github, args=(df.copy(),), daemon=True).start()

        return saved_local

    def read_match_details(self) -> Dict[str, Any]:
        """
        Đọc chi tiết các trận đấu (KDA, Elo deltas cá nhân hóa, AI analysis).
        1. Ưu tiên đọc trực tiếp từ bảng matches trên Supabase (cột synergies_applied -> metadata).
        2. Fallback đọc từ file JSON cục bộ data/match_details.json.
        """
        details: Dict[str, Any] = {}

        # 1. Thử lấy từ Supabase
        if supabase_service.is_configured():
            try:
                raw_matches = supabase_service.get_all_matches_raw(order='created_at.asc')
                if raw_matches:
                    for idx, m in enumerate(raw_matches):
                        mid = m.get('id')
                        syn = m.get('synergies_applied') or {}
                        meta = syn.get('metadata') if isinstance(syn, dict) else {}
                        if not meta and isinstance(syn, dict) and 'player_deltas' in syn:
                            meta = syn

                        p_deltas = meta.get('player_deltas', {}) if meta else {}
                        p_perfs = meta.get('player_performances', []) if meta else {}
                        ai_sum = meta.get('ai_summary', '') if meta else ''

                        record = {
                            'id': mid,
                            'match_index': idx,
                            'match_code': m.get('match_code', f"M-{mid}"),
                            'created_at': m.get('created_at', ''),
                            'winner': m.get('winner', 'team1'),
                            'team1': m.get('team1_players', []),
                            'team2': m.get('team2_players', []),
                            'team1_power': float(m.get('team1_power', 0.0)),
                            'team2_power': float(m.get('team2_power', 0.0)),
                            'player_deltas': p_deltas,
                            'player_performances': p_perfs,
                            'ai_summary': ai_sum,
                            'notes': m.get('notes', '')
                        }
                        # Đánh dấu theo cả số thứ tự dòng (cho EloService) và theo ID trận
                        details[str(idx)] = record
                        if mid:
                            details[str(mid)] = record

                    # Đồng bộ cache cục bộ
                    try:
                        with open(LOCAL_MATCH_DETAILS_FILE, 'w', encoding='utf-8') as f:
                            json.dump(details, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    return details
            except Exception as e:
                print(f"[DataManager] Supabase read match details notice: {e}")

        # 2. Đọc từ file JSON cục bộ
        if os.path.exists(LOCAL_MATCH_DETAILS_FILE):
            try:
                with open(LOCAL_MATCH_DETAILS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DataManager] Read match_details.json error: {e}")
        return details

    def save_match_details(self, details: Dict[str, Any]) -> bool:
        """Lưu chi tiết các trận đấu vào file JSON cục bộ."""
        with self._lock:
            try:
                with open(LOCAL_MATCH_DETAILS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(details, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                print(f"[DataManager] Save match_details.json error: {e}")
                return False

    def append_match(
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
        ai_summary: str = ''
    ) -> pd.DataFrame:
        """
        Ghi nhận trận đấu mới đa chiều:
        1. Ghi nhận vào Supabase (bảng matches + match_participants kèm siêu dữ liệu AI).
        2. Đồng thời cập nhật ma trận DataFrame cục bộ để duy trì tương thích và sao lưu.
        3. Lưu chi tiết cá nhân hóa (KDA, delta Elo cá nhân, AI summary) vào match_details.json.
        """
        # 1. Ghi vào Supabase
        if supabase_service.is_configured():
            try:
                supabase_service.insert_match(
                    team1=team1,
                    team2=team2,
                    winner=winner,
                    team1_power=team1_power,
                    team2_power=team2_power,
                    synergies=synergies,
                    notes=notes,
                    player_deltas=player_deltas,
                    player_performances=player_performances,
                    ai_summary=ai_summary
                )
                with self._lock:
                    self._matches_df_cache = None
                    self._matches_cache_time = 0
            except Exception as e:
                print(f"[DataManager] Supabase insert match notice: {e}")

        # 2. Cập nhật local DataFrame
        df = self.read_matches_df(force_refresh=True)
        all_players_in_match = set(team1 + team2)
        for player in all_players_in_match:
            if player not in df.columns:
                df[player] = 0

        new_row = {col: 0 for col in df.columns}
        for player in team1:
            new_row[player] = 1
        for player in team2:
            new_row[player] = 2

        new_row['Result'] = 1 if winner == 'team1' else 2

        match_idx = len(df)
        new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_matches_df(new_df)

        # 3. Lưu chi tiết cá nhân hóa vào match_details.json
        match_details = self.read_match_details()
        match_record = {
            'match_index': match_idx,
            'timestamp': time.time(),
            'winner': winner,
            'team1': team1,
            'team2': team2,
            'team1_power': team1_power,
            'team2_power': team2_power,
            'player_deltas': player_deltas or {},
            'player_performances': player_performances or [],
            'ai_summary': ai_summary or '',
            'notes': notes or ''
        }
        match_details[str(match_idx)] = match_record
        self.save_match_details(match_details)

        return new_df

    def get_matches_history(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách toàn bộ lịch sử trận đấu kèm thông tin phong phú của tuyển thủ.
        Sắp xếp theo thời gian mới nhất lên đầu.
        """
        players_map = self.read_players_data()
        history: List[Dict[str, Any]] = []

        if supabase_service.is_configured():
            raw_matches = supabase_service.get_all_matches_raw(order='created_at.desc')
            for m in raw_matches:
                mid = m.get('id')
                syn = m.get('synergies_applied') or {}
                meta = syn.get('metadata') if isinstance(syn, dict) else {}
                if not meta and isinstance(syn, dict) and 'player_deltas' in syn:
                    meta = syn

                t1_pids = m.get('team1_players', [])
                t2_pids = m.get('team2_players', [])

                t1_objs = [
                    players_map.get(str(p).lower(), {
                        'id': p,
                        'nickname': str(p).capitalize(),
                        'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={p}",
                        'hidden_elo': 1200.0
                    }) for p in t1_pids
                ]
                t2_objs = [
                    players_map.get(str(p).lower(), {
                        'id': p,
                        'nickname': str(p).capitalize(),
                        'avatar': f"https://api.dicebear.com/7.x/bottts/svg?seed={p}",
                        'hidden_elo': 1200.0
                    }) for p in t2_pids
                ]

                history.append({
                    'id': mid,
                    'match_code': m.get('match_code', f"M-{mid}"),
                    'created_at': m.get('created_at', ''),
                    'winner': m.get('winner', 'team1'),
                    'result_code': m.get('result_code', 1 if m.get('winner') == 'team1' else 2),
                    'team1': t1_objs,
                    'team2': t2_objs,
                    'team1_players': t1_pids,
                    'team2_players': t2_pids,
                    'team1_power': m.get('team1_power', 0.0),
                    'team2_power': m.get('team2_power', 0.0),
                    'notes': m.get('notes', ''),
                    'ai_summary': meta.get('ai_summary', '') if meta else '',
                    'player_deltas': meta.get('player_deltas', {}) if meta else {},
                    'player_performances': meta.get('player_performances', []) if meta else []
                })
            return history

        # Fallback local
        df = self.read_matches_df()
        details = self.read_match_details()
        player_cols = [c for c in df.columns if c != 'Result']

        for idx, row in df.iterrows():
            result = row['Result']
            t1_pids = [p for p in player_cols if row[p] == 1]
            t2_pids = [p for p in player_cols if row[p] == 2]
            meta = details.get(str(idx), {})

            t1_objs = [
                players_map.get(p.lower(), {'id': p, 'nickname': p.capitalize()}) for p in t1_pids
            ]
            t2_objs = [
                players_map.get(p.lower(), {'id': p, 'nickname': p.capitalize()}) for p in t2_pids
            ]

            history.append({
                'id': idx + 1,
                'match_code': meta.get('match_code', f"M-LOCAL-{idx + 1}"),
                'created_at': meta.get('created_at', ''),
                'winner': 'team1' if result == 1 else 'team2',
                'result_code': int(result),
                'team1': t1_objs,
                'team2': t2_objs,
                'team1_players': t1_pids,
                'team2_players': t2_pids,
                'team1_power': meta.get('team1_power', 0.0),
                'team2_power': meta.get('team2_power', 0.0),
                'notes': meta.get('notes', ''),
                'ai_summary': meta.get('ai_summary', ''),
                'player_deltas': meta.get('player_deltas', {}),
                'player_performances': meta.get('player_performances', [])
            })

        history.reverse()
        return history

    def update_match(
        self,
        match_id: int,
        team1: List[str],
        team2: List[str],
        winner: str,
        notes: str = '',
        player_deltas: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """Cập nhật kết quả hoặc người chơi trong một trận đấu (Supabase + Local)."""
        all_players_map = self.read_players_data()
        t1_power = round(sum(all_players_map.get(str(p).lower(), {}).get('hidden_elo', 1200.0) for p in team1), 1)
        t2_power = round(sum(all_players_map.get(str(p).lower(), {}).get('hidden_elo', 1200.0) for p in team2), 1)

        # 1. Cập nhật Supabase
        if supabase_service.is_configured():
            try:
                # Đọc siêu dữ liệu cũ để giữ lại nếu có
                cur_details = self.read_match_details()
                old_record = cur_details.get(str(match_id), {})
                syn_payload = {}
                meta_payload = {
                    'player_deltas': player_deltas if player_deltas is not None else old_record.get('player_deltas', {}),
                    'player_performances': old_record.get('player_performances', []),
                    'ai_summary': old_record.get('ai_summary', '')
                }
                syn_payload['metadata'] = meta_payload

                update_data = {
                    'team1': team1,
                    'team2': team2,
                    'winner': winner,
                    'team1_power': t1_power,
                    'team2_power': t2_power,
                    'notes': notes,
                    'synergies_applied': syn_payload
                }
                ok, msg = supabase_service.update_match(match_id, update_data)
                if not ok:
                    return False, msg

                with self._lock:
                    self._matches_df_cache = None
                    self._matches_cache_time = 0
            except Exception as e:
                return False, f"Lỗi cập nhật Supabase: {str(e)}"

        # 2. Xóa cache và nạp lại để đồng bộ hóa hoàn toàn
        self.read_match_details()
        self.read_matches_df(force_refresh=True)

        return True, "Cập nhật trận đấu thành công"

    def delete_match(self, match_id: int) -> Tuple[bool, str]:
        """Xóa một trận đấu khỏi hệ thống (Supabase + Local)."""
        if supabase_service.is_configured():
            try:
                ok, msg = supabase_service.delete_match(match_id)
                if not ok:
                    return False, msg

                with self._lock:
                    self._matches_df_cache = None
                    self._matches_cache_time = 0
            except Exception as e:
                return False, f"Lỗi xóa Supabase: {str(e)}"

        # Xóa trong match_details.json nếu có
        details = self.read_match_details()
        if str(match_id) in details:
            del details[str(match_id)]
            self.save_match_details(details)

        self.read_matches_df(force_refresh=True)
        return True, "Đã xóa trận đấu thành công"


data_manager = DataManager()


