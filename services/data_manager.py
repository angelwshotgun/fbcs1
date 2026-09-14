import json
import os
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

# GitHub configurations
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = os.getenv('REPO_NAME', 'angelwshotgun/fbcs')
MATCH_FILE_PATH = 'match_data.csv'
PLAYERS_FILE_PATH = 'players.json'


class DataManager:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
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
    def read_local_players_data(self) -> Dict[str, Any]:
        """Đọc hồ sơ tuyển thủ từ file JSON cục bộ."""
        if os.path.exists(LOCAL_PLAYERS_FILE):
            try:
                with open(LOCAL_PLAYERS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DataManager] Read local players.json error: {e}")
        return {}

    def read_players_data(self) -> Dict[str, Any]:
        """
        Đọc hồ sơ tuyển thủ:
        1. Ưu tiên đọc từ Supabase PostgreSQL nếu khả dụng.
        2. Tự động lưu bản sao xuống local JSON để cache.
        3. Nếu Supabase lỗi hoặc chưa kết nối, fallback về local file hoặc GitHub.
        """
        # 1. Thử đọc từ Supabase trước
        if supabase_service.is_configured():
            try:
                sb_players = supabase_service.get_all_players()
                if sb_players is not None and len(sb_players) > 0:
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
            return local_data

        # 3. Fallback đọc từ GitHub
        if self.github_repo:
            try:
                file_content = self.github_repo.get_contents(PLAYERS_FILE_PATH)
                file_data = file_content.decoded_content.decode('utf-8')
                data = json.loads(file_data)
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

        # 1. Lưu local cache
        try:
            with open(LOCAL_PLAYERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(players_data, f, ensure_ascii=False, indent=2)
            saved_local = True
        except Exception as e:
            print(f"[DataManager] Save local players.json error: {e}")

        # 2. Đồng bộ lên Supabase nếu có
        if supabase_service.is_configured():
            try:
                supabase_service.sync_players_from_local(players_data)
            except Exception as e:
                print(f"[DataManager] Supabase sync error: {e}")

        # 3. Đồng bộ lên GitHub nếu có
        if self.github_repo:
            try:
                json_content = json.dumps(players_data, ensure_ascii=False, indent=2)
                try:
                    file_item = self.github_repo.get_contents(PLAYERS_FILE_PATH)
                    self.github_repo.update_file(PLAYERS_FILE_PATH, "Update players.json", json_content, file_item.sha)
                except Exception:
                    self.github_repo.create_file(PLAYERS_FILE_PATH, "Create players.json", json_content)
            except Exception as e:
                print(f"[DataManager] Save GitHub players.json error: {e}")

        return saved_local

    def save_single_player(self, player_obj: Dict[str, Any]) -> bool:
        """Thêm mới hoặc cập nhật 1 tuyển thủ (Supabase + Local)."""
        pid = player_obj.get('id', '').strip().lower()
        if not pid:
            return False

        # 1. Cập nhật Supabase
        if supabase_service.is_configured():
            try:
                supabase_service.upsert_player(player_obj)
            except Exception as e:
                print(f"[DataManager] Supabase upsert player error: {e}")

        # 2. Cập nhật local JSON
        players_data = self.read_local_players_data()
        players_data[pid] = player_obj
        try:
            with open(LOCAL_PLAYERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(players_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DataManager] Save single player local error: {e}")

        return True

    def delete_single_player(self, player_id: str) -> bool:
        """Xóa 1 tuyển thủ (Supabase + Local)."""
        pid = player_id.strip().lower()
        # 1. Xóa trên Supabase
        if supabase_service.is_configured():
            try:
                supabase_service.delete_player(pid)
            except Exception as e:
                print(f"[DataManager] Supabase delete player error: {e}")

        # 2. Xóa trong local JSON
        players_data = self.read_local_players_data()
        if pid in players_data:
            del players_data[pid]
            try:
                with open(LOCAL_PLAYERS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(players_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[DataManager] Delete single player local error: {e}")

        return True

    # ==========================================
    # MATCHES OPERATIONS (SUPABASE + LOCAL)
    # ==========================================
    def read_matches_df(self) -> pd.DataFrame:
        """
        Đọc danh sách trận đấu:
        1. Thử lấy từ Supabase (truy vấn bảng matches đa chiều và chuyển đổi thành ma trận Elo).
        2. Nếu Supabase chưa có hoặc lỗi, đọc từ local file match_data.csv.
        """
        # 1. Thử đọc từ Supabase
        if supabase_service.is_configured():
            try:
                players_map = self.read_players_data()
                all_pids = list(players_map.keys())
                df = supabase_service.get_matches_df(all_pids)
                if df is not None and not df.empty:
                    try:
                        df.to_csv(LOCAL_MATCH_FILE, index=False)
                    except Exception:
                        pass
                    return df
            except Exception as e:
                print(f"[DataManager] Supabase read matches error: {e}")

        # 2. Đọc từ local CSV
        if os.path.exists(LOCAL_MATCH_FILE):
            try:
                return pd.read_csv(LOCAL_MATCH_FILE)
            except Exception as e:
                print(f"[DataManager] Read local CSV error: {e}")

        # 3. Đọc từ GitHub nếu có
        if self.github_repo:
            try:
                file_content = self.github_repo.get_contents(MATCH_FILE_PATH)
                file_data = file_content.decoded_content.decode('utf-8')
                df = pd.read_csv(StringIO(file_data))
                try:
                    df.to_csv(LOCAL_MATCH_FILE, index=False)
                except Exception:
                    pass
                return df
            except Exception as e:
                print(f"[DataManager] Read GitHub CSV failed: {e}")

        return pd.DataFrame(columns=['Result'])

    def save_matches_df(self, df: pd.DataFrame) -> bool:
        """Lưu DataFrame trận đấu vào local file và GitHub."""
        saved_local = False
        try:
            df.to_csv(LOCAL_MATCH_FILE, index=False)
            saved_local = True
        except Exception as e:
            print(f"[DataManager] Save local CSV error: {e}")

        if self.github_repo:
            try:
                csv_content = df.to_csv(index=False)
                try:
                    file_item = self.github_repo.get_contents(MATCH_FILE_PATH)
                    self.github_repo.update_file(MATCH_FILE_PATH, "Update match_data.csv", csv_content, file_item.sha)
                except Exception:
                    self.github_repo.create_file(MATCH_FILE_PATH, "Create match_data.csv", csv_content)
            except Exception as e:
                print(f"[DataManager] Save GitHub CSV error: {e}")

        return saved_local

    def append_match(
        self,
        team1: List[str],
        team2: List[str],
        winner: str,
        team1_power: float = 0.0,
        team2_power: float = 0.0,
        synergies: Optional[Dict[str, Any]] = None,
        notes: str = ''
    ) -> pd.DataFrame:
        """
        Ghi nhận trận đấu mới đa chiều:
        1. Ghi nhận vào Supabase (bảng matches + match_participants).
        2. Đồng thời cập nhật ma trận DataFrame cục bộ để duy trì tương thích và sao lưu.
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
                    notes=notes
                )
            except Exception as e:
                print(f"[DataManager] Supabase insert match notice: {e}")

        # 2. Cập nhật local DataFrame
        df = self.read_matches_df()
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

        new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_matches_df(new_df)
        return new_df


data_manager = DataManager()
