import os
import json
from io import StringIO
from typing import Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

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

# Supabase configurations (optional)
SUPABASE_URL = os.getenv('SUPABASE_URL') or os.getenv('NEXT_PUBLIC_SUPABASE_URL')
SUPABASE_KEY = (
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    or os.getenv('SUPABASE_KEY')
    or os.getenv('SUPABASE_ANON_KEY')
    or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
)


class DataManager:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.github_repo = None
        self._init_github()

    def _init_github(self):
        if GITHUB_TOKEN:
            try:
                from github import Github
                g = Github(GITHUB_TOKEN)
                self.github_repo = g.get_repo(REPO_NAME)
            except Exception as e:
                print(f"[DataManager] GitHub init error: {e}. Falling back to local storage.")
                self.github_repo = None

    def get_storage_mode(self) -> str:
        if SUPABASE_URL and SUPABASE_KEY:
            return "supabase"
        if self.github_repo is not None:
            return "github"
        return "local"

    # ==========================
    # MATCH DATA CSV OPERATIONS
    # ==========================
    def read_matches_df(self) -> pd.DataFrame:
        """Đọc file match_data.csv cục bộ trước, nếu không có mới đọc từ GitHub."""
        # 1. Đọc từ local file trước
        if os.path.exists(LOCAL_MATCH_FILE):
            try:
                return pd.read_csv(LOCAL_MATCH_FILE)
            except Exception as e:
                print(f"[DataManager] Read local CSV error: {e}")

        # 2. Thử đọc từ GitHub nếu chưa có local file
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
                print(f"[DataManager] Read GitHub CSV failed: {e}. Falling back to empty.")

        return pd.DataFrame(columns=['Result'])

    def save_matches_df(self, df: pd.DataFrame) -> bool:
        """Lưu file match_data.csv cả cục bộ và GitHub nếu có."""
        saved_local = False
        saved_remote = False

        # 1. Luôn lưu vào local file
        try:
            df.to_csv(LOCAL_MATCH_FILE, index=False)
            saved_local = True
        except Exception as e:
            print(f"[DataManager] Save local CSV error: {e}")

        # 2. Cố gắng đồng bộ lên GitHub
        if self.github_repo:
            try:
                csv_content = df.to_csv(index=False)
                try:
                    file_item = self.github_repo.get_contents(MATCH_FILE_PATH)
                    self.github_repo.update_file(MATCH_FILE_PATH, "Update match_data.csv", csv_content, file_item.sha)
                except Exception:
                    self.github_repo.create_file(MATCH_FILE_PATH, "Create match_data.csv", csv_content)
                saved_remote = True
            except Exception as e:
                print(f"[DataManager] Save GitHub CSV error: {e}")

        return saved_local or saved_remote

    def append_match(self, team1: list, team2: list, winner: str) -> pd.DataFrame:
        """Thêm kết quả 1 trận đấu mới và cập nhật dữ liệu."""
        df = self.read_matches_df()

        # Đảm bảo tất cả người chơi đều có cột trong dataframe
        all_players_in_match = set(team1 + team2)
        for player in all_players_in_match:
            if player not in df.columns:
                df[player] = 0

        # Tạo dòng mới (mặc định 0 cho tất cả cột trừ Result)
        new_row = {col: 0 for col in df.columns}
        for player in team1:
            new_row[player] = 1
        for player in team2:
            new_row[player] = 2

        new_row['Result'] = 1 if winner == 'team1' else 2

        new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_matches_df(new_df)
        return new_df

    # ==========================
    # PLAYERS STATS JSON OPERATIONS
    # ==========================
    def read_players_data(self) -> Dict[str, Any]:
        """Đọc thông tin hồ sơ stats của tất cả người chơi từ JSON cục bộ trước."""
        # 1. Đọc từ local file trước
        if os.path.exists(LOCAL_PLAYERS_FILE):
            try:
                with open(LOCAL_PLAYERS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DataManager] Read local players.json error: {e}")

        # 2. Thử đọc từ GitHub nếu chưa có local file
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
        """Lưu thông tin hồ sơ người chơi vào JSON (local + GitHub)."""
        saved_local = False
        saved_remote = False

        # Lưu local
        try:
            with open(LOCAL_PLAYERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(players_data, f, ensure_ascii=False, indent=2)
            saved_local = True
        except Exception as e:
            print(f"[DataManager] Save local players.json error: {e}")

        # Lưu GitHub nếu có
        if self.github_repo:
            try:
                json_content = json.dumps(players_data, ensure_ascii=False, indent=2)
                try:
                    file_item = self.github_repo.get_contents(PLAYERS_FILE_PATH)
                    self.github_repo.update_file(PLAYERS_FILE_PATH, "Update players.json", json_content, file_item.sha)
                except Exception:
                    self.github_repo.create_file(PLAYERS_FILE_PATH, "Create players.json", json_content)
                saved_remote = True
            except Exception as e:
                print(f"[DataManager] Save GitHub players.json error: {e}")

        return saved_local or saved_remote


data_manager = DataManager()
