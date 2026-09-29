import os
import json
import time
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

CANDIDATE_MODELS = [
    'gemini-2.5-flash-lite',
    'gemini-3-flash-preview',
    'gemini-3.5-flash',
    'gemini-3.5-flash-lite',
    'gemini-3.1-flash-lite',
    'gemini-flash-latest'
]


class GeminiService:
    def __init__(self):
        self.api_key = GEMINI_API_KEY

    def _call_gemini_api(self, payload: Dict[str, Any], custom_key: str = None, timeout: int = 45) -> Dict[str, Any]:
        """Gọi Gemini API với cơ chế tự động thử nhiều model khả dụng khi gặp lỗi 404/503/Timeout."""
        key = custom_key or self.api_key
        if not key:
            raise ValueError('Chưa cấu hình Gemini API Key. Vui lòng dán API Key vào ô nhập hoặc mục Cài Đặt.')

        last_error = None
        for model_name in CANDIDATE_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            t0 = time.time()
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    res_body = json.loads(response.read().decode('utf-8'))
                    elapsed = time.time() - t0
                    logger.info(f"[Gemini API] Model {model_name} thành công trong {round(elapsed, 2)}s")
                    return res_body
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode('utf-8', errors='ignore')
                last_error = f"Model {model_name} HTTP {e.code}: {err_msg}"
                logger.warning(f"[Gemini API] {last_error}. Đang thử model kế tiếp...")
                if e.code in [404, 503, 429]:
                    continue
                else:
                    raise Exception(last_error)
            except Exception as e:
                last_error = f"Model {model_name} error: {str(e)}"
                logger.warning(f"[Gemini API] {last_error}. Đang thử model kế tiếp...")
                continue

        raise Exception(f"Không thể kết nối đến bất kỳ model Gemini nào khả dụng. Chi tiết: {last_error}")

    def analyze_matchup(self, team1_data: List[Dict[str, Any]], team2_data: List[Dict[str, Any]], custom_key: str = None) -> Dict[str, Any]:
        """Phân tích kèo đấu và chiến thuật bằng Gemini AI hoặc phân tích heuristic thông minh."""
        key = custom_key or self.api_key

        t1_names = [p['nickname'] for p in team1_data]
        t2_names = [p['nickname'] for p in team2_data]

        t1_on_fire = [p['nickname'] for p in team1_data if p.get('form', {}).get('status') == 'on_fire']
        t2_on_fire = [p['nickname'] for p in team2_data if p.get('form', {}).get('status') == 'on_fire']

        if not key:
            analysis = (
                f"### ⚔️ Phân Tích Chiến Thuật Dự Đoán\n\n"
                f"- **Đội Xanh (Team 1)**: Quy tụ các ngòi nổ {', '.join(t1_names[:3])}. "
                f"{'Đang có tuyển thủ phong độ cực cao: ' + ', '.join(t1_on_fire) + ' 🔥.' if t1_on_fire else 'Đội hình đồng đều, giữ cự ly tốt.'}\n"
                f"- **Đội Đỏ (Team 2)**: Điểm tựa vững chắc với {', '.join(t2_names[:3])}. "
                f"{'Các tuyển thủ đang vào form: ' + ', '.join(t2_on_fire) + ' 🔥.' if t2_on_fire else 'Lối đánh kiểm soát và phối hợp ổn định.'}\n\n"
                f"**Chiến thuật đề xuất**:\n"
                f"- Team 1 nên tận dụng khả năng giao tranh tổng sớm.\n"
                f"- Team 2 cần kiểm soát mục tiêu lớn và kiên nhẫn đợi ngưỡng sức mạnh.\n\n"
                f"*(Mẹo: Bạn có thể nhập Gemini API Key trong phần Cài đặt để nhận phân tích chuyên sâu chi tiết từ AI)*"
            )
            return {
                'source': 'heuristic',
                'analysis': analysis
            }

        prompt = f"""
You are an elite eSports analyst and professional shoutcaster (League of Legends / Dota 2). Analyze the upcoming 5v5 matchup between the following two teams:

Team 1 (Blue Side): {', '.join(t1_names)}
- High-form players: {', '.join(t1_on_fire) if t1_on_fire else 'Stable'}

Team 2 (Red Side): {', '.join(t2_names)}
- High-form players: {', '.join(t2_on_fire) if t2_on_fire else 'Stable'}

Provide a concise, high-energy 3-4 paragraph preview in Vietnamese (professional caster tone):
1. Team strengths comparison, lane matchup dynamics, and potential playmakers / X-factors.
2. Clear Win Conditions for each team.
3. MVP prediction and key player to watch.

Use clear markdown headings and bullet points for maximum readability.
"""
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            res_data = self._call_gemini_api(payload, custom_key=key, timeout=25)
            ai_text = res_data['candidates'][0]['content']['parts'][0]['text']
            return {
                'source': 'gemini',
                'analysis': ai_text
            }
        except Exception as e:
            return {
                'source': 'fallback',
                'analysis': f"Không thể kết nối Gemini API ({str(e)}). Vui lòng kiểm tra lại API Key."
            }

    def recognize_players_from_image(self, image_data: str, mime_type: str, known_players: List[Dict[str, Any]], custom_key: str = None) -> Dict[str, Any]:
        """
        Nhận diện linh hoạt người chơi từ ảnh chụp màn hình (Lobby custom 5vs5, scoreboard, hoặc ảnh 5 người).
        Tự động trích xuất tên theo 2 đội và ánh xạ với danh sách tuyển thủ hệ thống.
        """
        key = custom_key or self.api_key
        if not key:
            return {
                'success': False,
                'error': 'Chưa cấu hình Gemini API Key. Vui lòng dán API Key vào ô nhập hoặc mục Cài Đặt.'
            }

        # Tạo danh sách tuyển thủ đã biết để AI đối chiếu chính xác
        players_reference = []
        valid_ids_map = {}
        # Ánh xạ bí danh đặc biệt (ví dụ: người chơi nyan đã gộp/chuyển giao sang hungpui)
        SPECIAL_ALIASES = {
            'nyan': 'hungpui',
            'hungpui': 'hungpui'
        }

        for p in known_players:
            pid = str(p['id']).strip().lower()
            nick = str(p.get('nickname', pid)).strip()
            valid_ids_map[pid] = pid
            valid_ids_map[nick.lower()] = pid

            base_nick = ""
            if '#' in nick:
                base_nick = nick.split('#')[0].strip()
                if base_nick:
                    valid_ids_map[base_nick.lower()] = pid

            for al in p.get('aliases', []):
                if al:
                    valid_ids_map[str(al).strip().lower()] = pid

            if base_nick:
                players_reference.append(f"- ID: {p['id']}, Nickname: {nick} (Ingame lobby name: '{base_nick}')")
            else:
                players_reference.append(f"- ID: {p['id']}, Nickname: {nick}")

        # Gán bổ sung các alias đặc biệt
        for alias_k, target_id in SPECIAL_ALIASES.items():
            valid_ids_map[alias_k.lower()] = target_id

        ref_text = "\n".join(players_reference)

        prompt = f"""
You are an expert AI vision system specialized in gaming lobbies and match scoreboards (League of Legends custom lobbies, Discord rooms, in-game scoreboards, loading screens).
Task: Inspect the attached screenshot and extract all participating player names.

CRITICAL INSTRUCTIONS:
- The image may depict a 5v5 custom lobby (Blue vs Red side, Left vs Right columns, or Team 1 vs Team 2).
- Alternatively, it might only show 5 players from a single team (team card, end-game banner, etc.).
- Accurately transcribe all visible player names (ignore rank borders, summoner levels, ping indicators, and extraneous clan tags).
- Cross-reference every detected name against the registered system roster below:

=== SYSTEM REGISTERED PLAYER ROSTER ===
{ref_text}
======================================

MATCHING RULES:
1. Exact or Fuzzy Match: If a detected name matches or closely resembles (case-insensitive, whitespace variations, special character/icon differences) a registered player, set `matched_id` to that player's exact system ID.
2. RIOT ID & IN-GAME SUMMONER NAMES:
   - In League of Legends lobbies, summoner names usually appear without their Riot tagline (e.g., 'Nyan#Tabby' displays simply as 'Nyan').
   - Special alias mapping: The name 'Nyan' or 'nyan' MUST be mapped to the registered system ID 'hungpui'.
   - NEVER invent new IDs or return an unverified ID; only use the exact system IDs provided above.
3. Unrecognized Players: If a detected name does not match any registered player in the roster, set `matched_id` to null, but ALWAYS preserve the exact `raw_name` extracted from the image so the administrator can register them.
4. Team Allocation:
   - `team1`: List of players in Team 1 (Blue Side / Left Column / first 5 slots). Maximum 5 players.
   - `team2`: List of players in Team 2 (Red Side / Right Column / second 5 slots). Maximum 5 players (if only 1 team is present in the image, provide an empty list `[]`).

RETURN ONLY A VALID JSON OBJECT (no markdown backticks, no extra text):
{{
  "team1": [
    {{"raw_name": "Detected name in image", "matched_id": "system_player_id_or_null"}}
  ],
  "team2": [
    {{"raw_name": "Detected name in image", "matched_id": "system_player_id_or_null"}}
  ],
  "detected_names": ["List of all detected names from image"]
}}
"""
        clean_base64 = image_data
        if ',' in image_data:
            clean_base64 = image_data.split(',', 1)[1]

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type or "image/jpeg",
                            "data": clean_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        try:
            res_data = self._call_gemini_api(payload, custom_key=key, timeout=45)
            raw_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()

            if raw_text.startswith('```json'):
                raw_text = raw_text[7:]
            if raw_text.startswith('```'):
                raw_text = raw_text[3:]
            if raw_text.endswith('```'):
                raw_text = raw_text[:-3]

            parsed = json.loads(raw_text.strip())

            raw_team1 = parsed.get('team1', [])
            raw_team2 = parsed.get('team2', [])
            detected_names = parsed.get('detected_names', [])

            # Chuẩn hóa slots cho 5vs5 (mỗi team đủ 5 slot)
            known_ids_set = {p['id'].lower() for p in known_players}

            def find_player_id(text: str) -> Optional[str]:
                if not text:
                    return None
                t = str(text).strip().lower()
                # 1. Tra cứu trực tiếp
                if t in valid_ids_map:
                    return valid_ids_map[t]
                # 2. Bỏ Riot tag sau dấu '#' (ví dụ: "nyan#tabby" -> "nyan")
                if '#' in t:
                    base = t.split('#')[0].strip()
                    if base in valid_ids_map:
                        return valid_ids_map[base]
                # 3. Tìm theo dạng ngoặc đơn/phụ bản (ví dụ: "nyan(hungpui)")
                for k, target_pid in valid_ids_map.items():
                    if len(k) >= 3 and (k in t or t in k):
                        return target_pid
                return None

            def normalize_slot(item, idx):
                if not item:
                    return {"slot": idx + 1, "raw_name": "", "matched_id": None}
                raw_name = str(item.get('raw_name', '')).strip()
                matched_id = item.get('matched_id')

                final_id = None
                if matched_id:
                    clean_mid = str(matched_id).strip().lower()
                    if clean_mid in known_ids_set:
                        final_id = clean_mid
                    else:
                        final_id = find_player_id(clean_mid) or find_player_id(raw_name)
                elif raw_name:
                    final_id = find_player_id(raw_name)

                # Đảm bảo final_id phải là 1 ID hợp lệ trong known_ids_set
                if final_id and final_id not in known_ids_set:
                    final_id = None

                return {
                    "slot": idx + 1,
                    "raw_name": raw_name,
                    "matched_id": final_id
                }

            team1_slots = [normalize_slot(raw_team1[i] if i < len(raw_team1) else None, i) for i in range(5)]
            team2_slots = [normalize_slot(raw_team2[i] if i < len(raw_team2) else None, i) for i in range(5)]

            # Danh sách tất cả các ID đã khớp
            all_matched_ids = []
            for slot in team1_slots + team2_slots:
                if slot['matched_id'] and slot['matched_id'] not in all_matched_ids:
                    all_matched_ids.append(slot['matched_id'])

            return {
                'success': True,
                'team1_slots': team1_slots,
                'team2_slots': team2_slots,
                'matched_player_ids': all_matched_ids,
                'detected_names': detected_names,
                'count': len(all_matched_ids),
                'total_detected': len(detected_names)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Lỗi nhận diện ảnh từ Gemini: {str(e)}"
            }

    def analyze_match_scoreboard(
        self,
        image_data: str,
        mime_type: str,
        team1_players: List[Dict[str, Any]],
        team2_players: List[Dict[str, Any]],
        winner: str = 'team1',
        custom_key: str = None
    ) -> Dict[str, Any]:
        """
        Phân tích ảnh chụp màn hình bảng điểm sau trận đấu (Scoreboard / End-game screen).
        AI tự động đọc KDA, sát thương, vàng, danh hiệu MVP/SVP, và tính toán điểm Elo phù hợp cho từng tuyển thủ.
        """
        key = custom_key or self.api_key
        if not key:
            return {
                'success': False,
                'error': 'Chưa cấu hình Gemini API Key. Vui lòng dán API Key vào ô nhập hoặc mục Cài Đặt.'
            }

        def format_player_line(p):
            pid = p.get('id', '')
            nick = p.get('nickname', pid)
            base = nick.split('#')[0].strip() if '#' in nick else ""
            if base and base.lower() != nick.lower():
                return f"- ID: {pid}, Name: {nick} (In-game display in screenshot: '{base}')"
            return f"- ID: {pid}, Name: {nick}"

        t1_lines = [format_player_line(p) for p in team1_players]
        t2_lines = [format_player_line(p) for p in team2_players]

        winning_team_label = "Team 1 (Đội 1 Xanh / Blue Side)" if winner == 'team1' else "Team 2 (Đội 2 Đỏ / Red Side)"
        losing_team_label = "Team 2 (Đội 2 Đỏ / Red Side)" if winner == 'team1' else "Team 1 (Đội 1 Xanh / Blue Side)"

        prompt = f"""
You are a world-class eSports analyst (League of Legends, Dota 2, Valorant) and a competitive Elo rating mathematician.
Task: Analyze the attached post-game scoreboard screenshot (End-Game Scoreboard) with utmost precision.

MATCH CONTEXT & RESULT (GROUND TRUTH):
- {winning_team_label} is the WINNER. All players on this team MUST receive a POSITIVE Elo delta (+3 to +28).
- {losing_team_label} is the LOSER. All players on this team MUST receive a NEGATIVE Elo delta (-5 to -26).

SCREENSHOT LAYOUT IN LEAGUE OF LEGENDS:
- The TOP panel shows 'ĐỘI 1' (Team 1, Blue Side).
- The BOTTOM panel shows 'ĐỘI 2' (Team 2, Red Side).
- Map each of the 10 players from the scoreboard to their exact player_id in the roster below.
- CRITICAL: Do NOT duplicate stats or champions across players. Each player has their own row.

CRITICAL INSTRUCTIONS ON MVP & SVP:
- WINNING TEAM MVP: Awarded ONLY to the single best player on the WINNING team ({winning_team_label}). Tag: "MVP", Delta: +24 to +28.
- LOSING TEAM SVP: Awarded ONLY to the standout best player on the LOSING team ({losing_team_label}). Tag: "SVP", Delta: -5 to -9 (they valiantly tried to carry, so they lose the LEAST Elo).
- STRICT RULE: MVP and SVP MUST BE ON OPPOSITE TEAMS! (MVP on the winning team, SVP on the losing team).
- NEVER give MVP to a player on the losing team, and NEVER give SVP to a player on the winning team!

STRICT RULES ON ELO DELTA DIRECTION:
- WINNING TEAM PLAYERS ({winning_team_label}) MUST HAVE POSITIVE (+) DELTAS:
  * [MVP] Primary Carry (best on winning team): Score: 9.0 - 10.0 -> Delta: +24 to +28. Tag: "MVP".
  * [GREAT] Major Contributor / High damage/KP: Score: 7.5 - 8.9 -> Delta: +18 to +22. Tag: "GREAT".
  * [SOLID] Reliable / Role Player: Score: 6.0 - 7.4 -> Delta: +12 to +16. Tag: "SOLID".
  * [PASSENGER] Carried / High deaths / Burden: Score: 2.5 - 4.9 -> Delta: ONLY +3 to +7. Tag: "PASSENGER". (Lowest gain on winning team!).
- LOSING TEAM PLAYERS ({losing_team_label}) MUST HAVE NEGATIVE (-) DELTAS:
  * [SVP] Standout Effort / Valiant Carry: Score: 7.5 - 9.5 -> Delta: ONLY -5 to -9. Tag: "SVP". (Smallest loss on losing team!).
  * [SOLID] Fair / Decent effort: Score: 5.0 - 6.9 -> Delta: -12 to -16. Tag: "SOLID".
  * [FEEDER] Underperforming / Feeder (high deaths, 0 kills): Score: 1.0 - 4.4 -> Delta: -20 to -26. Tag: "FEEDER". (Biggest loss on losing team!).

10 PARTICIPATING PLAYERS ROSTER:
[TEAM 1 (BLUE SIDE)]:
{chr(10).join(t1_lines)}

[TEAM 2 (RED SIDE)]:
{chr(10).join(t2_lines)}

DATA EXTRACTION:
1. Extract stats for all 10 players from the scoreboard:
   - In-game summoner name and champion played.
   - KDA (Kills / Deaths / Assists, e.g. "13/7/6").
   - Secondary metrics: Damage (e.g. "24.5k"), Gold, CS.
   - Total team kills: `team1_kills` must be the total kills of Team 1, and `team2_kills` must be total kills of Team 2.

RETURN ONLY A VALID JSON OBJECT (no markdown backticks, no extra text):
{{
  "winner": "{winner}",
  "match_mvp": "Player name of the MVP (MUST be from {winning_team_label})",
  "match_svp": "Player name of the SVP (MUST be from {losing_team_label})",
  "team1_kills": 0,
  "team2_kills": 0,
  "ai_summary": "A concise 2-3 sentence match summary in Vietnamese highlighting key carries and decisive plays.",
  "players_analysis": [
    {{
      "player_id": "exact_id_from_roster",
      "nickname": "Player nickname",
      "team": 1,
      "champion": "Champion name (e.g. Smolder)",
      "kda": "13/7/6",
      "damage": "24.5k",
      "performance_score": 9.2,
      "performance_tag": "MVP",
      "recommended_delta": 24.0,
      "comment": "One concise sentence in Vietnamese explaining this rating/delta"
    }}
  ]
}}
Note: The `players_analysis` array MUST contain all 10 players (5 for team 1, 5 for team 2) with their exact `player_id`.
"""
        clean_base64 = image_data
        if ',' in image_data:
            clean_base64 = image_data.split(',', 1)[1]

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type or "image/jpeg",
                            "data": clean_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        try:
            res_data = self._call_gemini_api(payload, custom_key=key, timeout=60)
            raw_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()

            if raw_text.startswith('```json'):
                raw_text = raw_text[7:]
            if raw_text.startswith('```'):
                raw_text = raw_text[3:]
            if raw_text.endswith('```'):
                raw_text = raw_text[:-3]

            parsed = json.loads(raw_text.strip())

            # Chuẩn hóa lại map người chơi để đảm bảo đủ 10 người và có recommended_delta hợp lệ
            players_analysis = parsed.get('players_analysis', [])
            analysis_by_id = {str(item.get('player_id', '')).strip().lower(): item for item in players_analysis}
            analysis_by_nick = {}
            for item in players_analysis:
                n = str(item.get('nickname', '')).strip().lower()
                analysis_by_nick[n] = item
                if '#' in n:
                    analysis_by_nick[n.split('#')[0].strip()] = item

            # Ánh xạ bí danh đặc biệt (nyan -> hungpui)
            SPECIAL_MAP = {'nyan': 'hungpui'}
            for old_id, new_id in SPECIAL_MAP.items():
                if old_id in analysis_by_id and new_id not in analysis_by_id:
                    analysis_by_id[new_id] = analysis_by_id[old_id]

            def find_analysis_for_player(p_info):
                p_id = str(p_info['id']).strip().lower()
                p_nick = str(p_info.get('nickname', '')).strip().lower()
                p_base = p_nick.split('#')[0].strip() if '#' in p_nick else ""
                return (
                    analysis_by_id.get(p_id) or
                    analysis_by_nick.get(p_nick) or
                    (analysis_by_nick.get(p_base) if p_base else None) or
                    (analysis_by_id.get('nyan') if p_id == 'hungpui' else None) or
                    (analysis_by_nick.get('nyan') if p_id == 'hungpui' else None)
                )

            def normalize_delta_and_tag(raw_delta: float, raw_tag: str, is_winner: bool, score: float = 6.0) -> Tuple[float, str]:
                tag = str(raw_tag or 'SOLID').upper().strip()
                raw_abs = abs(raw_delta) if raw_delta != 0 else 16.0

                if is_winner:
                    if tag in ['MVP', 'CARRY']:
                        final_tag = 'MVP'
                        final_delta = max(22.0, min(28.0, raw_abs if raw_abs >= 20.0 else 24.0))
                    elif tag == 'GREAT':
                        final_tag = 'GREAT'
                        final_delta = max(17.0, min(22.0, raw_abs if 16.0 <= raw_abs <= 24.0 else 20.0))
                    elif tag in ['PASSENGER', 'CARRIED', 'FEEDER']:
                        final_tag = 'PASSENGER'
                        final_delta = max(3.0, min(8.0, raw_abs if raw_abs <= 10.0 else 6.0))
                    else:
                        final_tag = 'SOLID'
                        final_delta = max(11.0, min(16.0, raw_abs if 10.0 <= raw_abs <= 18.0 else 14.0))
                    return round(final_delta, 1), final_tag
                else:
                    if tag in ['SVP', 'MVP', 'CARRY']:
                        final_tag = 'SVP'
                        final_delta = -max(5.0, min(9.0, raw_abs if raw_abs <= 10.0 else 6.0))
                    elif tag == 'GREAT':
                        final_tag = 'GREAT'
                        final_delta = -max(7.0, min(10.0, raw_abs if raw_abs <= 12.0 else 8.0))
                    elif tag in ['FEEDER', 'PASSENGER', 'CARRIED']:
                        final_tag = 'FEEDER'
                        final_delta = -max(18.0, min(26.0, raw_abs if raw_abs >= 16.0 else 20.0))
                    else:
                        final_tag = 'SOLID'
                        final_delta = -max(11.0, min(16.0, raw_abs if 10.0 <= raw_abs <= 18.0 else 14.0))
                    return round(final_delta, 1), final_tag

            normalized_list = []
            for p in team1_players:
                found = find_analysis_for_player(p)
                is_win = (winner == 'team1')
                default_delta = 16.0 if is_win else -16.0
                if found:
                    raw_delta = float(found.get('recommended_delta', default_delta))
                    raw_tag = str(found.get('performance_tag', 'SOLID')).upper()
                    perf_score = float(found.get('performance_score', 6.0))
                    delta, tag = normalize_delta_and_tag(raw_delta, raw_tag, is_win, perf_score)

                    normalized_list.append({
                        "player_id": p['id'],
                        "nickname": p.get('nickname', p['id']),
                        "team": 1,
                        "champion": found.get('champion', '-'),
                        "kda": found.get('kda', '-'),
                        "damage": found.get('damage', '-'),
                        "performance_score": perf_score,
                        "performance_tag": tag,
                        "recommended_delta": delta,
                        "comment": found.get('comment', 'Thi đấu tròn vai')
                    })
                else:
                    normalized_list.append({
                        "player_id": p['id'],
                        "nickname": p.get('nickname', p['id']),
                        "team": 1,
                        "champion": '-',
                        "kda": '-',
                        "damage": '-',
                        "performance_score": 6.5 if is_win else 5.0,
                        "performance_tag": 'SOLID',
                        "recommended_delta": default_delta,
                        "comment": 'Tròn vai theo diễn biến trận đấu'
                    })

            for p in team2_players:
                found = find_analysis_for_player(p)
                is_win = (winner == 'team2')
                default_delta = 16.0 if is_win else -16.0
                if found:
                    raw_delta = float(found.get('recommended_delta', default_delta))
                    raw_tag = str(found.get('performance_tag', 'SOLID')).upper()
                    perf_score = float(found.get('performance_score', 6.0))
                    delta, tag = normalize_delta_and_tag(raw_delta, raw_tag, is_win, perf_score)

                    normalized_list.append({
                        "player_id": p['id'],
                        "nickname": p.get('nickname', p['id']),
                        "team": 2,
                        "champion": found.get('champion', '-'),
                        "kda": found.get('kda', '-'),
                        "damage": found.get('damage', '-'),
                        "performance_score": perf_score,
                        "performance_tag": tag,
                        "recommended_delta": delta,
                        "comment": found.get('comment', 'Thi đấu tròn vai')
                    })
                else:
                    normalized_list.append({
                        "player_id": p['id'],
                        "nickname": p.get('nickname', p['id']),
                        "team": 2,
                        "champion": '-',
                        "kda": '-',
                        "damage": '-',
                        "performance_score": 6.5 if is_win else 5.0,
                        "performance_tag": 'SOLID',
                        "recommended_delta": default_delta,
                        "comment": 'Tròn vai theo diễn biến trận đấu'
                    })

            # Tính tổng mạng hạ gục trực tiếp từ KDA cá nhân của từng đội để tránh nhầm lẫn giữa nhãn hiển thị và phe đấu
            t1_kills_sum = 0
            t2_kills_sum = 0
            for item in normalized_list:
                kda_str = str(item.get('kda', '-')).replace(' ', '').split('/')
                try:
                    k = int(kda_str[0])
                except Exception:
                    k = 0
                if item.get('team') == 1:
                    t1_kills_sum += k
                else:
                    t2_kills_sum += k

            # Luôn ưu tiên dùng tổng kill tính chính xác từ từng tuyển thủ
            if t1_kills_sum > 0 or t2_kills_sum > 0:
                final_t1_kills = t1_kills_sum
                final_t2_kills = t2_kills_sum
            else:
                final_t1_kills = int(parsed.get('team1_kills', 0) or 0)
                final_t2_kills = int(parsed.get('team2_kills', 0) or 0)

            winning_team_num = 1 if winner == 'team1' else 2
            losing_team_num = 2 if winner == 'team1' else 1

            winning_players = [p for p in normalized_list if p.get('team') == winning_team_num]
            losing_players = [p for p in normalized_list if p.get('team') == losing_team_num]

            def player_kda_tuple(p):
                kda_parts = str(p.get('kda', '0/0/0')).replace(' ', '').split('/')
                k = int(kda_parts[0]) if len(kda_parts) > 0 and kda_parts[0].isdigit() else 0
                d = int(kda_parts[1]) if len(kda_parts) > 1 and kda_parts[1].isdigit() else 1
                a = int(kda_parts[2]) if len(kda_parts) > 2 and kda_parts[2].isdigit() else 0
                return (float(p.get('performance_score', 0)), (k + a) / max(1, d), k)

            # Sanity check: đảm bảo MVP thuộc về đội thắng
            calculated_mvp_name = parsed.get('match_mvp', '')
            if winning_players:
                best_winner = max(winning_players, key=lambda p: (
                    1 if p.get('performance_tag') == 'MVP' else 0,
                    player_kda_tuple(p)[0],
                    player_kda_tuple(p)[1]
                ))
                for p in winning_players:
                    if p == best_winner:
                        p['performance_tag'] = 'MVP'
                        p['recommended_delta'] = max(24.0, p['recommended_delta'])
                    elif p.get('performance_tag') == 'MVP':
                        p['performance_tag'] = 'GREAT'
                        p['recommended_delta'] = min(20.0, p['recommended_delta'])
                calculated_mvp_name = best_winner.get('nickname', calculated_mvp_name)

            # Sanity check: đảm bảo SVP thuộc về đội thua (không bao giờ gán cho người feeder / 0 kill)
            calculated_svp_name = parsed.get('match_svp', '')
            if losing_players:
                best_loser = max(losing_players, key=lambda p: (
                    1 if p.get('performance_tag') == 'SVP' else 0,
                    player_kda_tuple(p)[0],
                    player_kda_tuple(p)[1]
                ))
                for p in losing_players:
                    if p == best_loser:
                        p['performance_tag'] = 'SVP'
                        p['recommended_delta'] = max(-9.0, min(-5.0, p['recommended_delta']))
                    elif p.get('performance_tag') == 'SVP':
                        p['performance_tag'] = 'SOLID'
                        p['recommended_delta'] = -13.0
                calculated_svp_name = best_loser.get('nickname', calculated_svp_name)

            # Tính match_closeness, is_stomp, balance_rating bằng elo_service chuẩn
            from services.elo_service import elo_service
            closeness, balance_rating, is_stomp = elo_service.calc_match_closeness(final_t1_kills, final_t2_kills)

            return {
                "success": True,
                "winner": winner,
                "match_mvp": parsed.get('match_mvp', ''),
                "match_svp": calculated_svp_name,
                "ai_summary": parsed.get('ai_summary', 'Đã phân tích thông số trận đấu thành công.'),
                "team1_kills": final_t1_kills,
                "team2_kills": final_t2_kills,
                "match_closeness": closeness,
                "is_stomp": is_stomp,
                "balance_rating": balance_rating,
                "players_analysis": normalized_list
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Lỗi phân tích bảng điểm từ Gemini: {str(e)}"
            }


gemini_service = GeminiService()

