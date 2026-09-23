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

        winning_team_label = "Team 1 (Blue Side / Đội 1 Xanh)" if winner == 'team1' else "Team 2 (Red Side / Đội 2 Đỏ)"
        losing_team_label = "Team 2 (Red Side / Đội 2 Đỏ)" if winner == 'team1' else "Team 1 (Blue Side / Đội 1 Xanh)"

        prompt = f"""
You are a world-class eSports analyst (League of Legends, Dota 2, Valorant) and a competitive Elo rating mathematician.
Task: Analyze the attached post-game scoreboard screenshot (End-Game Scoreboard) with utmost precision.

MATCH CONTEXT:
- {winning_team_label} is the WINNER.
- {losing_team_label} is the LOSER.

CRITICAL INSTRUCTION ON SCREENSHOT LAYOUT & TEAM ORIENTATION:
- In the League of Legends client end-game screen, the screenshot taker's team is ALWAYS displayed on the TOP panel, regardless of whether they were Blue side or Red side!
- Therefore, the TOP panel is NOT necessarily Team 1 (Blue Side)! If a player from Team 2 (Red Side) captured the screenshot, Team 2 will be shown at the top.
- You MUST cross-reference each player's in-game name and champion against the roster below to accurately map each player to [TEAM 1 (BLUE SIDE)] or [TEAM 2 (RED SIDE)].

STRICT ELO DELTA SIGN RULES:
- Winning team players ({winning_team_label}) MUST RECEIVE A POSITIVE Elo delta (+) ranging from +3 to +28 (MVP, GREAT, SOLID, PASSENGER).
- Losing team players ({losing_team_label}) MUST RECEIVE A NEGATIVE Elo delta (-) ranging from -5 to -26 (SVP, SOLID, FEEDER).
- UNDER NO CIRCUMSTANCES should a winning player receive a negative delta, or a losing player receive a positive delta!

10 PARTICIPATING PLAYERS ROSTER:
[TEAM 1 (BLUE SIDE)]:
{chr(10).join(t1_lines)}

[TEAM 2 (RED SIDE)]:
{chr(10).join(t2_lines)}

DATA EXTRACTION & ELO CALCULATION GUIDELINES:
1. Extract stats for all 10 players from the scoreboard:
   - In-game summoner name and champion played (matched with the 10 players listed above).
   - KDA (Kills / Deaths / Assists, e.g., "13/7/6").
   - Secondary metrics: Damage (e.g., "24.5k"), Gold, CS, MVP, SVP/ACE badges.
   - Total team kills: `team1_kills` and `team2_kills` (sum of individual kills per team, or read from team kill totals).

2. CRITICAL DISTINCTION: "CARRY / PLAYMAKER" VS "PASSENGER / CARRIED":
   - Winning team players with poor contributions (e.g., negative KDA like 1/7/2, 0/5/3, 2/9/4, bottom-tier damage, excessive deaths) are "PASSENGERS" who were merely carried by their teammates.
   - Passengers MUST NOT receive high Elo gains! Award them only a nominal +3 to +7 Elo (Tag: "PASSENGER").
   - Conversely, primary Carries / MVPs with dominant KDA and top damage must be handsomely rewarded (+24 to +28 Elo, Tag: "MVP").

3. DETAILED PERFORMANCE CATEGORIES & DELTA RANGES:
   - WINNING TEAM:
     * [MVP] Primary Carry: Dominant KDA, top-tier damage, clutch playmaking. Score: 9.0 - 10.0 -> Delta: +24 to +28. Tag: "MVP".
     * [GREAT] Major Contributor / Pillar: Solid KDA, high teamfight participation. Score: 7.5 - 8.9 -> Delta: +18 to +22. Tag: "GREAT".
     * [SOLID] Reliable / Role Player: Won lane or held even, balanced KDA. Score: 6.0 - 7.4 -> Delta: +14 to +16. Tag: "SOLID".
     * [PASSENGER] Carried / Burden: Heavily negative KDA, lowest damage, repeatedly caught out, won only due to stellar teammates. Score: 2.5 - 4.9 -> Delta: ONLY +3 to +7. Tag: "PASSENGER".
   - LOSING TEAM:
     * [SVP] Valiant Effort / Bright Spot: Strong KDA and high damage despite defeat. Score: 7.5 - 8.9 -> Delta: ONLY -5 to -9 (protect standout performance). Tag: "SVP".
     * [SOLID] Fair / Decent: Tried their best but unable to turn the tide. Score: 5.0 - 6.9 -> Delta: -12 to -15. Tag: "SOLID".
     * [FEEDER] Underperforming / Feeder: Excessive deaths, negative momentum, dragged team down. Score: 1.0 - 4.4 -> Delta: -20 to -26. Tag: "FEEDER".

RETURN ONLY A VALID JSON OBJECT (no markdown backticks, no extra text):
{{
  "winner": "{winner}",
  "match_mvp": "Player name of the MVP",
  "match_svp": "Player name of the SVP",
  "team1_kills": 0,
  "team2_kills": 0,
  "ai_summary": "A concise 2-3 sentence match summary in Vietnamese highlighting the tactical turning points, key carries, and decisive plays.",
  "players_analysis": [
    {{
      "player_id": "exact_id_from_roster",
      "nickname": "Player nickname",
      "team": 1,
      "champion": "Champion name (e.g. Lucian)",
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
`team1_kills` and `team2_kills` are the total kills for Team 1 and Team 2 respectively.
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

            normalized_list = []
            for p in team1_players:
                found = find_analysis_for_player(p)
                default_delta = 16.0 if winner == 'team1' else -16.0
                if found:
                    raw_delta = float(found.get('recommended_delta', default_delta))
                    raw_tag = str(found.get('performance_tag', 'SOLID')).upper()
                    # Bảo đảm dấu điểm Elo luôn chuẩn xác theo kết quả thắng/thua
                    if winner == 'team1':
                        delta = abs(raw_delta) if raw_delta != 0 else 16.0
                        tag = 'MVP' if raw_tag in ['MVP', 'SVP'] else ('GREAT' if raw_tag == 'GREAT' else ('PASSENGER' if raw_tag == 'FEEDER' else raw_tag))
                    else:
                        delta = -abs(raw_delta) if raw_delta != 0 else -16.0
                        tag = 'SVP' if raw_tag in ['MVP', 'SVP'] else ('FEEDER' if raw_tag in ['FEEDER', 'PASSENGER'] else 'SOLID')

                    normalized_list.append({
                        "player_id": p['id'],
                        "nickname": p.get('nickname', p['id']),
                        "team": 1,
                        "champion": found.get('champion', '-'),
                        "kda": found.get('kda', '-'),
                        "damage": found.get('damage', '-'),
                        "performance_score": float(found.get('performance_score', 6.0)),
                        "performance_tag": tag,
                        "recommended_delta": round(delta, 1),
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
                        "performance_score": 6.5 if winner == 'team1' else 5.0,
                        "performance_tag": 'SOLID',
                        "recommended_delta": default_delta,
                        "comment": 'Tròn vai theo diễn biến trận đấu'
                    })

            for p in team2_players:
                found = find_analysis_for_player(p)
                default_delta = 16.0 if winner == 'team2' else -16.0
                if found:
                    raw_delta = float(found.get('recommended_delta', default_delta))
                    raw_tag = str(found.get('performance_tag', 'SOLID')).upper()
                    # Bảo đảm dấu điểm Elo luôn chuẩn xác theo kết quả thắng/thua
                    if winner == 'team2':
                        delta = abs(raw_delta) if raw_delta != 0 else 16.0
                        tag = 'MVP' if raw_tag in ['MVP', 'SVP'] else ('GREAT' if raw_tag == 'GREAT' else ('PASSENGER' if raw_tag == 'FEEDER' else raw_tag))
                    else:
                        delta = -abs(raw_delta) if raw_delta != 0 else -16.0
                        tag = 'SVP' if raw_tag in ['MVP', 'SVP'] else ('FEEDER' if raw_tag in ['FEEDER', 'PASSENGER'] else 'SOLID')

                    normalized_list.append({
                        "player_id": p['id'],
                        "nickname": p.get('nickname', p['id']),
                        "team": 2,
                        "champion": found.get('champion', '-'),
                        "kda": found.get('kda', '-'),
                        "damage": found.get('damage', '-'),
                        "performance_score": float(found.get('performance_score', 6.0)),
                        "performance_tag": tag,
                        "recommended_delta": round(delta, 1),
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
                        "performance_score": 6.5 if winner == 'team2' else 5.0,
                        "performance_tag": 'SOLID',
                        "recommended_delta": default_delta,
                        "comment": 'Tròn vai theo diễn biến trận đấu'
                    })

            # Tính tổng mạng hạ gục từ AI hoặc suy ra từ KDA cá nhân
            t1k_ai = int(parsed.get('team1_kills', 0) or 0)
            t2k_ai = int(parsed.get('team2_kills', 0) or 0)

            if t1k_ai == 0 and t2k_ai == 0:
                # Fallback: tính từ cột Kills cá nhân trong normalized_list
                for item in normalized_list:
                    kda_str = item.get('kda', '-')
                    parts = str(kda_str).replace(' ', '').split('/')
                    try:
                        kills = int(parts[0])
                    except Exception:
                        kills = 0
                    if item.get('team') == 1:
                        t1k_ai += kills
                    else:
                        t2k_ai += kills

            # Tính match_closeness, is_stomp, balance_rating
            total_kills = t1k_ai + t2k_ai
            diff_kills = abs(t1k_ai - t2k_ai)
            closeness = round(1.0 - diff_kills / total_kills, 4) if total_kills > 0 else 0.5
            is_stomp = diff_kills > 15 or closeness < 0.40
            if diff_kills <= 5 or closeness >= 0.85:
                balance_rating = 'perfect'
            elif diff_kills <= 10 or closeness >= 0.60:
                balance_rating = 'fair'
            elif diff_kills <= 15 or closeness >= 0.40:
                balance_rating = 'unbalanced'
            else:
                balance_rating = 'stomp'

            return {
                "success": True,
                "winner": winner,
                "match_mvp": parsed.get('match_mvp', ''),
                "match_svp": parsed.get('match_svp', ''),
                "ai_summary": parsed.get('ai_summary', 'Đã phân tích thông số trận đấu thành công.'),
                "team1_kills": t1k_ai,
                "team2_kills": t2k_ai,
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

