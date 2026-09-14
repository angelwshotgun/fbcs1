import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

CANDIDATE_MODELS = [
    'gemini-3.5-flash',
    'gemini-3.8-flash',
    'gemini-3.7-flash',
    'gemini-3.6-flash',
    'gemini-3.5-flash-lite'
]


class GeminiService:
    def __init__(self):
        self.api_key = GEMINI_API_KEY

    def _call_gemini_api(self, payload: Dict[str, Any], custom_key: str = None, timeout: int = 25) -> Dict[str, Any]:
        """Gọi Gemini API với cơ chế tự động thử nhiều model khả dụng khi gặp lỗi 404/503."""
        key = custom_key or self.api_key
        if not key:
            raise ValueError('Chưa cấu hình Gemini API Key. Vui lòng dán API Key vào ô nhập hoặc mục Cài Đặt.')

        last_error = None
        for model_name in CANDIDATE_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode('utf-8', errors='ignore')
                last_error = f"Model {model_name} HTTP {e.code}: {err_msg}"
                # Thử model tiếp theo nếu 404 (model deprecated) hoặc 503 (quá tải)
                if e.code in [404, 503, 429]:
                    continue
                else:
                    raise Exception(last_error)
            except Exception as e:
                last_error = f"Model {model_name} error: {str(e)}"
                continue

        raise Exception(last_error or 'Không thể kết nối đến bất kỳ model Gemini nào khả dụng.')

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
Bạn là chuyên gia phân tích eSports (LMHT/Dota) hàng đầu. Hãy phân tích kèo đấu giữa 2 đội sau:
Team 1 ({', '.join(t1_names)}):
- Phong độ cao: {', '.join(t1_on_fire) if t1_on_fire else 'Bình ổn'}
Team 2 ({', '.join(t2_names)}):
- Phong độ cao: {', '.join(t2_on_fire) if t2_on_fire else 'Bình ổn'}

Hãy viết ngắn gọn 3-4 đoạn:
1. Đánh giá tương quan lực lượng và điểm đột biến
2. Điều kiện thắng (Win Condition) cho từng đội
3. Dự đoán tuyển thủ có thể trở thành MVP
Phong cách sôi nổi, hào hứng của bình luận viên chuyên nghiệp.
"""
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            res_data = self._call_gemini_api(payload, custom_key=key, timeout=15)
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
        for p in known_players:
            pid = str(p['id']).lower()
            nick = str(p['nickname'])
            valid_ids_map[pid] = nick
            valid_ids_map[nick.lower()] = pid
            players_reference.append(f"- ID: {p['id']}, Nickname: {nick}")
        ref_text = "\n".join(players_reference)

        prompt = f"""
Bạn là AI chuyên gia đọc sảnh chờ game (League of Legends custom lobby, Discord room, game scoreboard, loading screen).
Nhiệm vụ: Xem bức ảnh chụp màn hình này và trích xuất danh sách tên người chơi tham gia.

LƯU Ý QUAN TRỌNG:
- Bức ảnh có thể là sảnh chờ 2 đội 5vs5 (Đội Xanh / Đội Đỏ, Trái / Phải, hoặc Team 1 / Team 2).
- Bức ảnh cũng có thể chỉ hiển thị 5 người của 1 đội (bảng kết quả trận đấu, thẻ đội 5 người, v.v.).
- Hãy đọc chính xác tất cả tên người chơi (bỏ qua các tiền tố rank, cấp độ, ping, clan tag thừa nếu có thể).
- Hãy đối chiếu mỗi tên đọc được với danh sách hệ thống dưới đây:

=== DANH SÁCH NGƯỜI CHƠI HỆ THỐNG ===
{ref_text}
=====================================

Quy tắc ánh xạ:
1. Nếu tên trong ảnh giống hoặc tương đương (cho phép khác biệt nhỏ về hoa/thường, dấu cách, icon) với tuyển thủ trong danh sách hệ thống -> gán `matched_id` là đúng ID đó.
2. Nếu tên trong ảnh KHÔNG có trong danh sách hệ thống -> gán `matched_id` là null (hoặc rỗng), nhưng VẪN PHẢI GIỮ `raw_name` đọc được từ ảnh để người dùng có thể tạo người chơi mới.
3. Phân bổ vào 2 đội:
   - `team1`: danh sách người chơi Đội 1 (hoặc Đội Xanh / Cột Trái / 5 người đầu tiên). Tối đa 5 người.
   - `team2`: danh sách người chơi Đội 2 (hoặc Đội Đỏ / Cột Phải). Tối đa 5 người (nếu ảnh chỉ có 1 đội thì để mảng rỗng `[]`).

BẮT BUỘC TRẢ VỀ ĐÚNG 1 ĐỐI TƯỢNG JSON (không có markdown backticks ```json):
{{
  "team1": [
    {{"raw_name": "Tên trên ảnh", "matched_id": "id_he_thong_hoac_null"}}
  ],
  "team2": [
    {{"raw_name": "Tên trên ảnh", "matched_id": "id_he_thong_hoac_null"}}
  ],
  "detected_names": ["Tất cả các tên đọc được từ ảnh"]
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
            res_data = self._call_gemini_api(payload, custom_key=key, timeout=25)
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

            def normalize_slot(item, idx):
                if not item:
                    return {"slot": idx + 1, "raw_name": "", "matched_id": None}
                raw_name = str(item.get('raw_name', '')).strip()
                matched_id = item.get('matched_id')
                if matched_id:
                    matched_id = str(matched_id).strip().lower()
                    if matched_id not in known_ids_set:
                        # Thử tìm theo nickname
                        found_pid = valid_ids_map.get(matched_id) or valid_ids_map.get(raw_name.lower())
                        matched_id = found_pid if found_pid in known_ids_set else None
                elif raw_name:
                    # Thử match lại theo raw_name
                    found_pid = valid_ids_map.get(raw_name.lower())
                    if found_pid and found_pid in known_ids_set:
                        matched_id = found_pid

                return {
                    "slot": idx + 1,
                    "raw_name": raw_name,
                    "matched_id": matched_id
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

        t1_lines = [f"- ID: {p.get('id')}, Tên: {p.get('nickname')}" for p in team1_players]
        t2_lines = [f"- ID: {p.get('id')}, Tên: {p.get('nickname')}" for p in team2_players]

        winning_team_label = "Đội 1 (Xanh / Team 1)" if winner == 'team1' else "Đội 2 (Đỏ / Team 2)"
        losing_team_label = "Đội 2 (Đỏ / Team 2)" if winner == 'team1' else "Đội 1 (Xanh / Team 1)"

        prompt = f"""
Bạn là chuyên gia phân tích eSports (LMHT, Dota 2, Valorant) và chuyên gia toán học xếp hạng Elo.
Nhiệm vụ: Hãy xem ảnh chụp màn hình bảng thống kê chi tiết kết thúc trận đấu (End-game Scoreboard) đính kèm.

Bối cảnh trận đấu:
- {winning_team_label} là ĐỘI CHIẾN THẮNG (WINNER).
- {losing_team_label} là ĐỘI THUA CUỘC (LOSER).

Danh sách 10 tuyển thủ tham gia:
[ĐỘI 1 (XANH)]:
{chr(10).join(t1_lines)}

[ĐỘI 2 (ĐỎ)]:
{chr(10).join(t2_lines)}

HƯỚNG DẪN ĐỌC THÔNG SỐ VÀ TÍNH ĐIỂM ELO:
1. Đọc thông số của từng tuyển thủ từ bảng điểm trên ảnh:
   - Tên tuyển thủ và tướng/vị trí tương ứng (khớp với danh sách 10 người chơi trên).
   - Chỉ số KDA (Hạ gục / Bị hạ / Hỗ trợ, ví dụ "14/2/9").
   - Chỉ số phụ (nếu thấy): Sát thương (Damage), Vàng (Gold), CS (Lính), MVP, SVP/ACE.
2. ĐẶC BIỆT CHÚ Ý PHÂN BIỆT "NGƯỜI DẪN DẮT (CARRY)" VÀ "KẺ HƯỞNG KÉ / ĐƯỢC GÁNH (PASSENGER)":
   - Rất nhiều người chơi trong đội thắng nhưng thực tế KHÔNG HỀ DẪN DẮT hay đóng góp gì, thậm chí là gánh nặng (feed, chết liên tục, sát thương đáy bảng) và chỉ "HƯỞNG KÉ CHIẾN THẮNG" do đồng đội quá xuất sắc gánh.
   - Những người "hưởng ké" này TUYỆT ĐỐI KHÔNG ĐƯỢC NHẬN ĐIỂM ELO CAO, điểm Elo của họ chỉ được tăng tượng trưng từ +2 đến +7 Elo (thay vì mức +16 của cả đội)!
   - Ngược lại, những tuyển thủ thực sự là đầu tàu dẫn dắt (MVP/Carry) phải nhận điểm Elo vượt trội xứng đáng (+24 đến +28 Elo).

PHÂN LOẠI & ĐỀ XUẤT ĐIỂM ELO CHI TIẾT:
   - ĐỘI THẮNG (WINNER):
     + [MVP / CARRY] - Người Dẫn Dắt / Gánh Đội Xuất Sắc: KDA áp đảo, sát thương top đầu, mở giao tranh then chốt. Điểm: 9.0 - 10.0 -> Đề xuất Elo: +24 đến +28. Tag: "MVP"
     + [GREAT] - Đóng Góp Lớn / Trụ Cột: KDA đẹp, phối hợp chặt chẽ, tạo đột biến. Điểm: 7.5 - 8.9 -> Đề xuất Elo: +18 đến +22. Tag: "GREAT"
     + [SOLID] - Tròn Vai / Bình Ổn: Hoàn thành nhiệm vụ ở đường, KDA cân bằng. Điểm: 6.0 - 7.4 -> Đề xuất Elo: +14 đến +16. Tag: "SOLID"
     + [PASSENGER] - HƯỞNG KÉ / ĐƯỢC GÁNH: KDA âm sâu (ví dụ 1/7/2, 0/5/3, 2/9/4), sát thương thấp nhất đội, chết nhiều ở giai đoạn đi đường, gần như không có tác động tới chiến thắng mà chỉ hưởng ké thành quả của đồng đội. Điểm: 2.5 - 4.9 -> Đề xuất Elo: CHỈ +3 ĐẾN +7 ELO. Tag: "PASSENGER"
   - ĐỘI THUA (LOSER):
     + [SVP] - Nỗ Lực Gánh Đội Thua / Điểm Sáng Đơn Độc: KDA tốt, sát thương cao, chơi kiên cường nhưng đồng đội quá đuối. Điểm: 7.5 - 8.9 -> Đề xuất Elo: CHỈ TRỪ NHẸ -5 ĐẾN -9 ELO (để bảo vệ tuyển thủ chơi tốt). Tag: "SVP"
     + [SOLID] - Khá / Tròn Vai: Cố gắng thi đấu nhưng không lật được kèo. Điểm: 5.0 - 6.9 -> Đề xuất Elo: -12 đến -15. Tag: "SOLID"
     + [FEEDER] - Phá Game / Thọt Nặng / Tạ Của Đội: Feed mạng liên tục, mất kiểm soát, kéo cả đội xuống. Điểm: 1.0 - 4.4 -> Đề xuất Elo: TRỪ NẶNG -20 ĐẾN -26 ELO. Tag: "FEEDER"

BẮT BUỘC TRẢ VỀ ĐÚNG 1 ĐỐI TƯỢNG JSON (không có markdown backticks ```json):
{{
  "winner": "{winner}",
  "match_mvp": "Tên tuyển thủ MVP",
  "match_svp": "Tên tuyển thủ SVP",
  "ai_summary": "Tóm tắt 2-3 câu bình luận về trận đấu, điểm nhấn chiến thuật và sự tỏa sáng của các tuyển thủ.",
  "players_analysis": [
    {{
      "player_id": "id_chinh_xac_trong_danh_sach",
      "nickname": "Tên tuyển thủ",
      "team": 1,
      "champion": "Tên tướng (nếu nhận diện được)",
      "kda": "12/2/8",
      "damage": "24.5k",
      "performance_score": 9.2,
      "performance_tag": "MVP",
      "recommended_delta": 24.0,
      "comment": "Lý do ngắn gọn 1 câu giải thích mức điểm này"
    }}
  ]
}}
Lưu ý: Bắt buộc phải có đủ 10 người chơi trong mảng `players_analysis` (5 người team 1 và 5 người team 2) với `player_id` chính xác theo danh sách trên.
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
            res_data = self._call_gemini_api(payload, custom_key=key, timeout=30)
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
            analysis_by_id = {str(item.get('player_id', '')).lower(): item for item in players_analysis}
            analysis_by_nick = {str(item.get('nickname', '')).lower(): item for item in players_analysis}

            normalized_list = []
            for p in team1_players:
                pid = str(p['id']).lower()
                nick = str(p.get('nickname', '')).lower()
                found = analysis_by_id.get(pid) or analysis_by_nick.get(nick)
                default_delta = 16.0 if winner == 'team1' else -16.0
                if found:
                    delta = float(found.get('recommended_delta', default_delta))
                    normalized_list.append({
                        "player_id": p['id'],
                        "nickname": p.get('nickname', p['id']),
                        "team": 1,
                        "champion": found.get('champion', '-'),
                        "kda": found.get('kda', '-'),
                        "damage": found.get('damage', '-'),
                        "performance_score": float(found.get('performance_score', 6.0)),
                        "performance_tag": found.get('performance_tag', 'SOLID'),
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
                pid = str(p['id']).lower()
                nick = str(p.get('nickname', '')).lower()
                found = analysis_by_id.get(pid) or analysis_by_nick.get(nick)
                default_delta = 16.0 if winner == 'team2' else -16.0
                if found:
                    delta = float(found.get('recommended_delta', default_delta))
                    normalized_list.append({
                        "player_id": p['id'],
                        "nickname": p.get('nickname', p['id']),
                        "team": 2,
                        "champion": found.get('champion', '-'),
                        "kda": found.get('kda', '-'),
                        "damage": found.get('damage', '-'),
                        "performance_score": float(found.get('performance_score', 6.0)),
                        "performance_tag": found.get('performance_tag', 'SOLID'),
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

            return {
                "success": True,
                "winner": winner,
                "match_mvp": parsed.get('match_mvp', ''),
                "match_svp": parsed.get('match_svp', ''),
                "ai_summary": parsed.get('ai_summary', 'Đã phân tích thông số trận đấu thành công.'),
                "players_analysis": normalized_list
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Lỗi phân tích bảng điểm từ Gemini: {str(e)}"
            }


gemini_service = GeminiService()

