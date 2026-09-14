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


gemini_service = GeminiService()
