import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


class GeminiService:
    def __init__(self):
        self.api_key = GEMINI_API_KEY

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
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode('utf-8'))
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
        Nhận diện 10 người chơi từ ảnh chụp màn hình phòng đấu (Lobby / Discord / Match)
        bằng Gemini 1.5 Flash Vision.
        """
        key = custom_key or self.api_key
        if not key:
            return {
                'success': False,
                'error': 'Chưa cấu hình Gemini API Key. Vui lòng dán API Key vào ô nhập hoặc mục Cài Đặt.'
            }

        # Tạo danh sách tuyển thủ đã biết để AI đối chiếu chính xác
        players_reference = []
        for p in known_players:
            players_reference.append(f"- ID: {p['id']}, Nickname: {p['nickname']}")
        ref_text = "\n".join(players_reference)

        prompt = f"""
Bạn là AI chuyên gia đọc sảnh chờ game (League of Legends custom lobby, Discord room, game lobby, scoreboard).
Nhiệm vụ: Xem bức ảnh chụp màn hình này và tìm ra ĐÚNG 10 người chơi tham gia trận đấu.
Hãy đối chiếu các tên nhìn thấy với danh sách người chơi hệ thống dưới đây:
=== DANH SÁCH NGƯỜI CHƠI HỆ THỐNG ===
{ref_text}
=====================================

Quy tắc:
1. Nhận diện tối đa 10 người chơi (hoặc tất cả người chơi có mặt nếu ít hơn 10).
2. Ánh xạ các tên đọc được từ ảnh sang đúng `ID` trong danh sách trên (cho phép khác biệt nhỏ về hoa/thường, dấu cách, icon, clan tag).
3. Bắt buộc trả về đúng 1 đối tượng JSON duy nhất (không có backticks markdown ```json):
{{
  "matched_player_ids": ["id1", "id2", "id3", ...],
  "detected_names": ["tên đọc được 1", "tên đọc được 2", ...],
  "count": 10
}}
"""
        # Làm sạch base64 nếu có data uri scheme
        clean_base64 = image_data
        if ',' in image_data:
            clean_base64 = image_data.split(',', 1)[1]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
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
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=25) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                raw_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()

                if raw_text.startswith('```json'):
                    raw_text = raw_text[7:]
                if raw_text.startswith('```'):
                    raw_text = raw_text[3:]
                if raw_text.endswith('```'):
                    raw_text = raw_text[:-3]

                parsed = json.loads(raw_text.strip())
                matched_ids = parsed.get('matched_player_ids', [])

                # Lọc các ID thực sự tồn tại trong danh sách
                valid_ids_set = {p['id'] for p in known_players}
                filtered_ids = [pid for pid in matched_ids if pid in valid_ids_set]

                # Nếu AI trả về ID dạng chữ hoa, thử lowercase
                if len(filtered_ids) < len(matched_ids):
                    for pid in matched_ids:
                        p_lower = pid.lower()
                        if p_lower in valid_ids_set and p_lower not in filtered_ids:
                            filtered_ids.append(p_lower)

                return {
                    'success': True,
                    'matched_player_ids': filtered_ids,
                    'detected_names': parsed.get('detected_names', []),
                    'count': len(filtered_ids)
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Lỗi nhận diện ảnh từ Gemini: {str(e)}"
            }


gemini_service = GeminiService()
