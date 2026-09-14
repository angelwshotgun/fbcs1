from flask import Blueprint, request, jsonify, current_app
from services.player_service import player_service
from services.gemini_service import gemini_service

ai_bp = Blueprint('ai', __name__)


@ai_bp.route('/api/analyze_match_scoreboard', methods=['POST'])
def api_analyze_match_scoreboard():
    """Phân tích ảnh chụp màn hình bảng điểm sau trận đấu bằng Gemini AI để tối ưu hóa Elo từng tuyển thủ."""
    try:
        data = request.json or {}
        image_data = data.get('image', '')
        mime_type = data.get('mime_type', 'image/jpeg')
        team1_ids = data.get('team1', [])
        team2_ids = data.get('team2', [])
        winner = data.get('winner', 'team1')
        custom_key = data.get('api_key', '')

        if not image_data:
            return jsonify({'success': False, 'error': 'Vui lòng cung cấp ảnh chụp bảng điểm trận đấu'}), 400

        all_players_map = {p['id'].lower(): p for p in player_service.get_all_players()}
        team1_players = [all_players_map.get(str(pid).lower(), {'id': pid, 'nickname': pid}) for pid in team1_ids]
        team2_players = [all_players_map.get(str(pid).lower(), {'id': pid, 'nickname': pid}) for pid in team2_ids]

        result = gemini_service.analyze_match_scoreboard(
            image_data=image_data,
            mime_type=mime_type,
            team1_players=team1_players,
            team2_players=team2_players,
            winner=winner,
            custom_key=custom_key
        )
        return jsonify(result), 200
    except Exception as e:
        current_app.logger.error(f"Error in api_analyze_match_scoreboard: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_bp.route('/api/ai_analyze', methods=['POST'])
def api_ai_analyze():
    """Phân tích chiến thuật đội hình bằng Gemini AI."""
    try:
        data = request.json or {}
        team1 = data.get('team1', [])
        team2 = data.get('team2', [])
        custom_key = data.get('api_key', '')

        result = gemini_service.analyze_matchup(team1, team2, custom_key)
        return jsonify({'success': True, **result}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_bp.route('/api/ocr_screenshot', methods=['POST'])
def api_ocr_screenshot():
    """Nhận diện danh sách 10 tuyển thủ từ ảnh chụp màn hình phòng đấu bằng Gemini Vision."""
    try:
        data = request.json or {}
        image_data = data.get('image', '')
        mime_type = data.get('mime_type', 'image/jpeg')
        custom_key = data.get('api_key', '')

        if not image_data:
            return jsonify({'success': False, 'error': 'Vui lòng cung cấp dữ liệu ảnh'}), 400

        all_players = player_service.get_all_players()
        result = gemini_service.recognize_players_from_image(image_data, mime_type, all_players, custom_key)
        return jsonify(result), 200
    except Exception as e:
        current_app.logger.error(f"Error in api_ocr_screenshot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
