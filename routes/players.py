import os
import time
from flask import Blueprint, request, jsonify, current_app
from services.player_service import player_service

players_bp = Blueprint('players', __name__)


@players_bp.route('/api/players', methods=['GET'])
def api_get_players():
    """Lấy danh sách tất cả người chơi kèm Stats 1-10, Elo ẩn và Phong độ tự động."""
    try:
        players = player_service.get_all_players()
        return jsonify({'success': True, 'players': players}), 200
    except Exception as e:
        current_app.logger.error(f"Error in api_get_players: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@players_bp.route('/api/players', methods=['POST'])
def api_create_player():
    """Thêm người chơi mới vào hệ thống."""
    try:
        data = request.json or {}
        success, message, player_obj = player_service.create_player(data)
        if success:
            return jsonify({'success': True, 'message': message, 'player': player_obj}), 201
        return jsonify({'success': False, 'error': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@players_bp.route('/api/players/<player_id>', methods=['GET'])
def api_get_player_detail(player_id: str):
    """Lấy chi tiết một người chơi."""
    player = player_service.get_player(player_id)
    if player:
        return jsonify({'success': True, 'player': player}), 200
    return jsonify({'success': False, 'error': 'Không tìm thấy người chơi'}), 404


@players_bp.route('/api/players/<player_id>/details', methods=['GET'])
def api_get_player_full_details(player_id: str):
    """Lấy toàn bộ hồ sơ thống kê chi tiết, biểu đồ, lịch sử đấu và phân tích AI của tuyển thủ."""
    try:
        force_ai = request.args.get('refresh_ai', '').lower() in ['1', 'true', 'yes']
        custom_key = request.args.get('api_key', '')
        details = player_service.get_player_details(player_id, force_ai_refresh=force_ai, custom_key=custom_key)
        if details:
            return jsonify({'success': True, **details}), 200
        return jsonify({'success': False, 'error': 'Không tìm thấy tuyển thủ'}), 404
    except Exception as e:
        current_app.logger.error(f"Error in api_get_player_full_details: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@players_bp.route('/api/players/<player_id>/ai_analysis', methods=['POST'])
def api_regenerate_player_ai(player_id: str):
    """Tạo lại hoặc làm mới phân tích AI cá nhân hóa và danh hiệu cho tuyển thủ."""
    try:
        data = request.json or {}
        custom_key = data.get('api_key', '')
        details = player_service.get_player_details(player_id, force_ai_refresh=True, custom_key=custom_key)
        if details:
            return jsonify({'success': True, 'ai_analysis': details.get('ai_analysis', {}), 'player': details.get('player')}), 200
        return jsonify({'success': False, 'error': 'Không tìm thấy tuyển thủ'}), 404
    except Exception as e:
        current_app.logger.error(f"Error in api_regenerate_player_ai: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@players_bp.route('/api/players/<player_id>', methods=['PUT'])
def api_update_player(player_id: str):
    """Cập nhật thông tin người chơi (Phong độ & Elo ẩn tự động được bảo vệ)."""
    try:
        data = request.json or {}
        success, message, player_obj = player_service.update_player(player_id, data)
        if success:
            return jsonify({'success': True, 'message': message, 'player': player_obj}), 200
        return jsonify({'success': False, 'error': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@players_bp.route('/api/players/<player_id>', methods=['DELETE'])
def api_delete_player(player_id: str):
    """Xóa hồ sơ người chơi khỏi hệ thống."""
    try:
        success, message = player_service.delete_player(player_id)
        if success:
            return jsonify({'success': True, 'message': message}), 200
        return jsonify({'success': False, 'error': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@players_bp.route('/api/players/<player_id>/duck_image', methods=['POST'])
def api_upload_duck_image(player_id: str):
    """Upload ảnh PNG hoặc cập nhật link ảnh vịt đua tùy chỉnh cho người chơi."""
    try:
        pid = str(player_id).strip().lower()
        player = player_service.get_player(pid)
        if not player:
            return jsonify({'success': False, 'error': 'Không tìm thấy người chơi'}), 404

        duck_img_url = ''

        # 1. Trường hợp gửi Multipart Form File
        if 'file' in request.files or 'duck_image' in request.files:
            file = request.files.get('file') or request.files.get('duck_image')
            if file and file.filename:
                # Tạo thư mục uploads nếu chưa có
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                upload_dir = os.path.join(base_dir, 'static', 'uploads', 'ducks')
                os.makedirs(upload_dir, exist_ok=True)

                filename = f"{pid}.png"
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                duck_img_url = f"/static/uploads/ducks/{filename}?t={int(time.time())}"

        # 2. Trường hợp gửi JSON payload (URL hoặc Base64 Data URL)
        elif request.is_json:
            data = request.json or {}
            duck_img_url = str(data.get('duck_image', '')).strip()

        # Cập nhật hồ sơ tuyển thủ
        success, msg, updated = player_service.update_player(pid, {'duck_image': duck_img_url})
        if success:
            return jsonify({
                'success': True,
                'message': 'Đã cập nhật ảnh vịt đua thành công!',
                'duck_image': duck_img_url,
                'player': updated
            }), 200
        return jsonify({'success': False, 'error': msg}), 400
    except Exception as e:
        current_app.logger.error(f"Error in api_upload_duck_image: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

