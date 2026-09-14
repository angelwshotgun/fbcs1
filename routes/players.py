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
