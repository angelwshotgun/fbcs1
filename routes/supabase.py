from flask import Blueprint, jsonify
from services.data_manager import data_manager
from services.player_service import player_service
from services.supabase_service import supabase_service

supabase_bp = Blueprint('supabase', __name__)


@supabase_bp.route('/api/supabase/status', methods=['GET'])
def api_supabase_status():
    """Kiểm tra trạng thái kết nối tới Supabase PostgreSQL."""
    try:
        status = supabase_service.test_connection()
        status['storage_mode'] = data_manager.get_storage_mode()
        return jsonify({'success': True, **status}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'connected': False}), 500


@supabase_bp.route('/api/supabase/sync', methods=['POST'])
def api_supabase_sync():
    """Đồng bộ toàn bộ tuyển thủ cục bộ lên bảng players của Supabase."""
    try:
        local_players = data_manager.read_local_players_data()
        ok, message, count = supabase_service.sync_players_from_local(local_players)
        if ok:
            player_service.refresh_metrics()
            return jsonify({'success': True, 'message': message, 'synced_count': count}), 200
        return jsonify({'success': False, 'error': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
