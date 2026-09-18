from flask import Blueprint, request, jsonify, current_app
from services.data_manager import data_manager
from services.player_service import player_service

matches_bp = Blueprint('matches', __name__)


@matches_bp.route('/api/update_match_result', methods=['POST'])
def api_update_match_result():
    """Ghi nhận kết quả trận đấu đa chiều (Supabase + Local), tự động cập nhật Elo ẩn và tính lại Phong độ."""
    try:
        data = request.json or {}
        team1 = data.get('team1', [])
        team2 = data.get('team2', [])
        winner = data.get('winner', '')
        team1_power = float(data.get('team1_power', 0.0))
        team2_power = float(data.get('team2_power', 0.0))
        synergies = data.get('synergies', {})
        notes = data.get('notes', '')
        player_deltas = data.get('player_deltas')
        player_performances = data.get('player_performances')
        ai_summary = data.get('ai_summary', '')
        team1_kills = int(data.get('team1_kills', 0))
        team2_kills = int(data.get('team2_kills', 0))

        if not team1 or not team2 or winner not in ['team1', 'team2']:
            return jsonify({'success': False, 'error': 'Dữ liệu trận đấu không hợp lệ'}), 400

        # Nếu chưa truyền power, tự động tính tổng Elo 2 đội
        if team1_power == 0.0 or team2_power == 0.0:
            all_p_map = {p['id'].lower(): p for p in player_service.get_all_players()}
            team1_power = round(sum(all_p_map.get(str(pid).lower(), {}).get('hidden_elo', 1200.0) for pid in team1), 1)
            team2_power = round(sum(all_p_map.get(str(pid).lower(), {}).get('hidden_elo', 1200.0) for pid in team2), 1)

        # Lưu trận đấu mới vào Supabase và local cache
        data_manager.append_match(
            team1=team1,
            team2=team2,
            winner=winner,
            team1_power=team1_power,
            team2_power=team2_power,
            synergies=synergies,
            notes=notes,
            player_deltas=player_deltas,
            player_performances=player_performances,
            ai_summary=ai_summary,
            team1_kills=team1_kills,
            team2_kills=team2_kills
        )

        # Tính toán lại toàn bộ metrics ngay lập tức
        player_service.refresh_metrics()

        return jsonify({
            'success': True,
            'message': 'Đã lưu kết quả trận đấu thành công! Elo ẩn và Phong độ đã được cập nhật tự động.'
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error in api_update_match_result: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@matches_bp.route('/api/matches', methods=['GET'])
def api_get_matches():
    """Lấy danh sách toàn bộ lịch sử các trận đấu đã ghi nhận."""
    try:
        matches = data_manager.get_matches_history()
        return jsonify({'success': True, 'matches': matches, 'count': len(matches)}), 200
    except Exception as e:
        current_app.logger.error(f"Error in api_get_matches: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@matches_bp.route('/api/matches/<int:match_id>', methods=['PUT'])
def api_update_match(match_id: int):
    """Cập nhật thông tin trận đấu (đội hình 2 bên, đội thắng, ghi chú) và tính lại Elo."""
    try:
        data = request.json or {}
        team1 = data.get('team1', [])
        team2 = data.get('team2', [])
        winner = data.get('winner', 'team1')
        notes = data.get('notes', '')
        player_deltas = data.get('player_deltas')
        team1_kills = int(data.get('team1_kills', 0))
        team2_kills = int(data.get('team2_kills', 0))

        if len(team1) != 5 or len(team2) != 5:
            return jsonify({'success': False, 'error': 'Mỗi đội phải có chính xác 5 tuyển thủ'}), 400

        all_players = set(team1 + team2)
        if len(all_players) != 10:
            return jsonify({'success': False, 'error': 'Trùng lặp tuyển thủ giữa 2 đội'}), 400

        if winner not in ['team1', 'team2']:
            return jsonify({'success': False, 'error': 'Đội thắng không hợp lệ (phải là team1 hoặc team2)'}), 400

        ok, msg = data_manager.update_match(
            match_id=match_id,
            team1=team1,
            team2=team2,
            winner=winner,
            notes=notes,
            player_deltas=player_deltas,
            team1_kills=team1_kills,
            team2_kills=team2_kills
        )

        if not ok:
            return jsonify({'success': False, 'error': msg}), 400

        # Tính toán lại toàn bộ metrics ngay lập tức
        player_service.refresh_metrics()

        return jsonify({
            'success': True,
            'message': 'Đã cập nhật trận đấu thành công! Toàn bộ Elo và phong độ đã được tính toán lại.'
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error in api_update_match: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@matches_bp.route('/api/matches/<int:match_id>', methods=['DELETE'])
def api_delete_match(match_id: int):
    """Xóa một trận đấu khỏi hệ thống và tính toán lại Elo."""
    try:
        ok, msg = data_manager.delete_match(match_id)
        if not ok:
            return jsonify({'success': False, 'error': msg}), 400

        # Tính toán lại toàn bộ metrics ngay lập tức
        player_service.refresh_metrics()

        return jsonify({
            'success': True,
            'message': 'Đã xóa trận đấu thành công! Toàn bộ Elo và phong độ đã được tính toán lại.'
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error in api_delete_match: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Legacy endpoint
@matches_bp.route('/update_match_result', methods=['POST'])
def legacy_update_match_result():
    try:
        data = request.json or {}
        team1 = data.get('team1', [])
        team2 = data.get('team2', [])
        winner = data.get('winner', '')
        data_manager.append_match(team1, team2, winner)
        player_service.refresh_metrics()
        return jsonify({'message': 'Match result updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
