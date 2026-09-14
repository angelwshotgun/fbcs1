import os
from typing import Dict, Any
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from services.data_manager import data_manager
from services.player_service import player_service
from services.matchmaking import matchmaking_service
from services.gemini_service import gemini_service

load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

# Khởi tạo metrics ban đầu
player_service.refresh_metrics()


# ==========================================
# GIAO DIỆN WEB (FRONTEND ROUTES)
# ==========================================
@app.route('/')
def serve_index():
    return send_from_directory(os.getcwd(), 'index.html')


@app.route('/player')
def serve_player():
    return send_from_directory(os.getcwd(), 'index2.html')


@app.route('/pair')
def serve_pair():
    return send_from_directory(os.getcwd(), 'index3.html')


@app.route('/captains')
def serve_captains():
    return send_from_directory(os.getcwd(), 'index4.html')


# ==========================================
# REST API: QUẢN TRỊ & THỐNG KÊ NGƯỜI CHƠI
# ==========================================
@app.route('/api/players', methods=['GET'])
def api_get_players():
    """Lấy danh sách tất cả người chơi kèm Stats 1-10, Elo ẩn và Phong độ tự động."""
    try:
        players = player_service.get_all_players()
        return jsonify({'success': True, 'players': players}), 200
    except Exception as e:
        app.logger.error(f"Error in api_get_players: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/players', methods=['POST'])
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


@app.route('/api/players/<player_id>', methods=['GET'])
def api_get_player_detail(player_id: str):
    """Lấy chi tiết một người chơi."""
    player = player_service.get_player(player_id)
    if player:
        return jsonify({'success': True, 'player': player}), 200
    return jsonify({'success': False, 'error': 'Không tìm thấy người chơi'}), 404


@app.route('/api/players/<player_id>', methods=['PUT'])
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


@app.route('/api/players/<player_id>', methods=['DELETE'])
def api_delete_player(player_id: str):
    """Xóa hồ sơ người chơi khỏi hệ thống."""
    try:
        success, message = player_service.delete_player(player_id)
        if success:
            return jsonify({'success': True, 'message': message}), 200
        return jsonify({'success': False, 'error': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# REST API: THUẬT TOÁN CHIA ĐỘI THÔNG MINH
# ==========================================
@app.route('/api/create_teams', methods=['POST'])
def api_create_teams():
    """Thuật toán chia 10 người chơi thành 2 đội với độ cân bằng cao và RNG linh hoạt."""
    try:
        data = request.json or {}
        players = data.get('players', [])
        allow_rng = data.get('allow_rng', True)
        rng_tolerance = float(data.get('rng_tolerance', 0.6))
        balance_mode = data.get('balance_mode') or data.get('mode') or 'composite'

        if len(players) != 10:
            return jsonify({'success': False, 'error': 'Vui lòng chọn chính xác 10 người chơi'}), 400

        result = matchmaking_service.create_balanced_teams(
            players,
            allow_rng=allow_rng,
            rng_tolerance=rng_tolerance,
            balance_mode=balance_mode
        )
        return jsonify({'success': True, **result}), 200
    except Exception as e:
        app.logger.error(f"Error in api_create_teams: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/create_teams_with_captains', methods=['POST'])
def api_create_teams_with_captains():
    """Thuật toán chia đội với 2 Đội trưởng và 8 tuyển thủ (hỗ trợ RNG cân bằng)."""
    try:
        data = request.json or {}
        captain1 = data.get('captain1')
        captain2 = data.get('captain2')
        remaining = data.get('remaining_players', [])
        allow_rng = data.get('allow_rng', True)
        rng_tolerance = float(data.get('rng_tolerance', 0.6))
        balance_mode = data.get('balance_mode') or data.get('mode') or 'composite'

        if not captain1 or not captain2:
            return jsonify({'success': False, 'error': 'Vui lòng chọn đủ 2 Đội trưởng'}), 400
        if len(remaining) != 8:
            return jsonify({'success': False, 'error': 'Vui lòng chọn chính xác 8 người chơi còn lại'}), 400

        result = matchmaking_service.create_teams_with_captains(
            captain1,
            captain2,
            remaining,
            allow_rng=allow_rng,
            rng_tolerance=rng_tolerance,
            balance_mode=balance_mode
        )
        return jsonify({'success': True, **result}), 200
    except Exception as e:
        app.logger.error(f"Error in api_create_teams_with_captains: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/update_match_result', methods=['POST'])
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

        if not team1 or not team2 or winner not in ['team1', 'team2']:
            return jsonify({'success': False, 'error': 'Dữ liệu trận đấu không hợp lệ'}), 400

        # Lưu trận đấu mới vào Supabase và local cache
        data_manager.append_match(
            team1=team1,
            team2=team2,
            winner=winner,
            team1_power=team1_power,
            team2_power=team2_power,
            synergies=synergies,
            notes=notes
        )

        # Tính toán lại toàn bộ metrics ngay lập tức
        player_service.refresh_metrics()

        return jsonify({
            'success': True,
            'message': 'Đã lưu kết quả trận đấu đa chiều! Elo ẩn và Phong độ đã được cập nhật tự động.'
        }), 200
    except Exception as e:
        app.logger.error(f"Error in api_update_match_result: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# REST API: SUPABASE QUẢN TRỊ DỮ LIỆU ĐA CHIỀU
# ==========================================
@app.route('/api/supabase/status', methods=['GET'])
def api_supabase_status():
    """Kiểm tra trạng thái kết nối tới Supabase PostgreSQL."""
    try:
        from services.supabase_service import supabase_service
        status = supabase_service.test_connection()
        status['storage_mode'] = data_manager.get_storage_mode()
        return jsonify({'success': True, **status}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'connected': False}), 500


@app.route('/api/supabase/sync', methods=['POST'])
def api_supabase_sync():
    """Đồng bộ toàn bộ 36 tuyển thủ cục bộ lên bảng players của Supabase."""
    try:
        from services.supabase_service import supabase_service
        local_players = data_manager.read_local_players_data()
        ok, message, count = supabase_service.sync_players_from_local(local_players)
        if ok:
            player_service.refresh_metrics()
            return jsonify({'success': True, 'message': message, 'synced_count': count}), 200
        return jsonify({'success': False, 'error': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# REST API: BẢNG XẾP HẠNG & CẶP ĂN Ý & AI
# ==========================================
@app.route('/api/leaderboard', methods=['GET'])
def api_get_leaderboard():
    """Lấy bảng xếp hạng tổng quát tất cả người chơi."""
    try:
        players = player_service.get_all_players()
        return jsonify({'success': True, 'leaderboard': players}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/synergies', methods=['GET'])
def api_get_synergies():
    """Lấy danh sách các cặp bài trùng (Duo) và bộ ba tam tấu (Trio) có tỷ lệ thắng cao nhất."""
    try:
        metrics = player_service.get_metrics()
        pair_synergy = metrics.get('pair_synergy', {})
        trio_synergy = metrics.get('trio_synergy', {})

        pairs_list = list(pair_synergy.values())
        pairs_list.sort(key=lambda x: (x['matches'], x['winrate']), reverse=True)

        trios_list = list(trio_synergy.values())
        trios_list.sort(key=lambda x: (x['matches'], x['winrate']), reverse=True)

        return jsonify({
            'success': True,
            'pairs': pairs_list,
            'trios': trios_list,
            'synergies': pairs_list
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai_analyze', methods=['POST'])
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


@app.route('/api/ocr_screenshot', methods=['POST'])
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
        app.logger.error(f"Error in api_ocr_screenshot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def api_get_status():
    """Lấy trạng thái hệ thống và chế độ lưu trữ."""
    mode = data_manager.get_storage_mode()
    df = data_manager.read_matches_df()
    total_matches = len(df) if not df.empty else 0
    total_players = len(player_service.get_all_players())
    return jsonify({
        'success': True,
        'storage_mode': mode,
        'total_matches': total_matches,
        'total_players': total_players,
        'has_gemini': bool(os.getenv('GEMINI_API_KEY'))
    }), 200


# ==========================================
# BACKWARD COMPATIBILITY (TƯƠNG THÍCH CODE CŨ)
# ==========================================
@app.route('/players', methods=['GET'])
def legacy_get_players():
    try:
        df = data_manager.read_matches_df()
        player_names = [c for c in df.columns if c != 'Result']
        return jsonify({'players': sorted(player_names)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/create_teams', methods=['POST'])
def legacy_create_teams():
    try:
        selected_players = request.json.get('players', [])
        result = matchmaking_service.create_balanced_teams(selected_players)
        return jsonify({
            'team1': result['team1_names'],
            'team2': result['team2_names'],
            'team1_score': result['team1_power'],
            'team2_score': result['team2_power'],
            'team1_win_prob': result['team1_win_prob'],
            'team2_win_prob': result['team2_win_prob']
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/create_teams_with_captains', methods=['POST'])
def legacy_create_teams_with_captains():
    try:
        data = request.json or {}
        captain1 = data.get('captain1')
        captain2 = data.get('captain2')
        remaining = data.get('remaining_players', [])
        result = matchmaking_service.create_teams_with_captains(captain1, captain2, remaining)
        return jsonify({
            'team1': result['team1_names'],
            'team2': result['team2_names'],
            'team1_score': result['team1_power'],
            'team2_score': result['team2_power']
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/update_match_result', methods=['POST'])
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


@app.route('/sorted_coefficients', methods=['GET'])
def legacy_get_coefficients():
    players = player_service.get_all_players()
    res = [{'Player': p['id'], 'Coefficient': p['hidden_elo']} for p in players]
    return jsonify(res), 200


@app.route('/pair_coefficients', methods=['GET'])
def legacy_get_pair_coefficients():
    metrics = player_service.get_metrics()
    synergy_map = metrics.get('pair_synergy', {})
    res = []
    for s in synergy_map.values():
        res.append({
            'player1': s['p1'],
            'player2': s['p2'],
            'coefficient': s['synergy_score']
        })
    res.sort(key=lambda x: x['coefficient'], reverse=True)
    return jsonify(res), 200


@app.route('/retrain', methods=['POST'])
def legacy_retrain():
    player_service.refresh_metrics()
    return jsonify({'message': 'Model retrained successfully'}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
