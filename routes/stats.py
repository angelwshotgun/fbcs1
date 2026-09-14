import os
from flask import Blueprint, jsonify, request
from services.data_manager import data_manager
from services.player_service import player_service

stats_bp = Blueprint('stats', __name__)


@stats_bp.route('/api/leaderboard', methods=['GET'])
def api_get_leaderboard():
    """Lấy bảng xếp hạng tổng quát các tuyển thủ đã tham gia thi đấu (matches > 0)."""
    try:
        players = player_service.get_all_players()
        active_players = [p for p in players if p.get('matches', 0) > 0]
        return jsonify({'success': True, 'leaderboard': active_players}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@stats_bp.route('/api/synergies', methods=['GET'])
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


@stats_bp.route('/api/status', methods=['GET'])
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


# Legacy endpoints
@stats_bp.route('/players', methods=['GET'])
def legacy_get_players():
    try:
        df = data_manager.read_matches_df()
        player_names = [c for c in df.columns if c != 'Result']
        return jsonify({'players': sorted(player_names)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stats_bp.route('/sorted_coefficients', methods=['GET'])
def legacy_get_coefficients():
    players = player_service.get_all_players()
    res = [{'Player': p['id'], 'Coefficient': p['hidden_elo']} for p in players]
    return jsonify(res), 200


@stats_bp.route('/pair_coefficients', methods=['GET'])
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


@stats_bp.route('/retrain', methods=['POST'])
def legacy_retrain():
    player_service.refresh_metrics()
    return jsonify({'message': 'Model retrained successfully'}), 200
