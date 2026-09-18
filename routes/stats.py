import os
from flask import Blueprint, jsonify, request, current_app
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


@stats_bp.route('/api/balance_report', methods=['GET'])
def api_balance_report():
    """Báo cáo chất lượng cân đội qua các trận gần đây."""
    try:
        matches = data_manager.get_matches_history()
        metrics = player_service.get_metrics()
        closeness_history = metrics.get('match_closeness_history', [])
        
        # Lấy N trận gần nhất (mặc định 10)
        n = int(request.args.get('n', 10))
        recent = matches[:n]  # matches đã sắp xếp mới nhất trước
        
        # Thống kê balance rating
        rating_counts = {'perfect': 0, 'fair': 0, 'unbalanced': 0, 'stomp': 0, 'unknown': 0}
        total_closeness = 0.0
        matches_with_kills = 0
        stomp_matches = []
        
        for m in recent:
            rating = m.get('balance_rating', 'unknown')
            if rating in rating_counts:
                rating_counts[rating] += 1
            else:
                rating_counts['unknown'] += 1
            
            t1k = m.get('team1_kills', 0)
            t2k = m.get('team2_kills', 0)
            if t1k > 0 or t2k > 0:
                matches_with_kills += 1
                cl = m.get('match_closeness', 0.5)
                total_closeness += cl
            
            if m.get('is_stomp', False):
                stomp_matches.append({
                    'id': m.get('id'),
                    'match_code': m.get('match_code', ''),
                    'team1_kills': t1k,
                    'team2_kills': t2k,
                    'winner': m.get('winner', ''),
                    'created_at': m.get('created_at', '')
                })
        
        avg_closeness = round(total_closeness / max(1, matches_with_kills), 2)
        total_rated = sum(rating_counts.values())
        
        # Xu hướng: so sánh avg closeness 5 trận gần nhất vs 5 trận trước đó
        trend = 'stable'
        if len(closeness_history) >= 6:
            recent_5 = [h['closeness'] for h in closeness_history[-5:]]
            prev_5 = [h['closeness'] for h in closeness_history[-10:-5]]
            if prev_5:
                avg_recent = sum(recent_5) / len(recent_5)
                avg_prev = sum(prev_5) / len(prev_5)
                if avg_recent > avg_prev + 0.05:
                    trend = 'improving'
                elif avg_recent < avg_prev - 0.05:
                    trend = 'declining'
        
        return jsonify({
            'success': True,
            'total_matches_analyzed': len(recent),
            'matches_with_kill_data': matches_with_kills,
            'average_closeness': avg_closeness,
            'trend': trend,
            'rating_distribution': rating_counts,
            'stomp_rate': round(rating_counts['stomp'] / max(1, total_rated) * 100, 1) if total_rated > 0 else 0,
            'perfect_rate': round(rating_counts['perfect'] / max(1, total_rated) * 100, 1) if total_rated > 0 else 0,
            'stomp_matches': stomp_matches,
            'closeness_history': closeness_history[-20:]  # 20 trận gần nhất
        }), 200
    except Exception as e:
        import traceback
        current_app.logger.error(f"Error in api_balance_report: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


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
