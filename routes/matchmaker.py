from flask import Blueprint, request, jsonify, current_app
from services.matchmaking import matchmaking_service

matchmaker_bp = Blueprint('matchmaker', __name__)


@matchmaker_bp.route('/api/create_teams', methods=['POST'])
def api_create_teams():
    """Tạo 2 đội hình cân bằng tối ưu từ 10 tuyển thủ được chọn (hỗ trợ RNG cân bằng)."""
    try:
        data = request.json or {}
        players = data.get('players', [])
        allow_rng = data.get('allow_rng', True)
        rng_tolerance = float(data.get('rng_tolerance', 0.6))
        balance_mode = data.get('balance_mode', 'composite')

        result = matchmaking_service.create_balanced_teams(
            players,
            allow_rng=allow_rng,
            rng_tolerance=rng_tolerance,
            balance_mode=balance_mode
        )
        return jsonify({'success': True, **result}), 200
    except Exception as e:
        current_app.logger.error(f"Error in api_create_teams: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400


@matchmaker_bp.route('/api/create_teams_with_captains', methods=['POST'])
def api_create_teams_with_captains():
    """Chia đội với 2 Đội trưởng cố định và 8 tuyển thủ còn lại (hỗ trợ RNG cân bằng)."""
    try:
        data = request.json or {}
        captain1 = data.get('captain1')
        captain2 = data.get('captain2')
        remaining = data.get('remaining_players', [])
        allow_rng = data.get('allow_rng', True)
        rng_tolerance = float(data.get('rng_tolerance', 0.6))
        balance_mode = data.get('balance_mode', 'composite')

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
        current_app.logger.error(f"Error in api_create_teams_with_captains: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400


# Legacy endpoints
@matchmaker_bp.route('/create_teams', methods=['POST'])
def legacy_create_teams():
    try:
        selected_players = (request.json or {}).get('players', [])
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


@matchmaker_bp.route('/create_teams_with_captains', methods=['POST'])
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
