from flask import Flask
from routes.pages import pages_bp
from routes.players import players_bp
from routes.matchmaker import matchmaker_bp
from routes.matches import matches_bp
from routes.ai import ai_bp
from routes.supabase import supabase_bp
from routes.stats import stats_bp


def register_blueprints(app: Flask):
    """Đăng ký toàn bộ Blueprints vào ứng dụng Flask."""
    app.register_blueprint(pages_bp)
    app.register_blueprint(players_bp)
    app.register_blueprint(matchmaker_bp)
    app.register_blueprint(matches_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(supabase_bp)
    app.register_blueprint(stats_bp)
