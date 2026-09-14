import os
from flask import Blueprint, send_from_directory, render_template

pages_bp = Blueprint('pages', __name__)


@pages_bp.route('/')
def serve_index():
    template_path = os.path.join(os.getcwd(), 'templates', 'index.html')
    if os.path.exists(template_path):
        return render_template('index.html')
    return send_from_directory(os.getcwd(), 'index.html')


@pages_bp.route('/player')
def serve_player():
    return send_from_directory(os.getcwd(), 'index2.html')


@pages_bp.route('/pair')
def serve_pair():
    return send_from_directory(os.getcwd(), 'index3.html')


@pages_bp.route('/captains')
def serve_captains():
    return send_from_directory(os.getcwd(), 'index4.html')
