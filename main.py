import os
from flask import Flask
from dotenv import load_dotenv

from services.player_service import player_service
from routes import register_blueprints

load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

# Khởi tạo metrics ban đầu cho hệ thống
player_service.refresh_metrics()

# Đăng ký toàn bộ các route Blueprints theo từng module
register_blueprints(app)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
