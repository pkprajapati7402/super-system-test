import pkgutil
pkgutil.ImpImporter = pkgutil.zipimporter

from app import create_app
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = create_app()

# Allow CORS for specific origins
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://localhost:5173"]}}, supports_credentials=True)
# CORS(app)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, use_reloader=False)
