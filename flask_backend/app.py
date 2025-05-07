from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from werkzeug.utils import secure_filename
import time

# Pridanie cesty k modelom
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import modelu pre spracovanie obrázka
try:
    from models.chexagent import ChexAgent
    model_available = True
except ImportError:
    print("Upozornenie: Model ChexAgent nie je k dispozícii. Použijeme náhradnú odpoveď.")
    model_available = False

app = Flask(__name__)
CORS(app)  # Povolí Cross-Origin Resource Sharing pre komunikáciu s Next.js

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process', methods=['POST'])
def process_request():
    # Získanie správy (ak existuje)
    message = request.form.get('message', 'No message provided')
    
    # Získanie obrázka (ak existuje)
    if 'image' not in request.files:
        return jsonify({
            'result': f"Spracovaná správa: {message}. Nebol poskytnutý žiadny obrázok."
        })
    
    file = request.files['image']
    
    # Ak nebol vybraný žiadny súbor
    if file.filename == '':
        return jsonify({
            'result': f"Spracovaná správa: {message}. Nebol poskytnutý žiadny obrázok."
        })
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Tu by ste mali pridať kód na spracovanie obrázka
        # Napríklad, zavolanie funkcie vo vašom PPV modeli
        # result = your_model.process_image(filepath, message)
        
        # Pre ukážku vrátime jednoduchú odpoveď
        return jsonify({
            'result': f"Spracovaná správa: '{message}'. Obrázok prijatý a uložený ako '{filename}'."
        })
    
    return jsonify({
        'result': "Nepodporovaný formát súboru. Povolené typy: png, jpg, jpeg, gif"
    })

if __name__ == '__main__':
    app.run(debug=True)