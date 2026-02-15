import os
import io
import colorsys
from flask import Flask, request, jsonify, render_template

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False
    print("❌ numpy not found. Run: pip install numpy")

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("❌ Pillow not found. Run: pip install Pillow")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER']      = os.path.join('static', 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return jsonify({}), 200

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_water_color(image: Image.Image) -> dict:
    """
    Analyzes ocean image by extracting dominant color hue/saturation
    and classifying it into a pollution level.

    Returns a dict with:
      - pollution_level  : str  (CLEAN, MILD, MODERATE, HIGH, SEVERE, CRITICAL)
      - confidence       : float (0.0 - 1.0)
      - dominant_color   : str  hex color
      - avg_hue          : int  (0-360 degrees)
      - avg_saturation   : float
      - avg_brightness   : float
      - contamination_type: str
      - description      : str
      - recommendation   : str
    """

    img = image.convert('RGB')
    img = img.resize((200, 200), Image.LANCZOS)
    pixels = np.array(img)  
    mask = (pixels[:, :, 0] > 10) | (pixels[:, :, 1] > 10) | (pixels[:, :, 2] > 10)
    valid_pixels = pixels[mask]

    if len(valid_pixels) == 0:
        valid_pixels = pixels.reshape(-1, 3)

    avg_r = int(np.mean(valid_pixels[:, 0]))
    avg_g = int(np.mean(valid_pixels[:, 1]))
    avg_b = int(np.mean(valid_pixels[:, 2]))

    dominant_hex = f'#{avg_r:02x}{avg_g:02x}{avg_b:02x}'

    h, s, v = colorsys.rgb_to_hsv(avg_r / 255, avg_g / 255, avg_b / 255)
    hue_deg = round(h * 360)
    saturation = round(s, 3)
    brightness = round(v, 3)

    dark_pixels  = np.sum((pixels[:, :, 0] < 50) &
                          (pixels[:, :, 1] < 50) &
                          (pixels[:, :, 2] < 50))
    total_pixels = pixels.shape[0] * pixels.shape[1]
    dark_ratio   = dark_pixels / total_pixels

    red_dominant = (avg_r > avg_g + 40) and (avg_r > avg_b + 40)

    brown_like = (avg_r > 100 and avg_g > 60 and avg_b < 80 and
                  abs(avg_r - avg_g) < 80 and avg_r > avg_b)

    if dark_ratio > 0.35 or (brightness < 0.2 and saturation > 0.1):
        level            = 'CRITICAL'
        contamination    = 'Oil Spill / Chemical Dumping'
        description      = ('Critical pollution detected. Extremely dark water indicates '
                            'oil spill or heavy chemical contamination. Immediate intervention required.')
        recommendation   = 'Alert Coast Guard & Environmental Agencies Immediately'
        confidence       = 0.90

    elif red_dominant or (hue_deg > 345 or hue_deg < 15):
        level            = 'SEVERE'
        contamination    = 'Red Tide / Toxic Algal Bloom'
        description      = ('Severe pollution detected. Red-tinted water indicates harmful '
                            'algal bloom (Red Tide) — toxic to marine life and humans.')
        recommendation   = 'Avoid water contact. Notify environmental protection agencies.'
        confidence       = 0.87

    elif brown_like or (hue_deg > 25 and hue_deg < 50 and saturation > 0.25):
        level            = 'HIGH'
        contamination    = 'Sewage / Industrial Sediment'
        description      = ('High pollution detected. Brown/murky water suggests heavy '
                            'sediment load, sewage discharge, or industrial waste runoff.')
        recommendation   = 'Conduct water sampling. Report to water authority.'
        confidence       = 0.82

    elif hue_deg > 55 and hue_deg < 100 and saturation > 0.3:
        level            = 'MODERATE'
        contamination    = 'Eutrophication (Nutrient Overload)'
        description      = ('Moderate pollution. Yellow-green color indicates eutrophication '
                            'from agricultural runoff or sewage — reduces oxygen in water.')
        recommendation   = 'Monitor regularly. Reduce nutrient discharge upstream.'
        confidence       = 0.80

    elif hue_deg > 100 and hue_deg < 160 and saturation > 0.2:
        level            = 'MILD'
        contamination    = 'Algal Growth / Chlorophyll Increase'
        description      = ('Mild pollution detected. Green tint indicates early algal '
                            'growth caused by excess nutrients. Still manageable.')
        recommendation   = 'Schedule monitoring. Investigate nearby discharge points.'
        confidence       = 0.78

    else:
        level            = 'CLEAN'
        contamination    = 'None Detected'
        description      = ('Water appears healthy. Clear blue color indicates good '
                            'water quality with normal oxygen levels and low contamination.')
        recommendation   = 'Continue regular environmental monitoring.'
        confidence       = 0.88

    return {
        'pollution_level'   : level,
        'confidence'        : confidence,
        'dominant_color'    : dominant_hex,
        'avg_hue'           : hue_deg,
        'avg_saturation'    : saturation,
        'avg_brightness'    : brightness,
        'contamination_type': contamination,
        'description'       : description,
        'recommendation'    : recommendation,
        'avg_rgb'           : [avg_r, avg_g, avg_b],
    }

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    POST /analyze
    Accepts: multipart/form-data with 'image' field
    Returns: JSON with pollution analysis results
    """
    if not NUMPY_OK:
        return jsonify({'error': 'numpy not installed. Run: pip install numpy'}), 500
    if not PIL_OK:
        return jsonify({'error': 'Pillow not installed. Run: pip install Pillow'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Use: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        result = analyze_water_color(image)
        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f'Analysis error: {str(e)}')
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status'  : 'ok',
        'service' : 'OceanEye',
        'numpy'   : NUMPY_OK,
        'pillow'  : PIL_OK,
        'ready'   : NUMPY_OK and PIL_OK,
        'version' : '1.0.0'
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("🌊 OceanEye Backend starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)