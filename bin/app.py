# app.py
import os
import uuid
from flask import Flask, request, jsonify, send_file, render_template  # Import render_template
from werkzeug.utils import secure_filename
from main import process_video, Mode

app = Flask(__name__, static_folder='static', template_folder='templates') 
UPLOAD_FOLDER = 'inputs' # Folder to save uploaded videos
OUTPUT_FOLDER = 'output' # Folder to save processed videos
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'} # Allowed video file extensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Create input and output folders if they don't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mapping from frontend model values to Mode enum
MODEL_MAP = {
    "playerDetection": Mode.PLAYER_DETECTION,
    "ballDetection": Mode.BALL_DETECTION,
    "pitchDetection": Mode.PITCH_DETECTION,
    "playerTracking": Mode.PLAYER_TRACKING,
    "teamClassification": Mode.TEAM_CLASSIFICATION,
    "radar": Mode.RADAR,
    "voronoi": Mode.VORONOI,
    "ballTrack": Mode.BALL_TRACK,
    "lineProjection": Mode.LINE_PROJECTION,
    "cameraEstimator": Mode.CAMERA_ESTIMATOR,
    "possession": Mode.POSSESSION,
    "speedDistance": Mode.SPEED_DISTANCE,
    "heatMap": Mode.HEAT_MAP,
    "action": Mode.ACTION_RECOGNITION
}

@app.route('/') # Route for the root URL
def index():
    return render_template('index.html') 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/infer', methods=['POST'])
def infer_video():
    if 'videoFile' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    file = request.files['videoFile']
    model_name = request.form.get('model') # Get selected model from form data

    if file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400
    if file and allowed_file(file.filename) and model_name in MODEL_MAP:
        try:
            input_filename = secure_filename(file.filename)
            unique_id = uuid.uuid4() # Generate unique id for filenames
            input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{input_filename}")
            output_filename = f"{unique_id}_output_{input_filename}"
            output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            file.save(input_filepath) # Save uploaded video

            mode = MODEL_MAP[model_name] # Get Mode enum from model name
            device = 'cpu' # or 'cuda' if you have GPU support

            processed_video_path = process_video(input_filepath, output_filepath, device, mode)

            # Construct URL to access the processed video (static folder 'output' is served at /output)
            video_url = f"/output/{output_filename}"

            return jsonify({'success': True, 'video_url': video_url}), 200
        except Exception as e:
            os.remove(input_filepath) # Clean up input file on error
            if os.path.exists(output_filepath):
                os.remove(output_filepath) # Clean up output file on error
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type or model selected'}), 400
    pass

@app.route('/output/<filename>') # Route to serve processed videos from output folder
def serve_output_video(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True) # Run Flask app in debug mode