import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid, threading, json
from processor import VideoProcessor

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

jobs = {}

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    job_id = str(uuid.uuid4())
    safe_name = f"{job_id}_{file.filename.replace(' ', '_')}"
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)
    jobs[job_id] = {
        'status': 'uploaded',
        'progress': 0,
        'message': 'File uploaded successfully',
        'file': filepath
    }
    return jsonify({'job_id': job_id, 'filename': file.filename})

@app.route('/process/<job_id>', methods=['POST'])
def process(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    if jobs[job_id]['status'] == 'processing':
        return jsonify({'error': 'Already processing'}), 400
    options = request.get_json() or {}
    jobs[job_id]['status'] = 'processing'
    jobs[job_id]['options'] = options

    def run():
        try:
            def cb(progress, msg):
                jobs[job_id]['progress'] = progress
                jobs[job_id]['message'] = msg
            p = VideoProcessor(
                input_path=jobs[job_id]['file'],
                output_folder=OUTPUT_FOLDER,
                job_id=job_id,
                options=options,
                status_callback=cb
            )
            output_path = p.process()
            jobs[job_id].update({'status': 'done', 'output': output_path, 'progress': 100, 'message': 'Done!'})
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print("FULL ERROR:", err)
            jobs[job_id].update({'status': 'error', 'error': str(e), 'message': f'Error: {e}'})

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'message': 'Processing started'})

@app.route('/status/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    info = dict(jobs[job_id])
    info.pop('file', None)
    info.pop('output', None)
    return jsonify(info)

@app.route('/download/<job_id>')
def download(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    if jobs[job_id].get('status') != 'done':
        return jsonify({'error': 'Video not ready yet'}), 400
    return send_file(
        jobs[job_id]['output'],
        as_attachment=True,
        download_name='edited_video.mp4',
        mimetype='video/mp4'
    )

if __name__ == '__main__':
    if __name__ == '__main__':
    print("\n✅  Video Editor running at http://0.0.0.0:7860\n")
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
