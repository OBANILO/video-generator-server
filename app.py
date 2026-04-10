from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import uuid
import requests
import threading
import time

app = Flask(__name__)

jobs = {}

UPLOAD_FOLDER = '/tmp/video_jobs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AUDIO_SEGMENTS_FOLDER = '/tmp/audio_segments'
os.makedirs(AUDIO_SEGMENTS_FOLDER, exist_ok=True)


def download_file(url, dest_path):
    """Always download fresh - never use cache."""
    cache_bust = f"?nocache={int(time.time())}"
    full_url = url + cache_bust
    
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    
    r = requests.get(full_url, timeout=120, stream=True, headers=headers)
    if r.status_code != 200:
        r = requests.get(url, timeout=120, stream=True, headers=headers)
    
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest_path


def get_audio_duration(audio_path):
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())


def generate_video_job(job_id, image_path, audio_path, output_path):
    try:
        jobs[job_id]['status'] = 'processing'

        duration = get_audio_duration(audio_path)
        fps = 25
        frames = int(duration * fps)
        fade_out_start = max(duration - 3, duration * 0.95)

        # FULL CINEMATIC EFFECT CHAIN:
        # 1. Scale to working resolution
        # 2. Ken Burns breathing zoom (in/out pulse)
        # 3. Gentle sway (left/right + up/down)
        # 4. Warm cinematic color grade (golden tones)
        # 5. Vignette (dark edges, bright center)
        # 6. Subtle film grain noise
        # 7. Fade in + fade out

        video_filter = (
            # Step 1: Scale large enough for zoom without black borders
            f"scale=1920:1080,"
            # Step 2+3: Ken Burns zoom pulse + gentle sway
            f"zoompan="
            f"z='1.04+0.03*sin(on/200)':"
            f"x='iw/2-(iw/zoom/2)+10*sin(on/150)':"
            f"y='ih/2-(ih/zoom/2)+6*sin(on/180)':"
            f"d={frames}:"
            f"s=1280x720:"
            f"fps={fps},"
            # Step 4: Warm cinematic color grade
            f"curves="
            f"r='0/0 0.3/0.35 0.7/0.75 1/1':"
            f"g='0/0 0.3/0.28 0.7/0.68 1/0.95':"
            f"b='0/0 0.3/0.22 0.7/0.58 1/0.85',"
            # Step 5: Vignette effect
            f"vignette=PI/4,"
            # Step 6: Subtle film grain
            f"noise=alls=4:allf=t,"
            # Step 7: Fade in/out
            f"fade=t=in:st=0:d=2,"
            f"fade=t=out:st={fade_out_start:.2f}:d=3,"
            f"format=yuv420p"
        )

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-i', audio_path,
            '-vf', video_filter,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '20',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-t', str(duration),
            '-shortest',
            output_path
        ]

        proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=3600)

        if proc.returncode == 0 and os.path.exists(output_path):
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['video_url'] = f"/videos/{job_id}/{job_id}.mp4"
        else:
            # Fallback: simple zoom + fade only
            jobs[job_id]['error'] = proc.stderr[-300:]
            simple_filter = (
                f"scale=1920:1080,"
                f"zoompan="
                f"z='1.03+0.02*sin(on/200)':"
                f"x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s=1280x720:fps={fps},"
                f"fade=t=in:st=0:d=2,"
                f"fade=t=out:st={fade_out_start:.2f}:d=3,"
                f"format=yuv420p"
            )
            ffmpeg_simple = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-i', audio_path,
                '-vf', simple_filter,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '20',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-pix_fmt', 'yuv420p',
                '-t', str(duration),
                '-shortest',
                output_path
            ]
            proc2 = subprocess.run(ffmpeg_simple, capture_output=True, text=True, timeout=3600)
            if proc2.returncode == 0 and os.path.exists(output_path):
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['video_url'] = f"/videos/{job_id}/{job_id}.mp4"
            else:
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['error'] = proc2.stderr[-500:]

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)


@app.route('/generate', methods=['POST'])
def generate_video():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data'}), 400

    audio_url = data.get('audio_url')
    image_url = data.get('image_url')
    api_key = data.get('api_key', 'default')

    if not audio_url or not image_url:
        return jsonify({'error': 'Missing audio_url or image_url'}), 400

    job_id = api_key
    job_folder = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(job_folder, exist_ok=True)

    image_path = os.path.join(job_folder, 'image.jpg')
    audio_path = os.path.join(job_folder, 'audio.mp3')
    output_path = os.path.join(job_folder, f'{job_id}.mp4')

    jobs[job_id] = {'status': 'pending', 'video_url': None}

    def run():
        try:
            # Always delete old files first
            for f in [image_path, audio_path, output_path]:
                if os.path.exists(f):
                    os.remove(f)
            download_file(image_url, image_path)
            download_file(audio_url, audio_path)
            generate_video_job(job_id, image_path, audio_path, output_path)
        except Exception as e:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)

    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started', 'job_id': job_id}), 200


@app.route('/status/<api_key>', methods=['GET'])
def check_status(api_key):
    job = jobs.get(api_key)
    if not job:
        return jsonify({'status': 'not_found'}), 200

    response = {'status': job['status']}
    if job['status'] == 'completed':
        base_url = request.host_url.rstrip('/')
        response['video_url'] = base_url + f'/videos/{api_key}/{api_key}.mp4'

    if job.get('error'):
        response['error'] = job['error']

    return jsonify(response), 200


@app.route('/videos/<job_id>/<filename>', methods=['GET'])
def serve_video(job_id, filename):
    folder = os.path.join(UPLOAD_FOLDER, job_id)
    return send_from_directory(folder, filename)


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    data = request.get_json()
    api_key = data.get('api_key') if data else None

    if api_key and api_key in jobs:
        del jobs[api_key]

    if api_key:
        import shutil
        job_folder = os.path.join(UPLOAD_FOLDER, api_key)
        if os.path.exists(job_folder):
            shutil.rmtree(job_folder, ignore_errors=True)

    return jsonify({'status': 'cleared'}), 200


@app.route('/process-audio', methods=['POST'])
def process_audio():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data'}), 400

    audio_url = data.get('url')
    segment_duration = int(data.get('segment_duration', 60))

    if not audio_url:
        return jsonify({'error': 'Missing url'}), 400

    session_id = str(uuid.uuid4())[:8]
    audio_path = os.path.join(AUDIO_SEGMENTS_FOLDER, f'{session_id}_input.mp3')

    try:
        download_file(audio_url, audio_path)
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ], capture_output=True, text=True)

    try:
        total_duration = float(result.stdout.strip())
    except:
        return jsonify({'error': 'Could not read audio duration'}), 500

    segments = []
    start = 0
    seg_index = 0

    while start < total_duration:
        seg_filename = f'{session_id}_seg{seg_index:03d}.mp3'
        seg_path = os.path.join(AUDIO_SEGMENTS_FOLDER, seg_filename)

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-ss', str(start),
            '-t', str(segment_duration),
            '-c:a', 'libmp3lame',
            '-b:a', '192k',
            seg_path
        ]

        proc = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=120)
        if proc.returncode == 0 and os.path.exists(seg_path):
            segments.append(seg_filename)

        start += segment_duration
        seg_index += 1

    os.remove(audio_path)
    return jsonify({'segments': segments}), 200


@app.route('/audio_segments/<filename>', methods=['GET'])
def serve_audio_segment(filename):
    return send_from_directory(AUDIO_SEGMENTS_FOLDER, filename)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Video server running'}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
