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


def get_best_font():
    candidates = [
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
        '/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    result = subprocess.run(
        ['fc-match', '-f', '%{file}', 'serif:bold'],
        capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return candidates[-1]


def build_video_filter(duration, fps, font):
    frames      = int(duration * fps)
    fade_out_st = max(duration - 3, duration * 0.85)

    # ── 1. SLOW CINEMATIC ZOOM (linear keyframe, 1.00 → 1.08) ─────────────────
    z_inc = 0.08 / max(frames, 1)
    zoom_filter = (
        f"scale=3840:2160:flags=lanczos,"
        f"zoompan="
        f"z='min(1.00+{z_inc:.8f}*on,1.08)':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={frames}:"
        f"s=1280x720:"
        f"fps={fps}"
    )

    # ── 2. LIGHT PULSE ANIMATION ───────────────────────────────────────────────
    # Achieved via curves filter with time-varying brightness oscillation.
    # We layer two effects:
    #   a) Overall brightness pulse: gentle sine wave on all channels
    #   b) Warm highlight boost: pushes highlights up/down to simulate lights
    # FFmpeg 'curves' doesn't support time expressions, so we use 'colorbalance'
    # and 'curves' chained with 'lutrgb' using 't' variable for time animation.
    #
    # BEST approach for light flicker: use 'eq' filter with time-based brightness
    # eq=brightness expression + 'hue' saturation pulse = realistic light movement
    #
    # brightness oscillates: 0.0 ± 0.06 at ~0.7Hz (slow atmospheric pulse)
    # contrast stays ~1.05 to pop the lights
    # saturation pulses slightly: 1.0 ± 0.08 (makes warm lights glow warmer)

    light_filter = (
        f"eq="
        f"brightness='0.04*sin(t*2.2+0.3)':"       # slow brightness pulse
        f"contrast='1.05+0.04*sin(t*1.8+1.0)':"    # contrast breathes
        f"saturation='1.08+0.10*sin(t*2.5+0.8)'"   # saturation warms/cools
    )

    # ── 3. WARM GRADE + VIGNETTE + GRAIN ──────────────────────────────────────
    grade_filter = (
        f"curves="
        f"r='0/0 0.5/0.53 1/1':"
        f"g='0/0 0.5/0.48 1/0.95':"
        f"b='0/0 0.5/0.43 1/0.86',"
        f"vignette=PI/4.5,"
        f"noise=alls=3:allf=t"
    )

    # ── 4. FADE IN / OUT ───────────────────────────────────────────────────────
    fade_filter = (
        f"fade=t=in:st=0:d=2,"
        f"fade=t=out:st={fade_out_st:.2f}:d=3"
    )

    # ── 5. VIP WATERMARK — top right ──────────────────────────────────────────
    # thin gold line / SORLUNE (serif glowing gold) / thin gold line
    m = 20
    top_line = (
        f"drawtext="
        f"fontfile={font}:"
        f"text='                        ':"
        f"fontsize=7:fontcolor=0xD4AF37@0:"
        f"box=1:boxcolor=0xD4AF37@0.85:boxborderw=0:"
        f"x=w-tw-{m}:y=16"
    )
    bot_line = (
        f"drawtext="
        f"fontfile={font}:"
        f"text='                        ':"
        f"fontsize=7:fontcolor=0xD4AF37@0:"
        f"box=1:boxcolor=0xD4AF37@0.85:boxborderw=0:"
        f"x=w-tw-{m}:y=52"
    )
    glow_text = (
        f"drawtext="
        f"fontfile={font}:"
        f"text='SORLUNE':"
        f"fontsize=25:fontcolor=0xF5E080@0.22:"
        f"x=w-tw-{m}:y=27:"
        f"shadowcolor=0xD4AF37@0.35:shadowx=0:shadowy=0"
    )
    main_text = (
        f"drawtext="
        f"fontfile={font}:"
        f"text='SORLUNE':"
        f"fontsize=22:fontcolor=0xD4AF37@0.97:"
        f"x=w-tw-{m}:y=28:"
        f"shadowcolor=0x000000@0.95:shadowx=1:shadowy=1"
    )
    watermark_filter = f"{top_line},{bot_line},{glow_text},{main_text}"

    # ── 6. GLOWING EQUALIZER — center bottom, style like reference image ───────
    #
    # Reference image style:
    #   - Many thin tall bars growing from a CENTER horizontal line (up AND down)
    #   - Bright glowing core, fading toward edges
    #   - Hot pink/magenta glow — we use GOLD to match Sorlune brand
    #   - Bars are taller in center, shorter on edges (mountain shape envelope)
    #
    # Implementation:
    #   - 21 bars (odd number = clean center bar)
    #   - Each bar = TWO drawtext '|' chars: one going UP, one going DOWN
    #   - Center horizontal line = drawtext box (thin gold line)
    #   - Outer bars have lower amplitude (envelope shape)
    #   - Each bar has unique frequency + phase for organic movement
    #   - Center bars brighter (higher alpha), edge bars more transparent
    #
    # Bar center Y = h - 80 (80px from bottom = center line position)
    # Bars grow symmetrically up and down from this center line.

    bar_count   = 21          # total bars (must be odd for clean center)
    bar_gap     = 18          # px between bar centers  (21 bars * 18 = 378px total)
    center_y    = "h-80"      # center line Y position
    max_amp     = 55          # max half-height of tallest bar (center)
    min_amp     = 12          # min half-height of edge bars

    # Envelope: bars near center are tallest
    # amplitude[i] = min_amp + (max_amp - min_amp) * gaussian(i, center)
    half  = bar_count // 2    # = 10
    eq_parts = []

    # Center horizontal glow line
    line_total_w = (bar_count - 1) * bar_gap + 4
    line_x       = f"(w/2-{line_total_w//2})"
    center_line  = (
        f"drawtext="
        f"fontfile={font}:"
        f"text='{'─' * 42}':"
        f"fontsize=9:fontcolor=0xD4AF37@0:"
        f"box=1:boxcolor=0xD4AF37@0.55:boxborderw=0:"
        f"x={line_x}:y={center_y}"
    )
    eq_parts.append(center_line)

    # Frequencies and phases — varied for organic look
    import math
    freqs  = [1.5, 2.1, 2.7, 1.9, 3.1, 2.4, 1.7, 2.9, 2.2, 3.5, 2.0,
              3.5, 2.2, 2.9, 1.7, 2.4, 3.1, 1.9, 2.7, 2.1, 1.5]
    phases = [0.0, 0.5, 1.1, 1.7, 0.3, 0.9, 1.5, 0.2, 0.8, 1.4, 0.6,
              1.4, 0.8, 0.2, 1.5, 0.9, 0.3, 1.7, 1.1, 0.5, 0.0]

    for i in range(bar_count):
        # Gaussian envelope: center bars tallest
        dist      = abs(i - half) / half          # 0.0 at center, 1.0 at edge
        amplitude = min_amp + (max_amp - min_amp) * math.exp(-3.5 * dist * dist)
        amplitude = int(amplitude)

        # Alpha: center bars brighter
        alpha_up   = 0.75 - 0.35 * dist           # 0.75 center → 0.40 edge
        alpha_down = 0.45 - 0.20 * dist           # dimmer going down

        freq  = freqs[i]
        phase = phases[i]

        # X position (centered on screen)
        offset = (i - half) * bar_gap
        bar_x  = f"(w/2+({offset})-tw/2)"

        fs_expr = f"{4}+{amplitude}*abs(sin(t*{freq}+{phase}))"

        # UP bar: grows upward from center line
        y_up = f"({center_y}-({4}+{amplitude}*abs(sin(t*{freq}+{phase}))))"

        # DOWN bar: grows downward from center line
        y_down = f"({center_y})"

        fs_down = f"{3}+{max(4, amplitude//2)}*abs(sin(t*{freq}+{phase}+0.2))"
        y_down_pos = f"({center_y})"

        # UP bar
        bar_up = (
            f"drawtext="
            f"fontfile={font}:"
            f"text='|':"
            f"fontsize='{fs_expr}':"
            f"fontcolor=0xD4AF37@{alpha_up:.2f}:"
            f"x={bar_x}:"
            f"y={y_up}"
        )

        # DOWN bar (shorter, dimmer — like the reference image lower half)
        bar_down = (
            f"drawtext="
            f"fontfile={font}:"
            f"text='|':"
            f"fontsize='{fs_down}':"
            f"fontcolor=0xC49A20@{alpha_down:.2f}:"
            f"x={bar_x}:"
            f"y={y_down_pos}"
        )

        eq_parts.append(bar_up)
        eq_parts.append(bar_down)

    eq_filter = ",".join(eq_parts)

    # ── COMBINE ALL ────────────────────────────────────────────────────────────
    full_filter = (
        f"{zoom_filter},"
        f"{light_filter},"
        f"{grade_filter},"
        f"{fade_filter},"
        f"format=yuv420p,"
        f"{watermark_filter},"
        f"{eq_filter}"
    )

    return full_filter


def generate_video_job(job_id, image_path, audio_path, output_path):
    try:
        jobs[job_id]['status'] = 'processing'

        duration = get_audio_duration(audio_path)
        fps      = 25
        font     = get_best_font()

        video_filter = build_video_filter(duration, fps, font)

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
            # ── FALLBACK ───────────────────────────────────────────────────────
            jobs[job_id]['error'] = proc.stderr[-500:]
            frames  = int(duration * fps)
            z_inc   = 0.08 / max(frames, 1)
            fade_st = max(duration - 3, duration * 0.85)

            fallback_filter = (
                f"scale=1920:1080:flags=lanczos,"
                f"zoompan="
                f"z='min(1.00+{z_inc:.8f}*on,1.08)':"
                f"x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s=1280x720:fps={fps},"
                f"eq=brightness='0.04*sin(t*2.2)':saturation='1.08+0.10*sin(t*2.5)',"
                f"fade=t=in:st=0:d=2,"
                f"fade=t=out:st={fade_st:.2f}:d=3,"
                f"drawtext="
                f"fontfile={font}:"
                f"text='SORLUNE':"
                f"fontsize=22:fontcolor=0xD4AF37@0.97:"
                f"x=w-tw-20:y=28:"
                f"shadowcolor=0x000000@0.95:shadowx=1:shadowy=1,"
                f"format=yuv420p"
            )
            ffmpeg_fb = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-i', audio_path,
                '-vf', fallback_filter,
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
            proc_fb = subprocess.run(ffmpeg_fb, capture_output=True, text=True, timeout=3600)
            if proc_fb.returncode == 0 and os.path.exists(output_path):
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['video_url'] = f"/videos/{job_id}/{job_id}.mp4"
            else:
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['error'] = proc_fb.stderr[-500:]

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
    api_key   = data.get('api_key', 'default')

    if not audio_url or not image_url:
        return jsonify({'error': 'Missing audio_url or image_url'}), 400

    job_id     = api_key
    job_folder = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(job_folder, exist_ok=True)

    image_path  = os.path.join(job_folder, 'image.jpg')
    audio_path  = os.path.join(job_folder, 'audio.mp3')
    output_path = os.path.join(job_folder, f'{job_id}.mp4')

    jobs[job_id] = {'status': 'pending', 'video_url': None}

    def run():
        try:
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
    data    = request.get_json()
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

    audio_url        = data.get('url')
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
    except Exception:
        return jsonify({'error': 'Could not read audio duration'}), 500

    segments  = []
    start     = 0
    seg_index = 0

    while start < total_duration:
        seg_filename = f'{session_id}_seg{seg_index:03d}.mp3'
        seg_path     = os.path.join(AUDIO_SEGMENTS_FOLDER, seg_filename)

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

        start     += segment_duration
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
