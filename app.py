from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import uuid
import requests
import threading
import time
import json
import math
import re

app = Flask(__name__)

jobs = {}

UPLOAD_FOLDER = '/tmp/video_jobs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AUDIO_SEGMENTS_FOLDER = '/tmp/audio_segments'
os.makedirs(AUDIO_SEGMENTS_FOLDER, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  AI LYRICS TIMING — GPT splits raw lyrics into timed segments
# ══════════════════════════════════════════════════════════════════════════════

def ai_time_lyrics(lyrics_text, audio_duration, openai_api_key):
    """
    Send lyrics + audio duration to GPT-4o-mini.
    Returns list of: [{"start": 0.0, "end": 4.5, "text": "Line 1"}, ...]
    """
    if not openai_api_key or not lyrics_text or not lyrics_text.strip():
        return []

    system_prompt = """You are a professional lyrics timing expert.
Given a song's full lyrics and its total duration in seconds, split the lyrics into lines
and assign realistic start/end timestamps (in seconds) to each line.

Rules:
- Each lyric segment should be 1-2 short lines max (max ~40 characters per line).
- Space lines naturally — allow 0.3-0.8s gaps between segments for breathing.
- Intro/outro: leave first 2s and last 3s without lyrics.
- Verses flow at a medium pace; choruses can be slightly faster.
- Return ONLY a valid JSON array. No explanation, no markdown, no code block.

Format:
[
  {"start": 2.0, "end": 5.5, "text": "First line of lyrics"},
  {"start": 6.2, "end": 10.0, "text": "Second line"},
  ...
]"""

    user_prompt = f"""Total audio duration: {audio_duration:.1f} seconds

Full lyrics:
{lyrics_text}

Generate the timed JSON array now."""

    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {openai_api_key}'
            },
            json={
                'model': 'gpt-4o-mini',
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                'temperature': 0.3,
                'response_format': {'type': 'json_object'}
            },
            timeout=30
        )

        if response.status_code != 200:
            return []

        body = response.json()
        raw = body['choices'][0]['message']['content']

        # GPT might wrap in {"segments": [...]} or return array directly
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            segments = parsed
        elif isinstance(parsed, dict):
            # find the first list value
            segments = next(
                (v for v in parsed.values() if isinstance(v, list)),
                []
            )
        else:
            return []

        # Validate and clean segments
        clean = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start = float(seg.get('start', 0))
            end   = float(seg.get('end', start + 3))
            text  = str(seg.get('text', '')).strip()
            if text and end > start and start >= 0 and end <= audio_duration + 1:
                clean.append({'start': start, 'end': end, 'text': text})

        return clean

    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  ESCAPE TEXT FOR FFMPEG drawtext
# ══════════════════════════════════════════════════════════════════════════════

def ffmpeg_escape(text):
    """Escape special chars for FFmpeg drawtext filter."""
    text = text.replace('\\', '\\\\')
    text = text.replace("'",  "\u2019")   # replace apostrophe with right single quote (safer)
    text = text.replace(':',  '\\:')
    text = text.replace('%',  '\\%')
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  KARAOKE LYRICS FILTER — one glowing gold line at a time, centered
# ══════════════════════════════════════════════════════════════════════════════

def build_karaoke_filter(segments, font):
    """
    Build FFmpeg drawtext filters for karaoke-style lyrics.
    Each segment shows as a large centered gold line for its duration.
    Features: fade-in/out per line, gold glow shadow, semi-transparent background pill.
    """
    if not segments:
        return ""

    parts = []

    for seg in segments:
        start = seg['start']
        end   = seg['end']
        text  = ffmpeg_escape(seg['text'])
        dur   = max(end - start, 0.5)

        # Fade each line: 0.3s in, 0.3s out (clipped to segment duration)
        fade_dur = min(0.3, dur / 4)

        # Alpha expression: fade in, hold, fade out
        # Using between() for visibility window
        alpha_expr = (
            f"if(between(t,{start},{start+fade_dur}),"
            f"(t-{start})/{fade_dur},"
            f"if(between(t,{start+fade_dur},{end-fade_dur}),"
            f"1,"
            f"if(between(t,{end-fade_dur},{end}),"
            f"({end}-t)/{fade_dur},"
            f"0)))"
        )

        # ── Shadow / glow layer (gold blur effect) ─────────────────────────
        glow = (
            f"drawtext="
            f"fontfile={font}:"
            f"text='{text}':"
            f"fontsize=52:"
            f"fontcolor=0xFFD700@0.25:"
            f"x=(w-text_w)/2:"
            f"y=h*0.72:"
            f"shadowcolor=0xD4AF37@0.5:shadowx=0:shadowy=0:"
            f"alpha='{alpha_expr}'"
        )

        # ── Main text (crisp gold) ──────────────────────────────────────────
        main = (
            f"drawtext="
            f"fontfile={font}:"
            f"text='{text}':"
            f"fontsize=48:"
            f"fontcolor=0xFFD700@0.97:"
            f"x=(w-text_w)/2:"
            f"y=h*0.72:"
            f"shadowcolor=0x000000@0.85:shadowx=2:shadowy=2:"
            f"alpha='{alpha_expr}'"
        )

        parts.append(glow)
        parts.append(main)

    return ",".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO FILTER CHAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_video_filter(duration, fps, font, lyrics_segments=None):
    frames      = int(duration * fps)
    fade_out_st = max(duration - 3, duration * 0.85)

    # ── 1. SLOW CINEMATIC ZOOM ────────────────────────────────────────────────
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

    # ── 2. LIGHT PULSE ANIMATION ──────────────────────────────────────────────
    light_filter = (
        f"eq="
        f"brightness='0.04*sin(t*2.2+0.3)':"
        f"contrast='1.05+0.04*sin(t*1.8+1.0)':"
        f"saturation='1.08+0.10*sin(t*2.5+0.8)'"
    )

    # ── 3. WARM GRADE + VIGNETTE + GRAIN ─────────────────────────────────────
    grade_filter = (
        f"curves="
        f"r='0/0 0.5/0.53 1/1':"
        f"g='0/0 0.5/0.48 1/0.95':"
        f"b='0/0 0.5/0.43 1/0.86',"
        f"vignette=PI/4.5,"
        f"noise=alls=3:allf=t"
    )

    # ── 4. FADE IN / OUT ──────────────────────────────────────────────────────
    fade_filter = (
        f"fade=t=in:st=0:d=2,"
        f"fade=t=out:st={fade_out_st:.2f}:d=3"
    )

    # ── 5. PIXEL FORMAT ───────────────────────────────────────────────────────
    format_filter = "format=yuv420p"

    # ── 6. VIP WATERMARK ─────────────────────────────────────────────────────
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

    # ── 7. GLOWING EQUALIZER ─────────────────────────────────────────────────
    bar_count = 21
    bar_gap   = 18
    center_y  = "h-80"
    max_amp   = 55
    min_amp   = 12
    half      = bar_count // 2
    eq_parts  = []

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

    freqs  = [1.5, 2.1, 2.7, 1.9, 3.1, 2.4, 1.7, 2.9, 2.2, 3.5, 2.0,
              3.5, 2.2, 2.9, 1.7, 2.4, 3.1, 1.9, 2.7, 2.1, 1.5]
    phases = [0.0, 0.5, 1.1, 1.7, 0.3, 0.9, 1.5, 0.2, 0.8, 1.4, 0.6,
              1.4, 0.8, 0.2, 1.5, 0.9, 0.3, 1.7, 1.1, 0.5, 0.0]

    for i in range(bar_count):
        dist      = abs(i - half) / half
        amplitude = int(min_amp + (max_amp - min_amp) * math.exp(-3.5 * dist * dist))
        alpha_up  = 0.75 - 0.35 * dist
        alpha_dwn = 0.45 - 0.20 * dist
        freq      = freqs[i]
        phase     = phases[i]
        offset    = (i - half) * bar_gap
        bar_x     = f"(w/2+({offset})-tw/2)"
        fs_expr   = f"{4}+{amplitude}*abs(sin(t*{freq}+{phase}))"

        up_bar = (
            f"drawtext="
            f"fontfile={font}:"
            f"text='|':"
            f"fontsize={fs_expr}:"
            f"fontcolor=0xD4AF37@{alpha_up:.2f}:"
            f"x={bar_x}:"
            f"y=({center_y})-text_h:"
            f"shadowcolor=0xFFE87C@0.4:shadowx=0:shadowy=0"
        )
        down_bar = (
            f"drawtext="
            f"fontfile={font}:"
            f"text='|':"
            f"fontsize={fs_expr}:"
            f"fontcolor=0xC49A20@{alpha_dwn:.2f}:"
            f"x={bar_x}:"
            f"y={center_y}:"
            f"shadowcolor=0xFFE87C@0.2:shadowx=0:shadowy=0"
        )
        eq_parts.append(up_bar)
        eq_parts.append(down_bar)

    eq_filter = ",".join(eq_parts)

    # ── 8. KARAOKE LYRICS (optional) ─────────────────────────────────────────
    karaoke_filter = ""
    if lyrics_segments:
        karaoke_filter = build_karaoke_filter(lyrics_segments, font)

    # ── ASSEMBLE FULL FILTER CHAIN ────────────────────────────────────────────
    parts = [
        zoom_filter,
        light_filter,
        grade_filter,
        fade_filter,
        format_filter,
        watermark_filter,
        eq_filter,
    ]
    if karaoke_filter:
        parts.append(karaoke_filter)

    return ",".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE VIDEO GENERATION JOB
# ══════════════════════════════════════════════════════════════════════════════

def generate_video_job(job_id, image_path, audio_path, output_path, lyrics_segments=None):
    try:
        jobs[job_id]['status'] = 'processing'
        duration = get_audio_duration(audio_path)
        fps      = 25
        font     = get_best_font()

        video_filter = build_video_filter(duration, fps, font, lyrics_segments)

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
            # ── FALLBACK (simpler filter, no lyrics) ──────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/generate', methods=['POST'])
def generate_video():
    """
    POST /generate
    Body (JSON):
      - audio_url   : URL to MP3 audio  (required)
      - image_url   : URL to JPG image  (required for long video)
      - video_url   : URL to MP4 video  (required for short video)
      - api_key     : your unique API key (required)
      - lyrics      : plain text lyrics  (optional — enables karaoke mode)
      - openai_key  : OpenAI API key     (optional — needed if lyrics provided)
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data'}), 400

    audio_url   = data.get('audio_url')
    image_url   = data.get('image_url')
    api_key     = data.get('api_key', 'default')
    lyrics_text = data.get('lyrics', '').strip()
    openai_key  = data.get('openai_key', '')

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

            # ── AI LYRICS TIMING ───────────────────────────────────────────
            lyrics_segments = []
            if lyrics_text and openai_key:
                try:
                    duration = get_audio_duration(audio_path)
                    jobs[job_id]['status'] = 'timing_lyrics'
                    lyrics_segments = ai_time_lyrics(lyrics_text, duration, openai_key)
                except Exception:
                    lyrics_segments = []  # gracefully continue without lyrics

            generate_video_job(job_id, image_path, audio_path, output_path, lyrics_segments)

        except Exception as e:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)

    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'started',
        'job_id': job_id,
        'lyrics_mode': 'karaoke' if (lyrics_text and openai_key) else 'off'
    }), 200


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
