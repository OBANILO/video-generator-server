from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import uuid
import requests
import threading
import time
import math
import re
from PIL import Image

app = Flask(__name__)

jobs = {}

UPLOAD_FOLDER = '/tmp/video_jobs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AUDIO_SEGMENTS_FOLDER = '/tmp/audio_segments'
os.makedirs(AUDIO_SEGMENTS_FOLDER, exist_ok=True)

ICON_DIR = '/tmp/social_icons'
os.makedirs(ICON_DIR, exist_ok=True)

YOUTUBE_ICON_SRC = os.path.join(os.path.dirname(__file__), 'youtube.png')
TIKTOK_ICON_SRC  = os.path.join(os.path.dirname(__file__), 'tiktok.png')

YOUTUBE_ICON = os.path.join(ICON_DIR, 'youtube.png')
TIKTOK_ICON  = os.path.join(ICON_DIR, 'tiktok.png')


def prepare_icons():
    pairs = [(YOUTUBE_ICON_SRC, YOUTUBE_ICON), (TIKTOK_ICON_SRC, TIKTOK_ICON)]
    for src, dst in pairs:
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except Exception:
                pass

        if not os.path.exists(src):
            print(f"[Icons] Source not found: {src}")
            continue

        try:
            img = Image.open(src).convert('RGBA')
            img = img.resize((42, 42), Image.LANCZOS)
            img.save(dst)
            print(f"[Icons] Prepared: {dst}")
        except Exception as e:
            print(f"[Icons] Failed: {e}")

prepare_icons()

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

LYRICS_Y    = 0.84
EQ_CENTER_Y = 0.93
DARK_START  = 0.78


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def download_file(url, dest_path):
    cache_bust = f"?nocache={int(time.time())}"
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    r = requests.get(url + cache_bust, timeout=120, stream=True, headers=headers)
    if r.status_code != 200:
        r = requests.get(url, timeout=120, stream=True, headers=headers)
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest_path


def get_audio_duration(audio_path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())


def get_best_font():
    candidates = [
        os.path.join(os.path.dirname(__file__), 'Anton-Regular.ttf'),
        os.path.join(os.path.dirname(__file__), 'BebasNeue-Regular.ttf'),
        os.path.join(os.path.dirname(__file__), 'Montserrat-ExtraBold.ttf'),
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[Font] Using: {path}")
            return path

    result = subprocess.run(['fc-match', '-f', '%{file}', 'sans:bold'], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    return '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL LYRICS CLEANING (ONLY USED AS LAST FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════

_SECTION_WORDS = (
    r'verse|chorus|bridge|hook|outro|intro|pre[\-\s]?chorus|post[\-\s]?chorus|'
    r'refrain|interlude|instrumental|spoken|rap|breakdown|solo|'
    r'ad[\-\s]?lib|vamp|coda|tag|skit|fade'
)
SECTION_LABEL_PATTERNS = [
    r'^\[.*\]$', r'^\(.*\)$',
    rf'^({_SECTION_WORDS})\s*[\d:.\-]*\s*$',
    rf'^({_SECTION_WORDS})\s*[\(\[].*[\)\]][\s:]*$',
    rf'^({_SECTION_WORDS})\s*\d*\s*:$',
    r'^[\d\s\.\)\(\:\-]+$',
]
SECTION_REGEX = [re.compile(p, re.IGNORECASE) for p in SECTION_LABEL_PATTERNS]


def is_section_label(line):
    stripped = line.strip()
    for pattern in SECTION_REGEX:
        if pattern.match(stripped) or pattern.match(stripped.rstrip(':').strip()):
            return True
    return False


def split_lyrics_lines(lyrics_text):
    if not lyrics_text:
        return []
    lines = []
    for line in lyrics_text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        clean = line.strip()
        if not clean or is_section_label(clean):
            continue
        lines.append(clean)
    return lines


def normalize_word(w):
    return re.sub(r"[^\w']", "", (w or "").lower()).strip()


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER TRANSCRIPTION — STRONG SYNC FROM REAL AUDIO
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio_words_with_whisper(audio_path, openai_api_key):
    if not openai_api_key or not os.path.exists(audio_path):
        return []

    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {openai_api_key}"},
                files={"file": audio_file},
                data={
                    "model": "whisper-1",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
                timeout=300
            )

        if response.status_code != 200:
            print(f"[Whisper] API error {response.status_code}: {response.text[:500]}")
            return []

        data = response.json()
        raw_words = data.get("words", [])
        cleaned_words = []

        for w in raw_words:
            word_text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")

            if not word_text or start is None or end is None:
                continue

            start = float(start)
            end = float(end)

            if end <= start:
                continue

            cleaned_words.append({
                "word": word_text,
                "norm": normalize_word(word_text),
                "start": start,
                "end": end,
            })

        if cleaned_words:
            print(f"[Whisper] {len(cleaned_words)} word timestamps")
            return cleaned_words

        segments = data.get("segments", [])
        seg_words = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            start = seg.get("start")
            end = seg.get("end")
            if not text or start is None or end is None:
                continue

            seg_words.append({
                "word": text,
                "norm": normalize_word(text),
                "start": float(start),
                "end": float(end),
            })

        if seg_words:
            print(f"[Whisper] word timestamps missing, using {len(seg_words)} segment blocks")

        return seg_words

    except Exception as e:
        print(f"[Whisper] Error: {e}")
        return []


def build_lines_from_transcribed_words(words, max_gap=0.45, max_words=5, max_duration=2.8):
    if not words:
        return []

    lines = []
    current = [words[0]]

    def flush_line(line_words):
        if not line_words:
            return None
        text = " ".join(w["word"] for w in line_words).strip()
        if not text:
            return None
        return {
            "start": round(line_words[0]["start"], 2),
            "end": round(line_words[-1]["end"], 2),
            "text": text
        }

    for w in words[1:]:
        prev = current[-1]
        gap = w["start"] - prev["end"]
        line_duration = w["end"] - current[0]["start"]

        should_break = (
            gap > max_gap or
            len(current) >= max_words or
            line_duration > max_duration
        )

        if should_break:
            item = flush_line(current)
            if item:
                lines.append(item)
            current = [w]
        else:
            current.append(w)

    item = flush_line(current)
    if item:
        lines.append(item)

    cleaned = []
    for seg in lines:
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg["text"].strip()

        if not text:
            continue

        min_dur = max(0.60, min(1.40, len(text.split()) * 0.22))
        if end - start < min_dur:
            end = start + min_dur

        if cleaned and start < cleaned[-1]["end"]:
            start = round(cleaned[-1]["end"] + 0.03, 2)
            end = max(end, start + min_dur)

        cleaned.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text
        })

    print(f"[Lyrics] Built {len(cleaned)} subtitle lines from transcribed words")
    return cleaned


def transcribe_lyrics_with_whisper(audio_path, openai_api_key, lyrics_text=""):
    words = transcribe_audio_words_with_whisper(audio_path, openai_api_key)
    return build_lines_from_transcribed_words(words)


# ══════════════════════════════════════════════════════════════════════════════
# FFMPEG TEXT ESCAPE
# ══════════════════════════════════════════════════════════════════════════════

def ffmpeg_escape(text):
    text = text.replace('\\', '\\\\')
    text = text.replace("'", "\u2019")
    text = text.replace(':', '\\:')
    text = text.replace('%', '\\%')
    text = text.replace('[', '\\[')
    text = text.replace(']', '\\]')
    text = text.replace(',', '\\,')
    return text


# ══════════════════════════════════════════════════════════════════════════════
# KARAOKE LYRICS
# ══════════════════════════════════════════════════════════════════════════════

def wrap_lyric_line(text, max_chars=44):
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    best_split = len(words) // 2
    best_diff = float('inf')
    for i in range(1, len(words)):
        p1, p2 = " ".join(words[:i]), " ".join(words[i:])
        diff = abs(len(p1) - len(p2))
        if diff < best_diff and len(p1) <= max_chars and len(p2) <= max_chars:
            best_diff, best_split = diff, i
    return [" ".join(words[:best_split]), " ".join(words[best_split:])]


def build_karaoke_filter(segments, font):
    if not segments:
        return ""

    parts = []
    FONT_SIZE = 44
    LINE_HEIGHT = 54
    MAX_CHARS = 44

    for seg in segments:
        start, end, raw_text = seg["start"], seg["end"], seg["text"]
        dur = max(end - start, 0.5)
        fade_dur = min(0.18, dur / 5)

        alpha_expr = (
            f"if(between(t,{start},{start+fade_dur}),"
            f"(t-{start})/{fade_dur},"
            f"if(between(t,{start+fade_dur},{end-fade_dur}),"
            f"1,"
            f"if(between(t,{end-fade_dur},{end}),"
            f"({end}-t)/{fade_dur},"
            f"0)))"
        )

        lines = wrap_lyric_line(raw_text, max_chars=MAX_CHARS)

        if len(lines) == 1:
            parts.append(
                f"drawtext=fontfile={font}:text='{ffmpeg_escape(lines[0])}':"
                f"fontsize={FONT_SIZE}:fontcolor=white@1.0:"
                f"borderw=4:bordercolor=black@1.0:"
                f"shadowcolor=black@0.95:shadowx=3:shadowy=3:"
                f"x=(w-text_w)/2:y=h*{LYRICS_Y}:alpha='{alpha_expr}'"
            )
        else:
            base_y = LYRICS_Y - 0.045
            for li, line in enumerate(lines):
                y_pos = f"h*{base_y}+{li * LINE_HEIGHT}"
                parts.append(
                    f"drawtext=fontfile={font}:text='{ffmpeg_escape(line)}':"
                    f"fontsize={FONT_SIZE}:fontcolor=white@1.0:"
                    f"borderw=4:bordercolor=black@1.0:"
                    f"shadowcolor=black@0.95:shadowx=3:shadowy=3:"
                    f"x=(w-text_w)/2:y={y_pos}:alpha='{alpha_expr}'"
                )

    return ",".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# EQ SOUNDBAR
# ══════════════════════════════════════════════════════════════════════════════

def build_eq_bar(font):
    parts = []
    bar_count = 30
    bar_gap = 14
    max_amp = 36
    min_amp = 5
    half = bar_count // 2

    center_y = f"h*{EQ_CENTER_Y}"

    freqs = [1.3, 2.1, 2.7, 1.9, 3.1, 2.4, 1.7, 2.9, 2.2, 3.5,
             2.0, 2.8, 2.1, 2.8, 2.0, 3.5, 2.2, 2.9, 1.7, 2.4,
             3.1, 1.9, 2.7, 2.1, 1.3, 1.8, 2.5, 3.0, 1.6, 2.3]
    phases = [0.0, 0.5, 1.1, 1.7, 0.3, 0.9, 1.5, 0.2, 0.8, 1.4,
              0.6, 1.2, 0.0, 1.2, 0.6, 1.4, 0.8, 0.2, 1.5, 0.9,
              0.3, 1.7, 1.1, 0.5, 0.0, 0.7, 1.3, 0.4, 1.0, 1.6]

    for i in range(bar_count):
        dist = abs(i - half) / half
        amplitude = int(min_amp + (max_amp - min_amp) * math.exp(-2.5 * dist * dist))
        alpha_up = 0.90 - 0.25 * dist
        alpha_dwn = 0.40 - 0.15 * dist
        offset = (i - half) * bar_gap
        bar_x = f"(w/2+({offset})-tw/2)"
        fs_expr = f"{4}+{amplitude}*abs(sin(t*{freqs[i]}+{phases[i]}))"

        parts.append(
            f"drawtext=fontfile={font}:text='|':"
            f"fontsize={fs_expr}:fontcolor=0xD4AF37@{alpha_up:.2f}:"
            f"x={bar_x}:y=({center_y})-text_h"
        )
        parts.append(
            f"drawtext=fontfile={font}:text='|':"
            f"fontsize={fs_expr}:fontcolor=0xB8860B@{alpha_dwn:.2f}:"
            f"x={bar_x}:y={center_y}"
        )

    return ",".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# SOCIAL BADGES
# ══════════════════════════════════════════════════════════════════════════════

def build_social_badges_info(yt_channel="sorlune", tt_channel="sorlune08"):
    has_yt = os.path.exists(YOUTUBE_ICON)
    has_tt = os.path.exists(TIKTOK_ICON)

    ICON_W, ICON_H = 36, 36
    PADDING = 18
    GAP = 12

    icon_x = f"W-{ICON_W + PADDING + 120}"
    yt_icon_y = f"H-{ICON_H * 2 + GAP + PADDING}"
    tt_icon_y = f"H-{ICON_H + PADDING}"

    yt_text_x = f"W-{PADDING + 92}"
    tt_text_x = yt_text_x
    yt_text_y = f"H-{ICON_H * 2 + GAP + PADDING + 4}"
    tt_text_y = f"H-{ICON_H + PADDING + 4}"

    overlay_segs = []
    icon_paths = []
    base = 2

    if has_yt:
        icon_paths.append(YOUTUBE_ICON)
        idx = base + len(icon_paths) - 1
        overlay_segs.append(f"[{idx}:v]scale={ICON_W}:{ICON_H},format=rgba[yt_icon]")

    if has_tt:
        icon_paths.append(TIKTOK_ICON)
        idx = base + len(icon_paths) - 1
        overlay_segs.append(f"[{idx}:v]scale={ICON_W}:{ICON_H},format=rgba[tt_icon]")

    return {
        "has_yt": has_yt,
        "has_tt": has_tt,
        "icon_paths": icon_paths,
        "overlay_segs": overlay_segs,
        "icon_x": icon_x,
        "yt_icon_y": yt_icon_y,
        "tt_icon_y": tt_icon_y,
        "yt_text_x": yt_text_x,
        "tt_text_x": tt_text_x,
        "yt_text_y": yt_text_y,
        "tt_text_y": tt_text_y,
        "yt_channel": yt_channel,
        "tt_channel": tt_channel,
    }


def build_social_text_filter(font, info):
    parts = []

    if info["has_yt"]:
        parts.append(
            f"drawtext=fontfile='{font}':text='{ffmpeg_escape(info['yt_channel'])}':"
            f"fontsize=24:fontcolor=white:"
            f"borderw=3:bordercolor=black@0.95:"
            f"x={info['yt_text_x']}:y={info['yt_text_y']}:"
            f"enable='between(t,0,99999)'"
        )

    if info["has_tt"]:
        parts.append(
            f"drawtext=fontfile='{font}':text='{ffmpeg_escape(info['tt_channel'])}':"
            f"fontsize=24:fontcolor=white:"
            f"borderw=3:bordercolor=black@0.95:"
            f"x={info['tt_text_x']}:y={info['tt_text_y']}:"
            f"enable='between(t,0,99999)'"
        )

    return ",".join(parts) if parts else ""


# ══════════════════════════════════════════════════════════════════════════════
# BUILD FFMPEG COMMAND
# ══════════════════════════════════════════════════════════════════════════════

def build_ffmpeg_command(image_path, audio_path, output_path,
                         duration, fps, font,
                         lyrics_segments=None,
                         song_title="", artist_name="SORLUNE",
                         yt_channel="sorlune", tt_channel="sorlune08"):

    frames = int(duration * fps)
    fade_out_st = max(duration - 3, duration * 0.85)
    z_inc = 0.08 / max(frames, 1)

    zoom_filter = (
        f"scale=3840:2160:flags=lanczos,"
        f"zoompan="
        f"z='min(1.00+{z_inc:.8f}*on,1.08)':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s=1280x720:fps={fps}"
    )
    light_filter = (
        f"eq=brightness='0.03*sin(t*2.2+0.3)':"
        f"contrast='1.04+0.03*sin(t*1.8+1.0)':"
        f"saturation='1.06+0.08*sin(t*2.5+0.8)'"
    )
    grade_filter = (
        f"curves=r='0/0 0.5/0.53 1/1':g='0/0 0.5/0.48 1/0.95':b='0/0 0.5/0.43 1/0.86',"
        f"vignette=PI/4.5,noise=alls=3:allf=t"
    )
    fade_filter = f"fade=t=in:st=0:d=2,fade=t=out:st={fade_out_st:.2f}:d=3"
    format_filter = "format=yuv420p"

    dark_overlay = (
        f"drawtext=fontfile={font}:text=' ':"
        f"fontsize=1:fontcolor=black@0:"
        f"box=1:boxcolor=black@0.52:boxborderw=0:"
        f"x=0:y=h*{DARK_START}:fix_bounds=1"
    )

    karaoke_filter = build_karaoke_filter(lyrics_segments, font) if lyrics_segments else ""
    eq_filter = build_eq_bar(font)

    badge_info = build_social_badges_info(yt_channel, tt_channel)
    social_text_filter = build_social_text_filter(font, badge_info)
    has_icons = badge_info["has_yt"] or badge_info["has_tt"]

    vf_parts = [zoom_filter, light_filter, grade_filter, fade_filter, format_filter, dark_overlay]
    if karaoke_filter:
        vf_parts.append(karaoke_filter)
    vf_parts.append(eq_filter)
    if social_text_filter:
        vf_parts.append(social_text_filter)

    vf_chain = ",".join(vf_parts)

    if has_icons:
        fc_parts = list(badge_info["overlay_segs"])
        fc_parts.append(f"[0:v]{vf_chain}[vfout]")
        stream = "vfout"

        if badge_info["has_yt"]:
            fc_parts.append(
                f"[{stream}][yt_icon]overlay=x={badge_info['icon_x']}:y={badge_info['yt_icon_y']}:format=auto[after_yt]"
            )
            stream = "after_yt"

        if badge_info["has_tt"]:
            fc_parts.append(
                f"[{stream}][tt_icon]overlay=x={badge_info['icon_x']}:y={badge_info['tt_icon_y']}:format=auto[final]"
            )
            stream = "final"

        filter_complex = ";".join(fc_parts)
        audio_index = len(badge_info["icon_paths"]) + 1
        extra_inputs = []
        for p in badge_info["icon_paths"]:
            extra_inputs += ['-i', p]

        cmd = (
            ['ffmpeg', '-y', '-loop', '1', '-i', image_path]
            + extra_inputs
            + ['-i', audio_path,
               '-filter_complex', filter_complex,
               '-map', f'[{stream}]', '-map', f'{audio_index}:a',
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
               '-c:a', 'aac', '-b:a', '192k', '-pix_fmt', 'yuv420p',
               '-t', str(duration), '-shortest', output_path]
        )
    else:
        cmd = [
            'ffmpeg', '-y', '-loop', '1', '-i', image_path, '-i', audio_path,
            '-vf', vf_chain,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
            '-c:a', 'aac', '-b:a', '192k', '-pix_fmt', 'yuv420p',
            '-t', str(duration), '-shortest', output_path
        ]

    return cmd


# ══════════════════════════════════════════════════════════════════════════════
# CORE VIDEO GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_video_job(job_id, image_path, audio_path, output_path,
                       lyrics_segments=None, song_title="", artist_name="SORLUNE",
                       yt_channel="sorlune", tt_channel="sorlune08"):
    try:
        jobs[job_id]['status'] = 'processing'
        duration = get_audio_duration(audio_path)
        fps = 25
        font = get_best_font()

        cmd = build_ffmpeg_command(
            image_path, audio_path, output_path,
            duration, fps, font,
            lyrics_segments=lyrics_segments,
            song_title=song_title, artist_name=artist_name,
            yt_channel=yt_channel, tt_channel=tt_channel
        )

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if proc.returncode == 0 and os.path.exists(output_path):
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['video_url'] = f"/videos/{job_id}/{job_id}.mp4"
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = proc.stderr[-3000:]
            print(f"[FFmpeg ERROR]\n{proc.stderr[-3000:]}")

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/generate', methods=['POST'])
def generate_video():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data'}), 400

    audio_url = data.get('audio_url')
    image_url = data.get('image_url')
    api_key = data.get('api_key', 'default')
    lyrics_text = data.get('lyrics', '').strip()
    openai_key = data.get('openai_key', '').strip()
    song_title = data.get('title', '').strip()
    artist_name = data.get('artist', 'SORLUNE').strip()
    yt_channel = data.get('yt_channel', 'sorlune').strip()
    tt_channel = data.get('tt_channel', 'sorlune08').strip()

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
            for f in [image_path, audio_path, output_path]:
                if os.path.exists(f):
                    os.remove(f)

            jobs[job_id]['status'] = 'downloading_assets'
            download_file(image_url, image_path)
            download_file(audio_url, audio_path)

            lyrics_segments = []

            if openai_key:
                try:
                    jobs[job_id]['status'] = 'transcribing_lyrics'
                    lyrics_segments = transcribe_lyrics_with_whisper(audio_path, openai_key, lyrics_text)
                    print(f"[Lyrics] {len(lyrics_segments)} subtitle segments built from real audio")
                except Exception as e:
                    print(f"[Lyrics] Transcription failed: {e}")
                    lyrics_segments = []

            if not lyrics_segments and lyrics_text:
                duration = get_audio_duration(audio_path)
                lines = split_lyrics_lines(lyrics_text)
                if lines:
                    step = max(duration / len(lines), 1.8)
                    current = 0.0
                    for line in lines:
                        lyrics_segments.append({
                            "start": round(current, 2),
                            "end": round(min(current + step, duration), 2),
                            "text": line
                        })
                        current += step

            generate_video_job(
                job_id, image_path, audio_path, output_path,
                lyrics_segments=lyrics_segments,
                song_title=song_title, artist_name=artist_name,
                yt_channel=yt_channel, tt_channel=tt_channel
            )

        except Exception as e:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)

    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'started',
        'job_id': job_id,
        'lyrics_mode': 'whisper' if openai_key else 'fallback'
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
    return send_from_directory(os.path.join(UPLOAD_FOLDER, job_id), filename)


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
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
    ], capture_output=True, text=True)

    try:
        total_duration = float(result.stdout.strip())
    except Exception:
        return jsonify({'error': 'Could not read audio duration'}), 500

    segments = []
    start, seg_idx = 0, 0

    while start < total_duration:
        seg_filename = f'{session_id}_seg{seg_idx:03d}.mp3'
        seg_path = os.path.join(AUDIO_SEGMENTS_FOLDER, seg_filename)

        proc = subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', str(start), '-t', str(segment_duration),
            '-c:a', 'libmp3lame', '-b:a', '192k', seg_path
        ], capture_output=True, timeout=120)

        if proc.returncode == 0 and os.path.exists(seg_path):
            segments.append(seg_filename)

        start += segment_duration
        seg_idx += 1

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
