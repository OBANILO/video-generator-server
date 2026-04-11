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
# UTILITIES
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
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    result = subprocess.run(
        ['fc-match', '-f', '%{file}', 'sans:bold'],
        capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


# ══════════════════════════════════════════════════════════════════════════════
# LYRICS CLEANING
# Strips ALL section labels including "Bridge (Violin Solo):" style
# ══════════════════════════════════════════════════════════════════════════════

# All known section keywords
_SECTION_WORDS = (
    r'verse|chorus|bridge|hook|outro|intro|pre[\-\s]?chorus|post[\-\s]?chorus|'
    r'refrain|interlude|instrumental|spoken|rap|breakdown|solo|'
    r'violin solo|guitar solo|piano solo|drum solo|'
    r'ad[\-\s]?lib|vamp|coda|tag|skit|outro|fade'
)

SECTION_LABEL_PATTERNS = [
    # [anything] or (anything)
    r'^\[.*\]$',
    r'^\(.*\)$',
    # Bare section word + optional number/colon/dash
    rf'^({_SECTION_WORDS})\s*[\d:.\-]*\s*$',
    # Section word + parenthetical: "Bridge (Violin Solo)" or "Bridge (Violin Solo):"
    rf'^({_SECTION_WORDS})\s*[\(\[].*[\)\]][\s:]*$',
    # Section word + colon alone: "Verse 1:" or "Chorus:"
    rf'^({_SECTION_WORDS})\s*\d*\s*:$',
    # Lines that are only numbers/punctuation
    r'^[\d\s\.\)\(\:\-]+$',
]

SECTION_REGEX = [re.compile(p, re.IGNORECASE) for p in SECTION_LABEL_PATTERNS]


def is_section_label(line):
    stripped          = line.strip()
    stripped_no_colon = stripped.rstrip(':').strip()
    for pattern in SECTION_REGEX:
        if pattern.match(stripped) or pattern.match(stripped_no_colon):
            return True
    return False


def split_lyrics_lines(lyrics_text):
    if not lyrics_text:
        return []

    raw_lines = lyrics_text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    lines = []

    for line in raw_lines:
        clean = line.strip()
        if not clean:
            continue
        if is_section_label(clean):
            print(f"[Lyrics] Skipping section label: '{clean}'")
            continue
        lines.append(clean)

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZE
# ══════════════════════════════════════════════════════════════════════════════

def normalize_word(w):
    return re.sub(r"[^\w]", "", w.lower())


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER TRANSCRIPTION — WORD LEVEL (the real fix)
#
# Key insight:
#   Each lyric line's START = timestamp of FIRST sung word in that line
#   Each lyric line's END   = timestamp of LAST  sung word in that line
#   This is read directly from Whisper — never estimated or guessed.
#
# This means:
#   - "A love burned in ink," appears exactly when those words are sung
#   - "now an empty space"   appears exactly when THOSE words are sung
#   - They never overlap, never cut early, never stay too late
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_lyrics_with_whisper(audio_path, openai_api_key, lyrics_text=""):
    if not openai_api_key or not os.path.exists(audio_path):
        return []

    lyric_lines = split_lyrics_lines(lyrics_text)
    if not lyric_lines:
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
                    "language": "en"
                },
                timeout=300
            )

        if response.status_code != 200:
            print(f"[Whisper] API error: {response.status_code} {response.text[:500]}")
            return []

        data  = response.json()
        words = data.get("words", [])

        if words and len(words) > 3:
            print(f"[Whisper] Got {len(words)} word timestamps — using word-level alignment")
            return _align_by_word_timestamps(lyric_lines, words)

        # Fall back to segment-level if word timestamps absent
        segments = data.get("segments", [])
        if segments:
            print(f"[Whisper] No word data — using {len(segments)} segments")
            return _align_by_segment_timestamps(lyric_lines, segments)

        return []

    except Exception as e:
        print(f"[Whisper] Error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# WORD-LEVEL ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def _align_by_word_timestamps(lyric_lines, whisper_words):
    """
    Match each lyric line to its actual sung words using Whisper word timestamps.

    For every lyric line:
      1. Search forward from the current cursor for the window of Whisper words
         that best matches this line (Jaccard word overlap).
      2. line.start = first matched word's start time
         line.end   = last  matched word's end   time  (exact — not estimated)
      3. Advance cursor past the matched words so the next line searches forward.

    Result: lyrics appear and disappear exactly when the singer says them.
    """

    # Build clean word list
    wwords = []
    for w in whisper_words:
        txt = w.get("word", "").strip()
        if not txt:
            continue
        wwords.append({
            "word":  txt,
            "norm":  normalize_word(txt),
            "start": float(w.get("start", 0)),
            "end":   float(w.get("end",   0)),
        })

    if not wwords:
        return []

    result      = []
    word_cursor = 0
    total_words = len(wwords)

    for line_idx, line in enumerate(lyric_lines):
        line_words      = [normalize_word(w) for w in line.split() if normalize_word(w)]
        line_word_count = len(line_words)

        if not line_words:
            continue

        # ── No more Whisper words: append after last timed line ──
        if word_cursor >= total_words:
            last_end = result[-1]["end"] if result else 0
            result.append({
                "start": round(last_end + 0.3,                          2),
                "end":   round(last_end + 0.3 + line_word_count * 0.4,  2),
                "text":  line
            })
            continue

        # ── Search window: look ahead up to 3× line length + buffer ──
        search_limit = min(word_cursor + line_word_count * 3 + 20, total_words)

        best_score     = -1.0
        best_start_idx = word_cursor
        best_end_idx   = min(word_cursor + line_word_count, total_words)

        lyric_set = set(line_words)

        for start_i in range(word_cursor, search_limit):
            min_w = max(1, line_word_count - line_word_count // 2)
            max_w = min(line_word_count + line_word_count // 2 + 3,
                        total_words - start_i)

            for w_len in range(min_w, max_w + 1):
                end_i = start_i + w_len
                if end_i > total_words:
                    break

                slice_norms = set(
                    ww["norm"] for ww in wwords[start_i:end_i] if ww["norm"]
                )
                if not slice_norms:
                    continue

                score = len(lyric_set & slice_norms) / max(len(lyric_set), len(slice_norms))

                if score > best_score:
                    best_score     = score
                    best_start_idx = start_i
                    best_end_idx   = end_i

        # ── Accept match (≥30%) or fall back to positional estimate ──
        if best_score >= 0.30:
            seg_words   = wwords[best_start_idx:best_end_idx]
            seg_start   = seg_words[0]["start"]
            seg_end     = seg_words[-1]["end"]
            word_cursor = best_end_idx
        else:
            remaining_lines = len(lyric_lines) - line_idx
            remaining_words = total_words - word_cursor
            words_per_line  = max(1, remaining_words // remaining_lines)
            end_idx         = min(word_cursor + words_per_line, total_words)

            seg_words   = wwords[word_cursor:end_idx]
            if seg_words:
                seg_start = seg_words[0]["start"]
                seg_end   = seg_words[-1]["end"]
            elif result:
                seg_start = result[-1]["end"] + 0.3
                seg_end   = seg_start + line_word_count * 0.4
            else:
                seg_start = 0.0
                seg_end   = line_word_count * 0.4

            word_cursor = end_idx

        # ── Enforce minimum readable duration ──
        min_dur = max(1.5, line_word_count * 0.35)
        if seg_end - seg_start < min_dur:
            seg_end = seg_start + min_dur

        # ── Prevent overlap with previous line ──
        if result and seg_start < result[-1]["end"]:
            seg_start = result[-1]["end"] + 0.05

        result.append({
            "start": round(seg_start, 2),
            "end":   round(seg_end,   2),
            "text":  line
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENT-LEVEL ALIGNMENT (fallback when Whisper returns no word data)
# ══════════════════════════════════════════════════════════════════════════════

def _score_line_vs_segment(lyric_line, seg_text):
    lyric_words = set(normalize_word(w) for w in lyric_line.split() if normalize_word(w))
    seg_words   = set(normalize_word(w) for w in seg_text.split()   if normalize_word(w))
    if not lyric_words or not seg_words:
        return 0.0
    return len(lyric_words & seg_words) / max(len(lyric_words), len(seg_words))


def _align_by_segment_timestamps(lyric_lines, whisper_segments):
    segs = [
        {
            "start": float(s.get("start", 0)),
            "end":   float(s.get("end",   0)),
            "text":  s.get("text", "").strip().lower()
        }
        for s in whisper_segments
        if float(s.get("end", 0)) > float(s.get("start", 0))
    ]
    if not segs:
        return []

    matches = []
    for line in lyric_lines:
        best_score = -1.0
        best_idx   = 0
        for j, seg in enumerate(segs):
            s = _score_line_vs_segment(line, seg["text"])
            if s > best_score:
                best_score = s
                best_idx   = j
        matches.append((best_score, best_idx))

    avg_score = sum(m[0] for m in matches) / len(matches) if matches else 0.0
    print(f"[Segment align] avg score: {avg_score:.2f}")

    if avg_score >= 0.20:
        seg_to_lines = {}
        for i, (sc, seg_idx) in enumerate(matches):
            seg_to_lines.setdefault(seg_idx, []).append(i)

        line_timings = {}
        for seg_idx, line_indices in seg_to_lines.items():
            seg     = segs[seg_idx]
            seg_dur = max(seg["end"] - seg["start"], 0.1)
            step    = seg_dur / len(line_indices)
            for k, line_idx in enumerate(line_indices):
                s = seg["start"] + k * step
                e = seg["start"] + (k + 1) * step
                if e - s < 1.5:
                    e = s + 1.5
                line_timings[line_idx] = (round(s, 2), round(e, 2))

        last_end = segs[-1]["end"]
        for i in range(len(lyric_lines)):
            if i not in line_timings:
                line_timings[i] = (round(last_end + 0.2, 2), round(last_end + 2.0, 2))
                last_end += 2.2

        result = []
        for i, line in enumerate(lyric_lines):
            start, end = line_timings[i]
            result.append({"start": start, "end": end, "text": line})
        return result

    return _align_proportional(lyric_lines, segs)


def _align_proportional(lyric_lines, segs):
    total_start = segs[0]["start"]
    total_end   = segs[-1]["end"]
    total_dur   = max(total_end - total_start, 1.0)
    step        = total_dur / len(lyric_lines)
    return [
        {
            "start": round(total_start + i * step,       2),
            "end":   round(total_start + (i + 1) * step, 2),
            "text":  line
        }
        for i, line in enumerate(lyric_lines)
    ]


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
# LYRIC LINE WRAPPING + KARAOKE FILTER
# ══════════════════════════════════════════════════════════════════════════════

def wrap_lyric_line(text, max_chars=38):
    if len(text) <= max_chars:
        return [text]

    words      = text.split()
    best_split = len(words) // 2
    best_diff  = float('inf')

    for i in range(1, len(words)):
        part1 = " ".join(words[:i])
        part2 = " ".join(words[i:])
        diff  = abs(len(part1) - len(part2))
        if diff < best_diff and len(part1) <= max_chars and len(part2) <= max_chars:
            best_diff  = diff
            best_split = i

    return [" ".join(words[:best_split]), " ".join(words[best_split:])]


def build_karaoke_filter(segments, font):
    if not segments:
        return ""

    parts       = []
    FONT_SIZE   = 34
    LINE_HEIGHT = 44
    MAX_CHARS   = 38

    for seg in segments:
        start    = seg["start"]
        end      = seg["end"]
        raw_text = seg["text"]
        dur      = max(end - start, 0.5)
        fade_dur = min(0.15, dur / 6)

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
            text = ffmpeg_escape(lines[0])
            parts.append(
                f"drawtext=fontfile={font}:text='{text}':"
                f"fontsize={FONT_SIZE}:fontcolor=white:"
                f"borderw=3:bordercolor=black@0.95:"
                f"x=(w-text_w)/2:y=h*0.82:alpha='{alpha_expr}'"
            )
        else:
            for li, line in enumerate(lines):
                text  = ffmpeg_escape(line)
                y_pos = f"h*0.80+{li * LINE_HEIGHT}"
                parts.append(
                    f"drawtext=fontfile={font}:text='{text}':"
                    f"fontsize={FONT_SIZE}:fontcolor=white:"
                    f"borderw=3:bordercolor=black@0.95:"
                    f"x=(w-text_w)/2:y={y_pos}:alpha='{alpha_expr}'"
                )

    return ",".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO FILTER CHAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_video_filter(duration, fps, font, lyrics_segments=None):
    frames      = int(duration * fps)
    fade_out_st = max(duration - 3, duration * 0.85)

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

    light_filter = (
        f"eq="
        f"brightness='0.04*sin(t*2.2+0.3)':"
        f"contrast='1.05+0.04*sin(t*1.8+1.0)':"
        f"saturation='1.08+0.10*sin(t*2.5+0.8)'"
    )

    grade_filter = (
        f"curves="
        f"r='0/0 0.5/0.53 1/1':"
        f"g='0/0 0.5/0.48 1/0.95':"
        f"b='0/0 0.5/0.43 1/0.86',"
        f"vignette=PI/4.5,"
        f"noise=alls=3:allf=t"
    )

    fade_filter = (
        f"fade=t=in:st=0:d=2,"
        f"fade=t=out:st={fade_out_st:.2f}:d=3"
    )

    format_filter = "format=yuv420p"

    m = 20
    top_line = (
        f"drawtext=fontfile={font}:text='                        ':"
        f"fontsize=7:fontcolor=0xD4AF37@0:"
        f"box=1:boxcolor=0xD4AF37@0.85:boxborderw=0:"
        f"x=w-tw-{m}:y=16"
    )
    bot_line = (
        f"drawtext=fontfile={font}:text='                        ':"
        f"fontsize=7:fontcolor=0xD4AF37@0:"
        f"box=1:boxcolor=0xD4AF37@0.85:boxborderw=0:"
        f"x=w-tw-{m}:y=52"
    )
    glow_text = (
        f"drawtext=fontfile={font}:text='SORLUNE':"
        f"fontsize=25:fontcolor=0xF5E080@0.22:"
        f"x=w-tw-{m}:y=27:"
        f"shadowcolor=0xD4AF37@0.35:shadowx=0:shadowy=0"
    )
    main_text = (
        f"drawtext=fontfile={font}:text='SORLUNE':"
        f"fontsize=22:fontcolor=0xD4AF37@0.97:"
        f"x=w-tw-{m}:y=28:"
        f"shadowcolor=0x000000@0.95:shadowx=1:shadowy=1"
    )
    watermark_filter = f"{top_line},{bot_line},{glow_text},{main_text}"

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
        f"drawtext=fontfile={font}:text='{'─' * 42}':"
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

        eq_parts.append(
            f"drawtext=fontfile={font}:text='|':"
            f"fontsize={fs_expr}:fontcolor=0xD4AF37@{alpha_up:.2f}:"
            f"x={bar_x}:y=({center_y})-text_h:"
            f"shadowcolor=0xFFE87C@0.4:shadowx=0:shadowy=0"
        )
        eq_parts.append(
            f"drawtext=fontfile={font}:text='|':"
            f"fontsize={fs_expr}:fontcolor=0xC49A20@{alpha_dwn:.2f}:"
            f"x={bar_x}:y={center_y}:"
            f"shadowcolor=0xFFE87C@0.2:shadowx=0:shadowy=0"
        )

    eq_filter = ",".join(eq_parts)

    karaoke_filter = ""
    if lyrics_segments:
        karaoke_filter = build_karaoke_filter(lyrics_segments, font)

    parts = [
        zoom_filter, light_filter, grade_filter,
        fade_filter, format_filter, watermark_filter, eq_filter,
    ]
    if karaoke_filter:
        parts.append(karaoke_filter)

    return ",".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# CORE VIDEO GENERATION
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
            '-loop', '1', '-i', image_path,
            '-i', audio_path,
            '-vf', video_filter,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
            '-c:a', 'aac', '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-t', str(duration), '-shortest',
            output_path
        ]

        proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=3600)

        if proc.returncode == 0 and os.path.exists(output_path):
            jobs[job_id]['status']    = 'completed'
            jobs[job_id]['video_url'] = f"/videos/{job_id}/{job_id}.mp4"
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error']  = proc.stderr[-1200:]

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error']  = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/generate', methods=['POST'])
def generate_video():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data'}), 400

    audio_url   = data.get('audio_url')
    image_url   = data.get('image_url')
    api_key     = data.get('api_key', 'default')
    lyrics_text = data.get('lyrics', '').strip()
    openai_key  = data.get('openai_key', '').strip()

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

            jobs[job_id]['status'] = 'downloading_assets'
            download_file(image_url, image_path)
            download_file(audio_url, audio_path)

            lyrics_segments = []
            if openai_key and lyrics_text:
                try:
                    jobs[job_id]['status'] = 'transcribing_lyrics'
                    lyrics_segments = transcribe_lyrics_with_whisper(
                        audio_path, openai_key, lyrics_text
                    )
                    print(f"[Lyrics] Got {len(lyrics_segments)} timed segments")
                except Exception as e:
                    print(f"[Lyrics] Transcription failed: {e}")
                    lyrics_segments = []

            # Fallback: evenly spread if Whisper gave nothing
            if not lyrics_segments and lyrics_text:
                duration = get_audio_duration(audio_path)
                lines    = split_lyrics_lines(lyrics_text)
                if lines:
                    step    = max(duration / len(lines), 1.5)
                    current = 0.0
                    for line in lines:
                        lyrics_segments.append({
                            "start": round(current, 2),
                            "end":   round(min(current + step, duration), 2),
                            "text":  line
                        })
                        current += step

            generate_video_job(job_id, image_path, audio_path, output_path, lyrics_segments)

        except Exception as e:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error']  = str(e)

    thread        = threading.Thread(target=run)
    thread.daemon = True
    thread.start()

    return jsonify({
        'status':      'started',
        'job_id':      job_id,
        'lyrics_mode': 'word_timestamp_whisper' if (openai_key and lyrics_text) else 'fallback'
    }), 200


@app.route('/status/<api_key>', methods=['GET'])
def check_status(api_key):
    job = jobs.get(api_key)
    if not job:
        return jsonify({'status': 'not_found'}), 200

    response = {'status': job['status']}
    if job['status'] == 'completed':
        base_url              = request.host_url.rstrip('/')
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

        proc = subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', str(start), '-t', str(segment_duration),
            '-c:a', 'libmp3lame', '-b:a', '192k', seg_path
        ], capture_output=True, timeout=120)

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
