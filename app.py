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
    pairs = [
        (YOUTUBE_ICON_SRC, YOUTUBE_ICON),
        (TIKTOK_ICON_SRC,  TIKTOK_ICON),
    ]
    for src, dst in pairs:
        if os.path.exists(dst):
            continue
        if not os.path.exists(src):
            print(f"[Icons] Source not found: {src} — social badges disabled")
            continue
        try:
            img = Image.open(src).convert('RGBA')
            img = img.resize((52, 52), Image.LANCZOS)
            img.save(dst)
            print(f"[Icons] Prepared {dst}")
        except Exception as e:
            print(f"[Icons] Failed to prepare {dst}: {e}")


prepare_icons()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def download_file(url, dest_path):
    cache_bust = f"?nocache={int(time.time())}"
    full_url   = url + cache_bust
    headers    = {
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
    """Try to find a clean, modern sans-serif bold font."""
    candidates = [
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[Font] Using: {path}")
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
# ══════════════════════════════════════════════════════════════════════════════

_SECTION_WORDS = (
    r'verse|chorus|bridge|hook|outro|intro|pre[\-\s]?chorus|post[\-\s]?chorus|'
    r'refrain|interlude|instrumental|spoken|rap|breakdown|solo|'
    r'violin solo|guitar solo|piano solo|drum solo|'
    r'ad[\-\s]?lib|vamp|coda|tag|skit|outro|fade'
)

SECTION_LABEL_PATTERNS = [
    r'^\[.*\]$',
    r'^\(.*\)$',
    rf'^({_SECTION_WORDS})\s*[\d:.\-]*\s*$',
    rf'^({_SECTION_WORDS})\s*[\(\[].*[\)\]][\s:]*$',
    rf'^({_SECTION_WORDS})\s*\d*\s*:$',
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
    lines = []
    for line in lyrics_text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        clean = line.strip()
        if not clean:
            continue
        if is_section_label(clean):
            print(f"[Lyrics] Skipping section label: '{clean}'")
            continue
        lines.append(clean)
    return lines


def normalize_word(w):
    return re.sub(r"[^\w]", "", w.lower())


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER ALIGNMENT  — improved sync
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
            print(f"[Whisper] API error: {response.status_code} — {response.text[:300]}")
            return []

        data  = response.json()
        words = data.get("words", [])

        if words and len(words) > 3:
            print(f"[Whisper] Got {len(words)} word timestamps — using word alignment")
            return _align_by_word_timestamps(lyric_lines, words)

        segments = data.get("segments", [])
        if segments:
            print(f"[Whisper] Using {len(segments)} segments (fallback)")
            return _align_by_segment_timestamps(lyric_lines, segments)

        return []

    except Exception as e:
        print(f"[Whisper] Error: {e}")
        return []


def _align_by_word_timestamps(lyric_lines, whisper_words):
    """
    Improved alignment: sequential scan with small look-ahead window,
    prevents words from being reused, enforces forward-only movement.
    """
    wwords = []
    for w in whisper_words:
        txt = w.get("word", "").strip()
        if not txt:
            continue
        wwords.append({
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

        # All words exhausted — extrapolate timing from last segment
        if word_cursor >= total_words:
            last_end = result[-1]["end"] if result else 0
            result.append({
                "start": round(last_end + 0.2, 2),
                "end":   round(last_end + 0.2 + max(line_word_count * 0.35, 1.5), 2),
                "text":  line
            })
            continue

        # Look-ahead window: search up to 3× the line word count beyond cursor
        search_limit = min(word_cursor + max(line_word_count * 4, 20), total_words)
        best_score     = -1.0
        best_start_idx = word_cursor
        best_end_idx   = min(word_cursor + line_word_count, total_words)
        lyric_set      = set(line_words)

        for start_i in range(word_cursor, search_limit):
            min_w = max(1, line_word_count - 2)
            max_w = min(line_word_count + 3, total_words - start_i)
            for w_len in range(min_w, max_w + 1):
                end_i = start_i + w_len
                if end_i > total_words:
                    break
                slice_norms = set(ww["norm"] for ww in wwords[start_i:end_i] if ww["norm"])
                if not slice_norms:
                    continue
                score = len(lyric_set & slice_norms) / max(len(lyric_set), len(slice_norms))
                if score > best_score:
                    best_score     = score
                    best_start_idx = start_i
                    best_end_idx   = end_i

        if best_score >= 0.25:
            seg      = wwords[best_start_idx:best_end_idx]
            seg_start = seg[0]["start"]
            seg_end   = seg[-1]["end"]
            word_cursor = best_end_idx
        else:
            # Fallback: evenly distribute remaining words across remaining lines
            rem_lines = len(lyric_lines) - line_idx
            rem_words = total_words - word_cursor
            wpl       = max(1, rem_words // rem_lines)
            end_idx   = min(word_cursor + wpl, total_words)
            seg       = wwords[word_cursor:end_idx]
            if seg:
                seg_start = seg[0]["start"]
                seg_end   = seg[-1]["end"]
            elif result:
                seg_start = result[-1]["end"] + 0.2
                seg_end   = seg_start + max(line_word_count * 0.35, 1.5)
            else:
                seg_start = 0.0
                seg_end   = max(line_word_count * 0.35, 1.5)
            word_cursor = end_idx

        # Ensure minimum display duration per line
        min_dur = max(1.8, line_word_count * 0.30)
        if seg_end - seg_start < min_dur:
            seg_end = seg_start + min_dur

        # Never overlap previous segment
        if result and seg_start < result[-1]["end"]:
            seg_start = result[-1]["end"] + 0.05
            seg_end   = max(seg_end, seg_start + min_dur)

        result.append({"start": round(seg_start, 2), "end": round(seg_end, 2), "text": line})
        print(f"[Align] Line {line_idx:02d} [{seg_start:.2f}→{seg_end:.2f}] score={best_score:.2f}: {line[:50]}")

    return result


def _score_line_vs_segment(lyric_line, seg_text):
    lw = set(normalize_word(w) for w in lyric_line.split() if normalize_word(w))
    sw = set(normalize_word(w) for w in seg_text.split()   if normalize_word(w))
    if not lw or not sw:
        return 0.0
    return len(lw & sw) / max(len(lw), len(sw))


def _align_by_segment_timestamps(lyric_lines, whisper_segments):
    segs = [
        {"start": float(s.get("start", 0)), "end": float(s.get("end", 0)),
         "text": s.get("text", "").strip().lower()}
        for s in whisper_segments if float(s.get("end", 0)) > float(s.get("start", 0))
    ]
    if not segs:
        return []

    matches = []
    for line in lyric_lines:
        best_score, best_idx = -1.0, 0
        for j, seg in enumerate(segs):
            s = _score_line_vs_segment(line, seg["text"])
            if s > best_score:
                best_score, best_idx = s, j
        matches.append((best_score, best_idx))

    avg_score = sum(m[0] for m in matches) / len(matches) if matches else 0.0

    if avg_score >= 0.20:
        seg_to_lines = {}
        for i, (_, seg_idx) in enumerate(matches):
            seg_to_lines.setdefault(seg_idx, []).append(i)

        line_timings = {}
        for seg_idx, idxs in seg_to_lines.items():
            seg  = segs[seg_idx]
            dur  = max(seg["end"] - seg["start"], 0.1)
            step = dur / len(idxs)
            for k, li in enumerate(idxs):
                s = seg["start"] + k * step
                e = seg["start"] + (k + 1) * step
                line_timings[li] = (round(s, 2), round(max(e, s + 1.8), 2))

        last_end = segs[-1]["end"]
        for i in range(len(lyric_lines)):
            if i not in line_timings:
                line_timings[i] = (round(last_end + 0.2, 2), round(last_end + 2.0, 2))
                last_end += 2.2

        return [{"start": line_timings[i][0], "end": line_timings[i][1], "text": line}
                for i, line in enumerate(lyric_lines)]

    total_start = segs[0]["start"]
    total_dur   = max(segs[-1]["end"] - total_start, 1.0)
    step        = total_dur / len(lyric_lines)
    return [{"start": round(total_start + i * step, 2),
             "end":   round(total_start + (i + 1) * step, 2),
             "text":  line}
            for i, line in enumerate(lyric_lines)]


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
# LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
#  0%  ┌──────────────────────────┐
#      │   artist image / scene  │
#  62% ├──────────────────────────┤  ← dark overlay starts
#      │   EQ soundbar            │  63-66%
#      │   ── gold line ──        │  67%
#      │   SONG TITLE             │  68-74%
#      │   artist name (gold)     │  77-80%
#      │   ── gold line ──        │  82%
#      │   LYRICS (subtitle bar)  │  83-91%
# 100% └──────────────────────────┘
# ══════════════════════════════════════════════════════════════════════════════

LYRICS_Y_TOP    = 0.83   # lyrics text baseline
EQ_CENTER_Y     = 0.635  # EQ bar center
TITLE_SEPARATOR = 0.665  # gold line above title
TITLE_Y         = 0.675  # song title
ARTIST_Y        = 0.770  # artist name
LYRICS_SEP_Y    = 0.820  # gold line above lyrics
DARK_START      = 0.620  # dark overlay begins


# ══════════════════════════════════════════════════════════════════════════════
# KARAOKE LYRICS FILTER  — BOTTOM of video (subtitle style)
# ══════════════════════════════════════════════════════════════════════════════

def wrap_lyric_line(text, max_chars=42):
    if len(text) <= max_chars:
        return [text]
    words      = text.split()
    best_split = len(words) // 2
    best_diff  = float('inf')
    for i in range(1, len(words)):
        p1, p2 = " ".join(words[:i]), " ".join(words[i:])
        diff   = abs(len(p1) - len(p2))
        if diff < best_diff and len(p1) <= max_chars and len(p2) <= max_chars:
            best_diff, best_split = diff, i
    return [" ".join(words[:best_split]), " ".join(words[best_split:])]


def build_karaoke_filter(segments, font):
    """
    Lyrics rendered at the very bottom of the video (subtitle position).
    Uses a semi-transparent black pill background for readability.
    Font: larger, clean, bold with thick border.
    """
    if not segments:
        return ""

    parts       = []
    FONT_SIZE   = 38          # larger, easy to read
    LINE_HEIGHT = 46
    MAX_CHARS   = 42

    for seg in segments:
        start, end, raw_text = seg["start"], seg["end"], seg["text"]
        dur      = max(end - start, 0.5)
        fade_dur = min(0.20, dur / 5)

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

        # ── BOTTOM position: 84% for 1 line, 81% for 2 lines ──
        if len(lines) == 1:
            y_pos = f"h*0.855"
            parts.append(
                # Dark pill background
                f"drawtext=fontfile={font}:text='{ffmpeg_escape(lines[0])}':"
                f"fontsize={FONT_SIZE}:fontcolor=white:"
                f"borderw=4:bordercolor=black@1.0:"
                f"shadowcolor=black@0.85:shadowx=2:shadowy=2:"
                f"box=1:boxcolor=black@0.55:boxborderw=14:"
                f"x=(w-text_w)/2:y={y_pos}:alpha='{alpha_expr}'"
            )
        else:
            for li, line in enumerate(lines):
                y_pos = f"h*0.840+{li * LINE_HEIGHT}"
                parts.append(
                    f"drawtext=fontfile={font}:text='{ffmpeg_escape(line)}':"
                    f"fontsize={FONT_SIZE}:fontcolor=white:"
                    f"borderw=4:bordercolor=black@1.0:"
                    f"shadowcolor=black@0.85:shadowx=2:shadowy=2:"
                    f"box=1:boxcolor=black@0.55:boxborderw=12:"
                    f"x=(w-text_w)/2:y={y_pos}:alpha='{alpha_expr}'"
                )

    return ",".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# TITLE BLOCK — song title + artist name in the middle of the dark band
# ══════════════════════════════════════════════════════════════════════════════

def build_title_block(font, song_title, artist_name):
    parts = []

    # ── Top gold separator line ──
    parts.append(
        f"drawtext=fontfile={font}:text='{'─' * 55}':"
        f"fontsize=8:fontcolor=0xD4AF37@0:"
        f"box=1:boxcolor=0xD4AF37@0.70:boxborderw=0:"
        f"x=(w-tw)/2:y=h*{TITLE_SEPARATOR}"
    )

    title_words = song_title.split()
    if len(song_title) > 32 and len(title_words) >= 3:
        mid   = len(title_words) // 2
        line1 = ffmpeg_escape(" ".join(title_words[:mid]).upper())
        line2 = ffmpeg_escape(" ".join(title_words[mid:]).upper())
        parts.append(
            f"drawtext=fontfile={font}:text='{line1}':"
            f"fontsize=46:fontcolor=white@0.97:"
            f"borderw=3:bordercolor=black@0.80:"
            f"shadowcolor=black@0.60:shadowx=2:shadowy=2:"
            f"x=(w-text_w)/2:y=h*{TITLE_Y}"
        )
        parts.append(
            f"drawtext=fontfile={font}:text='{line2}':"
            f"fontsize=46:fontcolor=white@0.97:"
            f"borderw=3:bordercolor=black@0.80:"
            f"shadowcolor=black@0.60:shadowx=2:shadowy=2:"
            f"x=(w-text_w)/2:y=h*{TITLE_Y + 0.055}"
        )
        artist_y_val = ARTIST_Y + 0.025
    else:
        parts.append(
            f"drawtext=fontfile={font}:text='{ffmpeg_escape(song_title.upper())}':"
            f"fontsize=46:fontcolor=white@0.97:"
            f"borderw=3:bordercolor=black@0.80:"
            f"shadowcolor=black@0.60:shadowx=2:shadowy=2:"
            f"x=(w-text_w)/2:y=h*{TITLE_Y}"
        )
        artist_y_val = ARTIST_Y

    parts.append(
        f"drawtext=fontfile={font}:text='{ffmpeg_escape(artist_name.upper())}':"
        f"fontsize=26:fontcolor=0xD4AF37@0.95:"
        f"borderw=2:bordercolor=black@0.70:"
        f"shadowcolor=black@0.50:shadowx=1:shadowy=1:"
        f"x=(w-text_w)/2:y=h*{artist_y_val}"
    )

    # ── Bottom gold separator line (above lyrics) ──
    parts.append(
        f"drawtext=fontfile={font}:text='{'─' * 55}':"
        f"fontsize=8:fontcolor=0xD4AF37@0:"
        f"box=1:boxcolor=0xD4AF37@0.55:boxborderw=0:"
        f"x=(w-tw)/2:y=h*{LYRICS_SEP_Y}"
    )

    return ",".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# EQ BAR — animated gold bars, positioned ABOVE the title block (≈63%)
# ══════════════════════════════════════════════════════════════════════════════

def build_eq_bar(font):
    """
    EQ soundbar sits between the image area and the title block.
    center_y = h*0.635  →  just above the first gold separator line.
    """
    parts     = []
    bar_count = 25
    bar_gap   = 16
    max_amp, min_amp = 45, 8
    half      = bar_count // 2

    center_y = f"h*{EQ_CENTER_Y}"

    freqs  = [1.4, 2.0, 2.6, 1.8, 3.0, 2.3, 1.6, 2.8, 2.1, 3.4, 1.9, 2.7, 2.0,
              2.7, 1.9, 3.4, 2.1, 2.8, 1.6, 2.3, 3.0, 1.8, 2.6, 2.0, 1.4]
    phases = [0.0, 0.5, 1.1, 1.7, 0.3, 0.9, 1.5, 0.2, 0.8, 1.4, 0.6, 1.2, 0.0,
              1.2, 0.6, 1.4, 0.8, 0.2, 1.5, 0.9, 0.3, 1.7, 1.1, 0.5, 0.0]

    for i in range(bar_count):
        dist      = abs(i - half) / half
        amplitude = int(min_amp + (max_amp - min_amp) * math.exp(-2.8 * dist * dist))
        alpha_up  = 0.85 - 0.30 * dist
        alpha_dwn = 0.50 - 0.20 * dist
        offset    = (i - half) * bar_gap
        bar_x     = f"(w/2+({offset})-tw/2)"
        fs_expr   = f"{5}+{amplitude}*abs(sin(t*{freqs[i]}+{phases[i]}))"

        parts.append(
            f"drawtext=fontfile={font}:text='|':"
            f"fontsize={fs_expr}:fontcolor=0xD4AF37@{alpha_up:.2f}:"
            f"x={bar_x}:y=({center_y})-text_h:"
            f"shadowcolor=0xFFE87C@0.35:shadowx=0:shadowy=0"
        )
        parts.append(
            f"drawtext=fontfile={font}:text='|':"
            f"fontsize={fs_expr}:fontcolor=0xC49A20@{alpha_dwn:.2f}:"
            f"x={bar_x}:y={center_y}"
        )

    return ",".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# SOCIAL BADGES
# ══════════════════════════════════════════════════════════════════════════════

def build_social_badges_info(yt_channel="sorlune", tt_channel="sorlune08"):
    has_yt = os.path.exists(YOUTUBE_ICON)
    has_tt = os.path.exists(TIKTOK_ICON)

    ICON_W  = 36
    ICON_H  = 36
    PADDING = 18
    GAP     = 12

    def pulse(period, offset, lo=0.65, hi=1.0):
        amp = (hi - lo) / 2
        mid = (hi + lo) / 2
        return f"{mid}+{amp}*sin(6.2832/{period}*t+{offset})"

    yt_alpha = pulse(3.0, 0.0)
    tt_alpha = pulse(3.0, 3.14)

    icon_x    = f"W-{ICON_W + PADDING + 114}"
    yt_icon_y = f"H-{ICON_H * 2 + GAP + PADDING + 10}"
    tt_icon_y = f"H-{ICON_H + PADDING}"

    yt_text_x = f"W-{PADDING + 108}"
    tt_text_x = yt_text_x
    yt_text_y = f"H-{ICON_H * 2 + GAP + PADDING + 3}"
    tt_text_y = f"H-{ICON_H + PADDING - 2}"

    overlay_segs = []
    icon_paths   = []
    input_index_base = 2

    if has_yt:
        icon_paths.append(YOUTUBE_ICON)
        yt_idx = input_index_base + len(icon_paths) - 1
        overlay_segs.append(
            f"[{yt_idx}:v]scale={ICON_W}:{ICON_H},"
            f"format=rgba,"
            f"colorchannelmixer=aa={yt_alpha}[yt_icon]"
        )

    if has_tt:
        icon_paths.append(TIKTOK_ICON)
        tt_idx = input_index_base + len(icon_paths) - 1
        overlay_segs.append(
            f"[{tt_idx}:v]scale={ICON_W}:{ICON_H},"
            f"format=rgba,"
            f"colorchannelmixer=aa={tt_alpha}[tt_icon]"
        )

    return {
        "has_yt": has_yt, "has_tt": has_tt,
        "icon_paths": icon_paths, "overlay_segs": overlay_segs,
        "icon_x": icon_x,
        "yt_icon_y": yt_icon_y, "tt_icon_y": tt_icon_y,
        "yt_text_x": yt_text_x, "tt_text_x": tt_text_x,
        "yt_text_y": yt_text_y, "tt_text_y": tt_text_y,
        "yt_alpha": yt_alpha, "tt_alpha": tt_alpha,
        "yt_channel": yt_channel, "tt_channel": tt_channel,
    }


def build_social_text_filter(font, info):
    parts = []
    if info["has_yt"]:
        parts.append(
            f"drawtext=fontfile={font}:text='{ffmpeg_escape(info['yt_channel'])}':"
            f"fontsize=18:fontcolor=white@{info['yt_alpha']}:"
            f"borderw=2:bordercolor=black@0.80:"
            f"x={info['yt_text_x']}:y={info['yt_text_y']}"
        )
    if info["has_tt"]:
        parts.append(
            f"drawtext=fontfile={font}:text='{ffmpeg_escape(info['tt_channel'])}':"
            f"fontsize=18:fontcolor=white@{info['tt_alpha']}:"
            f"borderw=2:bordercolor=black@0.80:"
            f"x={info['tt_text_x']}:y={info['tt_text_y']}"
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

    frames      = int(duration * fps)
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
        f"eq=brightness='0.04*sin(t*2.2+0.3)':"
        f"contrast='1.05+0.04*sin(t*1.8+1.0)':"
        f"saturation='1.08+0.10*sin(t*2.5+0.8)'"
    )

    grade_filter = (
        f"curves=r='0/0 0.5/0.53 1/1':g='0/0 0.5/0.48 1/0.95':b='0/0 0.5/0.43 1/0.86',"
        f"vignette=PI/4.5,noise=alls=3:allf=t"
    )

    fade_filter   = f"fade=t=in:st=0:d=2,fade=t=out:st={fade_out_st:.2f}:d=3"
    format_filter = "format=yuv420p"

    # ── Dark gradient overlay covering bottom ~38% of frame ──
    # This covers: EQ bar + title block + lyrics area
    dark_overlay = (
        f"drawtext=fontfile={font}:text=' ':"
        f"fontsize=1:fontcolor=black@0:"
        f"box=1:boxcolor=black@0.68:boxborderw=0:"
        f"x=0:y=h*{DARK_START}:fix_bounds=1"
    )

    eq_filter      = build_eq_bar(font)
    title_filter   = build_title_block(font, song_title, artist_name) if song_title else ""
    karaoke_filter = build_karaoke_filter(lyrics_segments, font) if lyrics_segments else ""

    badge_info         = build_social_badges_info(yt_channel, tt_channel)
    social_text_filter = build_social_text_filter(font, badge_info)
    has_icons          = badge_info["has_yt"] or badge_info["has_tt"]

    # ── Assemble vf chain ──
    vf_parts = [zoom_filter, light_filter, grade_filter,
                fade_filter, format_filter, dark_overlay, eq_filter]
    if title_filter:
        vf_parts.append(title_filter)
    if karaoke_filter:
        vf_parts.append(karaoke_filter)
    if social_text_filter:
        vf_parts.append(social_text_filter)

    vf_chain = ",".join(vf_parts)

    if has_icons:
        fc_parts = list(badge_info["overlay_segs"])
        fc_parts.append(f"[0:v]{vf_chain}[vfout]")

        stream = "vfout"
        if badge_info["has_yt"]:
            fc_parts.append(
                f"[{stream}][yt_icon]overlay="
                f"x={badge_info['icon_x']}:y={badge_info['yt_icon_y']}:"
                f"format=auto[after_yt]"
            )
            stream = "after_yt"

        if badge_info["has_tt"]:
            fc_parts.append(
                f"[{stream}][tt_icon]overlay="
                f"x={badge_info['icon_x']}:y={badge_info['tt_icon_y']}:"
                f"format=auto[final]"
            )
            stream = "final"

        filter_complex = ";".join(fc_parts)
        audio_index    = len(badge_info["icon_paths"]) + 1

        extra_inputs = []
        for p in badge_info["icon_paths"]:
            extra_inputs += ['-i', p]

        cmd = (
            ['ffmpeg', '-y', '-loop', '1', '-i', image_path]
            + extra_inputs
            + ['-i', audio_path,
               '-filter_complex', filter_complex,
               '-map', f'[{stream}]',
               '-map', f'{audio_index}:a',
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
               '-c:a', 'aac', '-b:a', '192k',
               '-pix_fmt', 'yuv420p',
               '-t', str(duration), '-shortest',
               output_path]
        )
    else:
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1', '-i', image_path,
            '-i', audio_path,
            '-vf', vf_chain,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
            '-c:a', 'aac', '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-t', str(duration), '-shortest',
            output_path
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
        fps      = 25
        font     = get_best_font()

        cmd = build_ffmpeg_command(
            image_path, audio_path, output_path,
            duration, fps, font,
            lyrics_segments=lyrics_segments,
            song_title=song_title,
            artist_name=artist_name,
            yt_channel=yt_channel,
            tt_channel=tt_channel
        )

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if proc.returncode == 0 and os.path.exists(output_path):
            jobs[job_id]['status']    = 'completed'
            jobs[job_id]['video_url'] = f"/videos/{job_id}/{job_id}.mp4"
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error']  = proc.stderr[-2000:]

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
    song_title  = data.get('title',  '').strip()
    artist_name = data.get('artist', 'SORLUNE').strip()
    yt_channel  = data.get('yt_channel', 'sorlune').strip()
    tt_channel  = data.get('tt_channel', 'sorlune08').strip()

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

            # Fallback: evenly distribute if Whisper failed
            if not lyrics_segments and lyrics_text:
                duration = get_audio_duration(audio_path)
                lines    = split_lyrics_lines(lyrics_text)
                if lines:
                    step    = max(duration / len(lines), 1.8)
                    current = 0.0
                    for line in lines:
                        lyrics_segments.append({
                            "start": round(current, 2),
                            "end":   round(min(current + step, duration), 2),
                            "text":  line
                        })
                        current += step

            generate_video_job(
                job_id, image_path, audio_path, output_path,
                lyrics_segments=lyrics_segments,
                song_title=song_title,
                artist_name=artist_name,
                yt_channel=yt_channel,
                tt_channel=tt_channel
            )

        except Exception as e:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error']  = str(e)

    thread = threading.Thread(target=run)
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
