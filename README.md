"""
processor.py — core video editing pipeline
Steps:
  1. Silence / gap detection  (FFmpeg silencedetect)
  2. Scene / retake detection (PySceneDetect)
  3. Best-shot scoring        (OpenCV sharpness + brightness)
  4. Cut & join               (FFmpeg concat demuxer)
  5. Background music         (Pixabay API or Jamendo or local fallback)
  6. Video effects            (FFmpeg vf filters)
"""

import subprocess, os, json, shutil, re, math
import cv2, numpy as np, requests


# ─── Royalty-free music fallback (CC0 / public domain samples) ─────────────────
FALLBACK_MUSIC = {
    "ambient":    "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0c6ff1bab.mp3",
    "upbeat":     "https://cdn.pixabay.com/download/audio/2021/11/25/audio_5bccf4caaf.mp3",
    "cinematic":  "https://cdn.pixabay.com/download/audio/2022/03/10/audio_c8c8a73467.mp3",
    "chill":      "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3",
    "corporate":  "https://cdn.pixabay.com/download/audio/2022/08/02/audio_884fe92c21.mp3",
}


class VideoProcessor:
    def __init__(self, input_path, output_folder, job_id, options, status_callback):
        self.input_path   = input_path
        self.output_folder = output_folder
        self.job_id       = job_id
        self.options      = options
        self.cb           = status_callback           # cb(progress: int, message: str)
        self.work_dir     = os.path.join(output_folder, job_id)
        os.makedirs(self.work_dir, exist_ok=True)

    # ── Public entry point ──────────────────────────────────────────────────────

    def process(self):
        self.cb(5,  "📋 Analysing video…")
        duration = self.get_duration(self.input_path)
        if duration < 1:
            raise RuntimeError("Video too short or unreadable.")

        self.cb(12, "🔇 Detecting silence & gaps…")
        segments = self.detect_silence()

        self.cb(28, "🎬 Detecting scenes & retakes…")
        scenes   = self.detect_scenes()

        self.cb(42, "🏆 Selecting best shots…")
        best     = self.select_best_shots(segments, scenes)

        self.cb(55, "✂️  Cutting & joining clips…")
        joined   = self.cut_and_join(best)

        if self.options.get('add_music', True):
            self.cb(70, "🎵 Adding background music…")
            joined = self.add_music(joined)

        self.cb(82, "✨ Applying video effects…")
        final    = self.apply_effects(joined)

        # Copy to a clean output path
        out_path = os.path.join(self.output_folder, f"{self.job_id}_final.mp4")
        shutil.copy2(final, out_path)
        return out_path

    # ── Step 1 – Silence detection ──────────────────────────────────────────────

    def detect_silence(self):
        noise_db  = self.options.get('silence_db',  -30)
        min_dur   = self.options.get('silence_dur',  0.5)

        cmd = [
            'ffmpeg', '-y', '-i', self.input_path,
            '-af', f'silencedetect=noise={noise_db}dB:d={min_dur}',
            '-f', 'null', '-'
        ]
        out = subprocess.run(cmd, capture_output=True, text=True).stderr

        starts, ends = [], []
        for line in out.split('\n'):
            m = re.search(r'silence_start:\s*([\d.]+)', line)
            if m: starts.append(float(m.group(1)))
            m = re.search(r'silence_end:\s*([\d.]+)', line)
            if m: ends.append(float(m.group(1)))

        total = self.get_duration(self.input_path)
        segments, prev = [], 0.0

        for ss, se in zip(starts, ends):
            if ss > prev + 0.15:
                segments.append({'start': prev, 'end': ss})
            prev = se

        if prev < total - 0.5:
            segments.append({'start': prev, 'end': total})

        if not segments:
            segments = [{'start': 0.0, 'end': total}]

        return [s for s in segments if (s['end'] - s['start']) > 0.3]

    # ── Step 2 – Scene / retake detection ──────────────────────────────────────

    def detect_scenes(self):
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector

            threshold = self.options.get('scene_threshold', 27.0)
            video   = open_video(self.input_path)
            sm      = SceneManager()
            sm.add_detector(ContentDetector(threshold=threshold))
            sm.detect_scenes(video)

            return [
                {'start': s[0].get_seconds(), 'end': s[1].get_seconds()}
                for s in sm.get_scene_list()
            ]
        except Exception as e:
            print(f"[scene-detect] {e}")
            return []

    # ── Step 3 – Score clip quality & pick best shot ────────────────────────────

    def score_clip(self, start, end, max_frames=12):
        cap = cv2.VideoCapture(self.input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = max(end - start, 0.01)
        step_sec = duration / max_frames

        scores = []
        for i in range(max_frames):
            t = start + i * step_sec
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            # penalise overexposed or underexposed frames
            brightness_score = 1 - abs(brightness - 128) / 128
            scores.append(sharpness * max(brightness_score, 0.1))

        cap.release()

        if not scores:
            return 0.0

        avg      = float(np.mean(scores))
        stability = 1.0 / (1.0 + float(np.std(scores)) / max(avg, 1))
        dur_bonus = min(1.0, duration / 8.0)   # slightly prefer longer clips
        return avg * stability * (0.7 + 0.3 * dur_bonus)

    def select_best_shots(self, segments, scenes):
        if not scenes:
            return segments

        scored = []
        for scene in scenes:
            for seg in segments:
                # segment must overlap the scene
                overlap_start = max(seg['start'], scene['start'])
                overlap_end   = min(seg['end'],   scene['end'])
                if overlap_end - overlap_start < 0.3:
                    continue
                clip = {'start': overlap_start, 'end': overlap_end}
                clip['score'] = self.score_clip(clip['start'], clip['end'])
                clip['_scene'] = id(scene)
                scored.append(clip)

        if not scored:
            return segments

        # Group retakes: same scene group → keep best
        by_scene = {}
        for c in scored:
            key = c['_scene']
            if key not in by_scene or c['score'] > by_scene[key]['score']:
                by_scene[key] = c

        result = sorted(by_scene.values(), key=lambda x: x['start'])

        # Remove helper key
        for r in result:
            r.pop('_scene', None)

        return result if result else segments

    # ── Step 4 – Cut & join ─────────────────────────────────────────────────────

    def cut_and_join(self, clips):
        clip_files = []

        for i, clip in enumerate(clips):
            dur = clip['end'] - clip['start']
            if dur < 0.3:
                continue
            out = os.path.join(self.work_dir, f"clip_{i:04d}.mp4")
            cmd = [
                'ffmpeg', '-y',
                '-ss', f"{clip['start']:.4f}",
                '-i', self.input_path,
                '-t', f"{dur:.4f}",
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac',
                '-avoid_negative_ts', 'make_zero',
                out
            ]
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(out) and os.path.getsize(out) > 1024:
                clip_files.append(out)

        if not clip_files:
            return self.input_path

        concat_txt = os.path.join(self.work_dir, 'concat.txt')
        with open(concat_txt, 'w') as f:
            for cf in clip_files:
                f.write(f"file '{os.path.abspath(cf)}'\n")

        joined = os.path.join(self.work_dir, 'joined.mp4')
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', concat_txt,
            '-c', 'copy',
            joined
        ]
        subprocess.run(cmd, capture_output=True)
        return joined if (os.path.exists(joined) and os.path.getsize(joined) > 1024) else self.input_path

    # ── Step 5 – Music ─────────────────────────────────────────────────────────

    def _download_music(self):
        genre      = self.options.get('music_genre', 'ambient')
        custom_url = self.options.get('custom_music_url', '')
        pix_key    = self.options.get('pixabay_key', '').strip()

        dest = os.path.join(self.work_dir, 'music.mp3')

        # 1) Pixabay API (if key provided)
        if pix_key:
            try:
                api_url = f"https://pixabay.com/api/music/?key={pix_key}&q={genre}&per_page=5&category=music"
                resp = requests.get(api_url, timeout=10)
                hits = resp.json().get('hits', [])
                if hits:
                    audio_resp = requests.get(hits[0]['audio'], timeout=30)
                    with open(dest, 'wb') as f: f.write(audio_resp.content)
                    return dest
            except Exception as e:
                print(f"[music] Pixabay error: {e}")

        # 2) Custom URL
        if custom_url:
            try:
                audio_resp = requests.get(custom_url, timeout=30)
                with open(dest, 'wb') as f: f.write(audio_resp.content)
                return dest
            except Exception as e:
                print(f"[music] Custom URL error: {e}")

        # 3) Bundled fallback
        fallback = FALLBACK_MUSIC.get(genre, FALLBACK_MUSIC['ambient'])
        try:
            audio_resp = requests.get(fallback, timeout=30)
            with open(dest, 'wb') as f: f.write(audio_resp.content)
            return dest
        except Exception as e:
            print(f"[music] Fallback error: {e}")

        return None

    def add_music(self, video_path):
        music = self._download_music()
        if not music:
            return video_path

        duration = self.get_duration(video_path)
        vol      = self.options.get('music_volume', 0.15)
        out      = os.path.join(self.work_dir, 'with_music.mp4')

        # Mix voice (full) with looped music (background)
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-stream_loop', '-1', '-i', music,
            '-filter_complex',
            (
                f'[0:a]volume=1.0[va];'
                f'[1:a]volume={vol},atrim=0:{duration:.4f}[ma];'
                f'[va][ma]amix=inputs=2:duration=first[aout]'
            ),
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'copy', '-c:a', 'aac',
            '-t', f"{duration:.4f}",
            out
        ]
        r = subprocess.run(cmd, capture_output=True)
        return out if (os.path.exists(out) and os.path.getsize(out) > 1024) else video_path

    # ── Step 6 – Effects ────────────────────────────────────────────────────────

    def apply_effects(self, video_path):
        effect  = self.options.get('effect', 'none')
        duration = self.get_duration(video_path)
        fade_d  = min(1.0, duration * 0.05)
        fade_out_t = max(0, duration - fade_d)

        vf_parts = []

        effect_map = {
            'cinematic': 'curves=vintage,vignette=angle=PI/4',
            'bright':    'eq=brightness=0.07:saturation=1.4:contrast=1.1',
            'warm':      'colorbalance=rs=0.15:gs=-0.03:bs=-0.12:rm=0.07:gm=-0.02:bm=-0.05',
            'cool':      'colorbalance=rs=-0.12:bs=0.12:rm=-0.06:bm=0.06',
            'bw':        'hue=s=0,curves=vintage',
            'dramatic':  'eq=contrast=1.3:brightness=-0.05:saturation=0.8,vignette',
            'pastel':    'eq=saturation=0.65:brightness=0.05,curves=psych',
        }

        if effect in effect_map:
            vf_parts.append(effect_map[effect])

        # Fade in/out always applied
        vf_parts.append(f"fade=t=in:st=0:d={fade_d:.2f}:alpha=0")
        vf_parts.append(f"fade=t=out:st={fade_out_t:.2f}:d={fade_d:.2f}:alpha=0")

        # Add smooth motion blur if requested
        if self.options.get('motion_blur', False):
            vf_parts.append('tmix=frames=3:weights=1 1 1')

        vf  = ','.join(vf_parts)
        out = os.path.join(self.work_dir, 'with_effects.mp4')

        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', vf,
            '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'copy',
            out
        ]
        r = subprocess.run(cmd, capture_output=True)
        return out if (os.path.exists(out) and os.path.getsize(out) > 1024) else video_path

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def get_duration(self, path):
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json', path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(json.loads(r.stdout)['format']['duration'])
        except Exception:
            return 0.0
