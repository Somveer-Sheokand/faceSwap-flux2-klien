"""Preprocess (face analysis) and swap (FaceFusion CLI) logic.

A swap splits the template video into chunks, swaps every chunk on its own GPU in parallel
(see modal_app.py), then joins the chunks again and puts the original audio back.
"""
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .utils import download

DATA_DIR = Path("/data")  # Modal Volume mount
TEMPLATES_DIR = DATA_DIR / "templates"
JOBS_DIR = DATA_DIR / "jobs"
FACEFUSION_DIR = Path("/facefusion")
SCAN_FRAMES = 60  # frames sampled during preprocess
CHUNK_SCAN_FRAMES = 16  # frames sampled per chunk when looking for the target
# Frames each GPU container should handle. Heavier settings get fewer frames per GPU so the wall time
# stays bounded; every extra chunk costs one GPU cold start, so light settings keep big chunks.
# Measured: balanced + enhancer = 1.9 s/frame on an L4 at 1080x1920 (388 frames = 12.5 min in one chunk).
FRAMES_PER_CHUNK = {"fast": 500, "balanced": 250, "best": 100}
ENHANCE_FRAME_DIVISOR = 2  # the enhancer adds a second network per frame
MIN_TARGET_SIMILARITY = 0.35  # below this a chunk is assumed not to contain the target person

# Chosen by benchmark (identity similarity to the source photo, see bench.py):
# inswapper_128 beat the HyperSwap/SimSwap/GhostFace/HiFiFace models, higher pixel boost helps with
# diminishing returns, and the face enhancers lowered identity similarity.
QUALITY_PRESETS = {
    # swapper_model, pixel_boost, detector_model, detector_size
    "fast": ("inswapper_128", "256x256", "yolo_face", "640x640"),
    "balanced": ("inswapper_128", "512x512", "yolo_face", "640x640"),
    "best": ("inswapper_128", "1024x1024", "yolo_face", "640x640"),
}

_analyser = None


def _get_analyser():
    global _analyser
    if _analyser is None:
        from insightface.app import FaceAnalysis

        # CPU containers (preprocess, orchestrator) have the CUDA libs but no driver: starting CUDA there crashes
        gpu = os.path.exists("/dev/nvidiactl")
        _analyser = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu
                                 else ["CPUExecutionProvider"])
        _analyser.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
    return _analyser


def template_dir(template_id: str) -> Path:
    if not template_id or not all(c.isalnum() or c in "-_" for c in template_id):
        raise ValueError("invalid template_id")
    return TEMPLATES_DIR / template_id


def load_template(template_id: str) -> dict:
    meta = template_dir(template_id) / "meta.json"
    if not meta.exists():
        raise FileNotFoundError(template_id)
    return json.loads(meta.read_text())


def _largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def target_embedding(target_path: str):
    img = cv2.imread(str(target_path))
    if img is None:
        raise ValueError("target face is not a readable image")
    faces = _get_analyser().get(img)
    if not faces:
        raise ValueError("no face found in the target face image")
    return _largest(faces).normed_embedding


def find_target(video_path: str, temb, n_frames: int):
    """Sample frames and return (info, best_frame, total, fps, width, height), where info is
    (similarity, frame_number, position) of the face closest to `temb`, or None if no faces."""
    app = _get_analyser()
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("cannot read video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    best, best_frame = None, None
    for n in np.linspace(0, total - 1, min(n_frames, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n))
        ok, frame = cap.read()
        if not ok:
            continue
        # FaceFusion orders faces left-to-right for reference_face_position
        faces = sorted(app.get(frame), key=lambda f: f.bbox[0])
        for pos, f in enumerate(faces):
            s = _cos(temb, f.normed_embedding)
            if best is None or s > best[0]:
                best, best_frame = (s, int(n), pos), frame
    cap.release()
    return best, best_frame, total, fps, width, height


def preprocess(template_id: str, video_url: str, target_face_url: str) -> dict:
    """Download the video and find where the target face is most clearly visible."""
    tdir = template_dir(template_id)
    tdir.mkdir(parents=True, exist_ok=True)
    video = download(video_url, tdir / "video.mp4")
    target = download(target_face_url, tdir / "target_face")
    return build_template(template_id, tdir, video, target)


def preprocess_files(template_id: str, video_bytes: bytes, target_bytes: bytes) -> dict:
    """Same as preprocess, from uploaded files (used for small test clips)."""
    tdir = template_dir(template_id)
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "video.mp4").write_bytes(video_bytes)
    (tdir / "target_face").write_bytes(target_bytes)
    return build_template(template_id, tdir, tdir / "video.mp4", tdir / "target_face")


def build_template(template_id: str, tdir: Path, video: Path, target: Path) -> dict:
    temb = target_embedding(str(target))
    best, best_frame, total, fps, width, height = find_target(str(video), temb, SCAN_FRAMES)
    if best is None:
        raise ValueError("no faces found in video")

    preview = tdir / "reference_frame.jpg"
    cv2.imwrite(str(preview), best_frame)

    meta = {
        "template_id": template_id,
        "reference_frame_url": f"/templates/{template_id}/reference_frame",
        "video_path": str(video),
        "target_face_path": str(target),
        "fps": fps,
        "frames": total,
        "width": width,
        "height": height,
        "reference_frame_number": best[1],
        "reference_face_position": best[2],
        "reference_similarity": round(best[0], 4),
        "target_found": best[0] >= MIN_TARGET_SIMILARITY,
    }
    if not meta["target_found"]:
        meta["warning"] = (f"the target face best matches the video at similarity {best[0]:.2f} "
                           f"(< {MIN_TARGET_SIMILARITY}): it is probably not in this video, so swaps will leave it unchanged")
    (tdir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------------------------------------
# swap: orchestration pieces
# ----------------------------------------------------------------------------------------------

def build_params(enhance=False, codeformer_fidelity=0.9, match_threshold=0.4, preserve_occlusions=True,
                 quality="balanced", swapper_model=None, pixel_boost=None, enhancer_model="codeformer",
                 enhancer_blend=None, detector_score=0.3, mask_blur=0.4, detect_rotated_faces=False) -> dict:
    preset_model, preset_boost, detector_model, detector_size = QUALITY_PRESETS[quality]
    # FaceFusion has no fidelity knob. Its enhancer blend is the share of the *un-enhanced*
    # (swapped) frame kept: 100 = no enhancement. Higher fidelity -> keep more of the swap.
    blend = enhancer_blend if enhancer_blend is not None else int(round(codeformer_fidelity * 100))
    return {
        "enhance": enhance,
        "enhancer_model": enhancer_model,
        "enhancer_blend": max(0, min(100, blend)),
        "match_threshold": match_threshold,
        "preserve_occlusions": preserve_occlusions,
        "swapper_model": swapper_model or preset_model,
        "pixel_boost": pixel_boost or preset_boost,
        "detector_model": detector_model,
        "detector_size": detector_size,
        "detector_score": detector_score,
        "mask_blur": mask_blur,
        "angles": ["0", "90", "270"] if detect_rotated_faces else ["0"],
    }


def prepare_sources(job_dir: Path, urls: list[str]) -> list[str]:
    """Download the source photos. Several photos of one person are averaged by FaceFusion."""
    sources = []
    for i, url in enumerate(urls):
        raw = download(str(url), job_dir / f"source_raw_{i}")
        img = cv2.imread(str(raw))
        if img is None:
            raise ValueError(f"source face #{i} is not a readable image: {url}")
        path = job_dir / f"source_{i}.png"  # lossless: no extra JPEG artifacts in the identity; FaceFusion reads the type from the extension
        cv2.imwrite(str(path), img)
        sources.append(str(path))
    return sources


def split_video(video: str, chunks_dir: Path, n_chunks: int) -> list[str]:
    """Split on keyframes without re-encoding. Audio is dropped here and added back at the end."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                           capture_output=True, text=True)
    duration = float(probe.stdout.strip())
    seg = max(1.0, duration / max(1, n_chunks))
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-an", "-c:v", "copy", "-f", "segment", "-segment_time", f"{seg:.3f}",
         "-reset_timestamps", "1", str(chunks_dir / "chunk_%03d.mp4")],
        capture_output=True, text=True)
    chunks = sorted(str(p) for p in chunks_dir.glob("chunk_*.mp4"))
    if r.returncode != 0 or not chunks:
        raise RuntimeError("video split failed:\n" + r.stderr[-800:])
    return chunks


def _run_streaming(cmd: list[str], tag: str, log_path: Path) -> tuple[int, str]:
    """Run a command, echoing its progress lines to the Modal logs while it runs."""
    lines, buf = [], ""
    with subprocess.Popen(cmd, cwd=FACEFUSION_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as popen:
        while ch := popen.stdout.read(1):
            if ch in "\r\n":
                if buf.strip():
                    lines.append(buf)
                    print(f"[{tag}] {buf}", flush=True)
                buf = ""
            else:
                buf += ch
        popen.wait()
    text = "\n".join(lines)
    log_path.write_text(text)
    return popen.returncode, text


def scan_chunk(chunk: str, target_face_path: str):
    """Find the target in a chunk. Returns (similarity, frame_number, face_position) or None if absent.
    Runs on the cheap CPU orchestrator so chunks without the target never start a GPU container."""
    best, _, _, _, _, _ = find_target(chunk, target_embedding(target_face_path), CHUNK_SCAN_FRAMES)
    if best is None or best[0] < MIN_TARGET_SIMILARITY:
        return None
    return best


def auto_chunks(frames: int, quality: str = "balanced", enhance: bool = False, max_chunks: int = 8) -> int:
    """Number of GPU chunks: enough that each handles a bounded number of frames for the chosen settings."""
    per_chunk = FRAMES_PER_CHUNK.get(quality, 250) // (ENHANCE_FRAME_DIVISOR if enhance else 1)
    return max(1, min(max_chunks, -(-int(frames) // per_chunk)))  # ceil division


def swap_chunk(job_id: str, idx: int, chunk: str, sources: list[str], params: dict, best) -> dict:
    """Swap one chunk (`best` comes from scan_chunk) on the GPU."""
    job_dir = JOBS_DIR / job_id

    out = job_dir / f"swapped_{idx:03d}.mp4"
    p = params
    cmd = [
        "python", "facefusion.py", "headless-run",
        "-s", *sources, "-t", chunk, "-o", str(out),
        "--processors", "face_swapper", *(["face_enhancer"] if p["enhance"] else []),
        "--face-swapper-model", p["swapper_model"],
        "--face-swapper-pixel-boost", p["pixel_boost"],
        "--face-detector-model", p["detector_model"], "--face-detector-size", p["detector_size"],
        "--face-detector-angles", *p["angles"],
        "--face-detector-score", str(p["detector_score"]),
        "--face-mask-blur", str(p["mask_blur"]),
        "--face-selector-mode", "reference",
        "--reference-face-position", str(best[2]),
        "--reference-frame-number", str(best[1]),
        "--reference-face-distance", str(round(1.0 - p["match_threshold"], 3)),
        "--face-mask-types", "box", *(["occlusion"] if p["preserve_occlusions"] else []),
        # near-lossless intermediate: the chunks are re-encoded once more in assemble()
        "--temp-frame-format", "bmp", "--output-video-quality", "95", "--output-video-preset", "medium",
    ]
    if p["enhance"]:
        cmd += ["--face-enhancer-model", p["enhancer_model"], "--face-enhancer-blend", str(p["enhancer_blend"])]

    code, text = _run_streaming(cmd, f"chunk {idx}", job_dir / f"log_{idx:03d}.txt")
    if code != 0 or not out.exists():
        raise RuntimeError(f"facefusion failed on chunk {idx}:\n" + " ".join(cmd) + "\n" + text[-1500:])
    return {"idx": idx, "path": str(out), "swapped": True, "similarity": round(best[0], 4)}


def assemble(chunk_paths: list[str], original: str, dest: Path, max_frames: int | None = None,
             fps: float | None = None) -> bool:
    """Join the chunks (in order) and add the original audio. Returns whether audio was added.

    Chunks come from different encoders (FaceFusion vs. stream-copied originals), so the join
    re-encodes once at high quality instead of risking glitches at the seams.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
         original], capture_output=True, text=True)
    has_audio = "audio" in probe.stdout

    listing = dest.parent / "concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in chunk_paths))
    # cap the result at the original's duration: joining re-encoded chunks can add a stray frame
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", original],
                         capture_output=True, text=True).stdout.strip()
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if has_audio:
        cmd += ["-i", original, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    # Cap at the original length with -t, which trims video and audio together. (-frames:v stopped the
    # muxer early and cut the audio down to a single packet.)
    cmd += (["-t", f"{max_frames / fps:.4f}"] if max_frames and fps else [])
    cmd += ["-c:v", "libx264", "-crf", "17", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dest.exists():
        raise RuntimeError("assemble failed:\n" + r.stderr[-1000:])
    return has_audio
