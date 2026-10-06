"""Benchmark swapper settings on single frames, scoring identity similarity to the source photo.

Runs inside the Modal GPU container (see bench_fn in modal_app.py).
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import engine
from .utils import download


def _faces_by_x(img):
    return sorted(engine._get_analyser().get(img), key=lambda f: f.bbox[0])


def run(run_id: str, source_urls: list[str], template_id: str, times: list[float], configs: list[dict]) -> dict:
    work = engine.JOBS_DIR / f"bench_{run_id}"
    work.mkdir(parents=True, exist_ok=True)
    meta = engine.load_template(template_id)
    sources = engine.prepare_sources(work, source_urls)
    app = engine._get_analyser()
    src_emb = np.mean([_faces_by_x(cv2.imread(s))[0].normed_embedding for s in sources], axis=0)
    temb = engine.target_embedding(str(engine.template_dir(template_id) / "target_face"))

    # frames from the template video, and the target's face position in each
    frames = []
    for t in times:
        path = work / f"frame_{t}.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", meta["video_path"], "-frames:v", "1", str(path)],
                       capture_output=True, check=True)
        img = cv2.imread(str(path))
        faces = _faces_by_x(img)
        if not faces:
            continue
        pos = int(np.argmax([engine._cos(temb, f.normed_embedding) for f in faces]))
        frames.append({"t": t, "path": str(path), "pos": pos, "img": img})

    rows, tiles = [], {}
    for cfg in configs:
        name = cfg["name"]
        params = engine.build_params(**{k: v for k, v in cfg.items() if k not in ("name", "extra")})
        scores = []
        for fr in frames:
            out = work / f"{name}_{fr['t']}.jpg"
            cmd = [
                "python", "facefusion.py", "headless-run", "-s", *sources, "-t", fr["path"], "-o", str(out),
                "--processors", "face_swapper", *(["face_enhancer"] if params["enhance"] else []),
                "--face-swapper-model", params["swapper_model"], "--face-swapper-pixel-boost", params["pixel_boost"],
                "--face-detector-model", params["detector_model"], "--face-detector-size", params["detector_size"],
                "--face-detector-score", str(params["detector_score"]), "--face-mask-blur", str(params["mask_blur"]),
                "--face-selector-mode", "reference", "--reference-face-position", str(fr["pos"]),
                "--reference-frame-number", "0", "--reference-face-distance", str(round(1.0 - params["match_threshold"], 3)),
                "--face-mask-types", "box", *(["occlusion"] if params["preserve_occlusions"] else []),
                *cfg.get("extra", []),
            ]
            if params["enhance"]:
                cmd += ["--face-enhancer-model", params["enhancer_model"], "--face-enhancer-blend", str(params["enhancer_blend"])]
            r = subprocess.run(cmd, cwd=engine.FACEFUSION_DIR, capture_output=True, text=True)
            if not out.exists():
                scores.append(None)
                print(f"[bench {name} t={fr['t']}] failed: {(r.stdout + r.stderr)[-300:]}", flush=True)
                continue
            faces = _faces_by_x(cv2.imread(str(out)))
            s = engine._cos(src_emb, faces[min(fr["pos"], len(faces) - 1)].normed_embedding) if faces else None
            scores.append(None if s is None else round(s, 4))
            print(f"[bench {name} t={fr['t']}] identity={scores[-1]}", flush=True)
            # crop the target face for the contact sheet
            if faces:
                f = faces[min(fr["pos"], len(faces) - 1)]
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                pad = int((x2 - x1) * 0.5)
                im = cv2.imread(str(out))
                crop = im[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                tiles.setdefault(name, []).append(cv2.resize(crop, (240, 300)))
        valid = [s for s in scores if s is not None]
        rows.append({"name": name, "scores": scores, "mean": round(float(np.mean(valid)), 4) if valid else None})

    # contact sheet: source | original crops | one row per config
    def crop_face(img, f):
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        pad = int((x2 - x1) * 0.5)
        return cv2.resize(img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad], (240, 300))

    src_img = cv2.imread(sources[0])
    header = [crop_face(src_img, _faces_by_x(src_img)[0])]
    for fr in frames:
        header.append(crop_face(fr["img"], _faces_by_x(fr["img"])[fr["pos"]]))
    sheet = [np.hstack(header + [np.zeros((300, 240, 3), np.uint8)] * max(0, len(header) - len(header)))]
    for row in rows:
        t = tiles.get(row["name"], [])
        label = np.zeros((300, 240, 3), np.uint8)
        cv2.putText(label, row["name"][:20], (6, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(label, f"id={row['mean']}", (6, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        line = [label] + t
        while len(line) < len(header):
            line.append(np.zeros((300, 240, 3), np.uint8))
        sheet.append(np.hstack(line[:len(header)]))
    sheet_path = work / "sheet.jpg"
    cv2.imwrite(str(sheet_path), np.vstack(sheet))
    return {"sheet_path": str(sheet_path), "results": sorted(rows, key=lambda r: -(r["mean"] or -1))}


def analyze(run_id: str, video_url: str, source_urls: list[str], template_id: str, step_s: float = 0.5) -> dict:
    """Per sampled frame: is the target person's face swapped, and how close is it to the source photo?"""
    work = engine.JOBS_DIR / f"analyze_{run_id}"
    work.mkdir(parents=True, exist_ok=True)
    vid = download(video_url, work / "out.mp4")
    sources = engine.prepare_sources(work, source_urls)
    src_emb = np.mean([_faces_by_x(cv2.imread(s))[0].normed_embedding for s in sources], axis=0)
    temb = engine.target_embedding(str(engine.template_dir(template_id) / "target_face"))
    meta = engine.load_template(template_id)
    orig = cv2.VideoCapture(meta["video_path"])
    out = cv2.VideoCapture(str(vid))
    fps = out.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    for n in range(0, total, max(1, int(round(step_s * fps)))):
        out.set(cv2.CAP_PROP_POS_FRAMES, n)
        orig.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok1, fo = out.read()
        ok2, fg = orig.read()
        if not (ok1 and ok2):
            continue
        # which face in the ORIGINAL frame is the target actor?
        of = _faces_by_x(fg)
        if not of:
            rows.append({"t": round(n / fps, 2), "faces": 0})
            continue
        sims = [engine._cos(temb, f.normed_embedding) for f in of]
        pos = int(np.argmax(sims))
        if sims[pos] < engine.MIN_TARGET_SIMILARITY:
            rows.append({"t": round(n / fps, 2), "faces": len(of), "target": False})
            continue
        # same position in the OUTPUT frame
        nf = _faces_by_x(fo)
        if not nf:
            rows.append({"t": round(n / fps, 2), "faces": len(of), "target": True, "out_face": False})
            continue
        # pick the output face whose box overlaps the original target box most
        tb = of[pos].bbox

        def iou(b):
            x1, y1 = max(b[0], tb[0]), max(b[1], tb[1])
            x2, y2 = min(b[2], tb[2]), min(b[3], tb[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a = (b[2] - b[0]) * (b[3] - b[1]) + (tb[2] - tb[0]) * (tb[3] - tb[1]) - inter
            return inter / a if a > 0 else 0

        f = max(nf, key=lambda f: iou(f.bbox))
        w = int(tb[2] - tb[0])
        rows.append({
            "t": round(n / fps, 2), "face_px": w,
            "orig_vs_target": round(sims[pos], 3),
            "out_vs_source": round(engine._cos(src_emb, f.normed_embedding), 3),
            "out_vs_orig_actor": round(engine._cos(temb, f.normed_embedding), 3),
        })
    scored = [r for r in rows if "out_vs_source" in r]
    return {
        "samples": len(rows), "target_frames": len(scored),
        "mean_out_vs_source": round(float(np.mean([r["out_vs_source"] for r in scored])), 3) if scored else None,
        "not_swapped": [r for r in scored if r["out_vs_orig_actor"] > r["out_vs_source"]],
        "missing_face": [r for r in rows if r.get("out_face") is False],
        "rows": rows,
    }
