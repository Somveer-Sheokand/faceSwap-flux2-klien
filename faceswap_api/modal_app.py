"""Face swap API on Modal.

Deploy:  modal deploy faceswap_api/modal_app.py   (run from the repo root)
Results are stored on the `faceswap-data` Modal Volume and served by this API (no external storage).
Auth: add a modal.Secret with FACESWAP_API_KEY to the functions to require X-API-Key (open by default)

POST /preprocess  {video_url, target_face_url, template_id?}  -> {job_id, template_id}
POST /swap        {template_id, source_face_url, enhance, codeformer_fidelity,
                   match_threshold, preserve_occlusions}      -> {job_id}
GET  /jobs/{id}         -> {status: running|done|failed, result|error}
GET  /jobs/{id}/result  -> the swapped mp4 (swap jobs)
GET  /templates/{id}    -> template metadata
"""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import modal

app = modal.App("faceswap-api")
volume = modal.Volume.from_name("faceswap-data", create_if_missing=True)

FACEFUSION_REF = "3.3.2"  # pin a FaceFusion release; bump deliberately


def _warm_models():
    import subprocess

    from insightface.app import FaceAnalysis

    FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]).prepare(ctx_id=-1)
    subprocess.run(["python", "facefusion.py", "force-download"], cwd="/facefusion", check=False)


image = (
    # CUDA + cuDNN runtime libs: onnxruntime-gpu silently falls back to CPU (~10 s/frame) without them
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("fastapi[standard]>=0.110", "httpx>=0.27", "pydantic>=2",
                 "insightface>=0.7.3", "opencv-python-headless>=4.9", "numpy<2")
    .run_commands(
        f"git clone --depth 1 --branch {FACEFUSION_REF} https://github.com/facefusion/facefusion /facefusion",
        "cd /facefusion && python install.py --onnxruntime cuda --skip-conda",
    )
    .run_function(_warm_models)
    .add_local_python_source("faceswap_api")
)



@app.function(image=image, cpu=4, volumes={"/data": volume}, timeout=1800)  # face analysis only: CPU, no GPU
def preprocess_fn(template_id: str, video_url: str, target_face_url: str) -> dict:
    from faceswap_api import engine

    meta = engine.preprocess(template_id, video_url, target_face_url)
    volume.commit()
    return meta


@app.function(image=image, cpu=2, volumes={"/data": volume}, timeout=900)
def make_template_fn(template_id: str, video_bytes: bytes, target_bytes: bytes) -> dict:
    """Build a template from uploaded files on CPU (for short test clips)."""
    from faceswap_api import engine

    meta = engine.preprocess_files(template_id, video_bytes, target_bytes)
    volume.commit()
    return meta


@app.function(image=image, gpu="L4", volumes={"/data": volume}, timeout=3600)
def bench_fn(run_id: str, source_urls: list, template_id: str, times: list, configs: list) -> dict:
    """Score swapper settings on single frames by identity similarity to the source photo."""
    from faceswap_api import bench

    volume.reload()
    try:
        return bench.run(run_id, source_urls, template_id, times, configs)
    finally:
        volume.commit()


@app.function(image=image, gpu="L4", volumes={"/data": volume}, timeout=1800)
def analyze_fn(run_id: str, video_url: str, source_urls: list, template_id: str) -> dict:
    """Per-frame check of a finished swap: is the face swapped and how close is it to the source?"""
    from faceswap_api import bench

    volume.reload()
    return bench.analyze(run_id, video_url, source_urls, template_id)


# cpu/memory matter: FaceFusion decodes, masks and pastes every frame on the CPU, and the default 0.125 core made it ~10 s/frame
@app.function(image=image, gpu="L4", cpu=8, memory=16384, volumes={"/data": volume}, timeout=3600)
def swap_chunk_fn(job_id: str, idx: int, chunk: str, sources: list, params: dict, best: list) -> dict:
    """Swap one chunk of the video on its own GPU."""
    from faceswap_api import engine

    volume.reload()  # see the chunk files the orchestrator committed
    try:
        return engine.swap_chunk(job_id, idx, chunk, sources, params, best)
    finally:
        volume.commit()


@app.function(image=image, cpu=4, volumes={"/data": volume}, timeout=3600)
def swap_fn(job_id: str, template_id: str, source_face_url: str, enhance: bool,
            codeformer_fidelity: float, match_threshold: float, preserve_occlusions: bool,
            options: dict) -> dict:
    """Split the video, swap the chunks in parallel on separate GPUs, join them and restore the audio."""
    from faceswap_api import engine

    options = dict(options)
    extra = [str(u) for u in options.pop("extra_source_face_urls", [])]
    n_chunks = options.pop("parallel_chunks", None)

    volume.reload()  # pick up templates committed by other containers
    job_dir = engine.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        meta = engine.load_template(template_id)
        params = engine.build_params(enhance, codeformer_fidelity, match_threshold, preserve_occlusions, **options)
        sources = engine.prepare_sources(job_dir, [source_face_url, *extra])
        n_chunks = int(n_chunks) if n_chunks else engine.auto_chunks(meta.get("frames") or 0, options.get("quality", "balanced"), enhance)
        chunks = engine.split_video(meta["video_path"], job_dir / "chunks", n_chunks)
        target = str(engine.template_dir(template_id) / "target_face")
        volume.commit()  # make sources and chunks visible to the GPU containers

        # find the target on CPU first; chunks without it skip the GPU entirely
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:  # onnxruntime releases the GIL
            found = dict(enumerate(pool.map(lambda c: engine.scan_chunk(c, target), chunks)))
        args = [(job_id, i, c, sources, params, list(found[i])) for i, c in enumerate(chunks) if found[i]]
        done = list(swap_chunk_fn.starmap(args)) if args else []
        done += [{"idx": i, "path": c, "swapped": False} for i, c in enumerate(chunks) if not found[i]]
        results = sorted(done, key=lambda r: r["idx"])

        volume.reload()  # pick up the swapped chunks
        final = job_dir / "output.mp4"
        has_audio = engine.assemble([r["path"] for r in results], meta["video_path"], final, meta.get("frames"), meta.get("fps"))
        result = {"path": str(final), "bytes": final.stat().st_size}  # download via GET /jobs/{call id}/result
        result.update(has_audio=has_audio, chunks=len(results),
                      chunks_swapped=sum(r["swapped"] for r in results))
        if not result["chunks_swapped"]:
            # nothing was swapped: say so instead of reporting a clean "done" with the original video
            result["warning"] = "target face not found in any chunk; the output is the unmodified video"
        return result
    finally:
        volume.commit()  # keep logs/output even when the swap fails


@app.function(image=image, volumes={"/data": volume}, min_containers=0)
@modal.asgi_app()
def web():
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field, HttpUrl

    from faceswap_api import engine

    api = FastAPI(title="Face Swap API", version="1.0.0")

    def auth(x_api_key: Optional[str] = Header(default=None)):
        key = os.environ.get("FACESWAP_API_KEY")
        if key and x_api_key != key:
            raise HTTPException(401, "invalid api key")

    class PreprocessIn(BaseModel):
        video_url: HttpUrl
        target_face_url: HttpUrl
        template_id: Optional[str] = None

    class SwapIn(BaseModel):
        template_id: str
        source_face_url: HttpUrl
        enhance: bool = False  # enhancers lowered identity similarity in benchmarks; opt in for extra sharpness
        codeformer_fidelity: float = Field(0.9, ge=0, le=1)
        match_threshold: float = Field(0.4, ge=0, le=1)
        preserve_occlusions: bool = True
        extra_source_face_urls: list[HttpUrl] = Field(default_factory=list, max_length=6)  # more photos of the same person
        # quality options
        quality: Literal["fast", "balanced", "best"] = "balanced"
        swapper_model: Optional[Literal["hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "inswapper_128",
                               "simswap_unofficial_512", "ghost_3_256", "hififace_unofficial_256"]] = None  # default from quality
        pixel_boost: Optional[Literal["256x256", "512x512", "768x768", "1024x1024"]] = None  # default from quality
        enhancer_model: Literal["codeformer", "gfpgan_1.4", "gpen_bfr_512", "restoreformer_plus_plus"] = "codeformer"
        enhancer_blend: Optional[int] = Field(None, ge=0, le=100)  # 100 = no enhancement; overrides fidelity
        detector_score: float = Field(0.3, ge=0.05, le=0.95)
        mask_blur: float = Field(0.4, ge=0, le=1)
        # speed options
        parallel_chunks: Optional[int] = Field(None, ge=1, le=16)  # default: auto, about one piece per 25 s (each piece is a GPU cold start)
        detect_rotated_faces: bool = False  # also search 90/270 degree rotated faces (about 3x slower detection)

    @api.get("/health")
    def health():
        return {"ok": True}

    @api.post("/preprocess", status_code=202, dependencies=[Depends(auth)])
    def preprocess(body: PreprocessIn):
        template_id = body.template_id or str(uuid.uuid4())
        try:
            engine.template_dir(template_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        call = preprocess_fn.spawn(template_id, str(body.video_url), str(body.target_face_url))
        return {"job_id": call.object_id, "template_id": template_id, "status_url": f"/jobs/{call.object_id}"}

    @api.get("/templates/{template_id}", dependencies=[Depends(auth)])
    def get_template(template_id: str):
        volume.reload()
        try:
            return engine.load_template(template_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "template not found (run /preprocess first)")

    @api.get("/templates/{template_id}/reference_frame", dependencies=[Depends(auth)])
    def reference_frame(template_id: str):
        volume.reload()
        try:
            path = engine.template_dir(template_id) / "reference_frame.jpg"
        except ValueError:
            raise HTTPException(404, "template not found")
        if not path.exists():
            raise HTTPException(404, "template not found")
        return FileResponse(path, media_type="image/jpeg")

    @api.post("/swap", status_code=202, dependencies=[Depends(auth)])
    def swap(body: SwapIn):
        volume.reload()
        try:
            engine.load_template(body.template_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "template not found (run /preprocess first)")
        job_id = str(uuid.uuid4())
        call = swap_fn.spawn(job_id, body.template_id, str(body.source_face_url), body.enhance,
                             body.codeformer_fidelity, body.match_threshold, body.preserve_occlusions,
                             body.model_dump(include={"extra_source_face_urls", "quality", "swapper_model", "pixel_boost", "enhancer_model",
                                                      "enhancer_blend", "detector_score", "mask_blur",
                                                      "parallel_chunks", "detect_rotated_faces"}))
        # job_id in the URL is the Modal call id; the output path is keyed by the returned value
        return {"job_id": call.object_id, "status_url": f"/jobs/{call.object_id}",
                "result_url": f"/jobs/{call.object_id}/result"}

    def _poll(job_id: str):
        # Accept ids with or without the "fc-" prefix (some clients strip it).
        if not job_id.startswith("fc-"):
            job_id = f"fc-{job_id}"
        try:
            call = modal.FunctionCall.from_id(job_id)
        except Exception:
            raise HTTPException(404, "job not found")
        try:
            return {"job_id": job_id, "status": "done", "result": call.get(timeout=0)}
        except TimeoutError:
            return {"job_id": job_id, "status": "running"}
        except modal.exception.NotFoundError:
            raise HTTPException(404, "job not found")
        except Exception as e:
            return {"job_id": job_id, "status": "failed", "error": str(e)}

    @api.get("/jobs/{job_id}", dependencies=[Depends(auth)])
    def job_status(job_id: str):
        return _poll(job_id)

    @api.get("/jobs/{job_id}/result", dependencies=[Depends(auth)])
    def job_result(job_id: str):
        st = _poll(job_id)
        if st["status"] != "done" or not isinstance(st["result"], dict) or "path" not in st["result"]:
            raise HTTPException(409, f"job is {st['status']}")
        volume.reload()
        return FileResponse(st["result"]["path"], media_type="video/mp4", filename="swap.mp4")

    return api
