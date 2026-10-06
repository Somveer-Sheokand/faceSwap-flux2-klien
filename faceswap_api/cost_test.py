"""Run one preprocess + swap against a deployed API and print how long each step took.

Use a short clip (2-3 s) to estimate cost before doing real runs. Rough cost = seconds x GPU rate
(L4 is about $0.000222/s, check modal.com/pricing).

    python faceswap_api/cost_test.py --url https://<workspace>--faceswap-api-web.modal.run \
        --video https://.../clip.mp4 --target https://.../actor.jpg --source https://.../me.jpg
"""
import argparse
import json
import time
import urllib.request


def call(base, path, body=None, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def wait(base, job_id, key, every=5, limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        st = call(base, f"/jobs/{job_id}", key=key)
        if st["status"] != "running":
            return st, time.time() - t0
        time.sleep(every)
    raise TimeoutError(job_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--key")
    ap.add_argument("--quality", default="fast", choices=["fast", "balanced", "best"])
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--gpu-rate", type=float, default=0.000222, help="$ per GPU-second (L4 default)")
    a = ap.parse_args()

    print("health:", call(a.url, "/health"))

    pre = call(a.url, "/preprocess", {"video_url": a.video, "target_face_url": a.target}, a.key)
    st, t_pre = wait(a.url, pre["job_id"], a.key)
    print(f"preprocess: {st['status']} in {t_pre:.0f}s")
    if st["status"] != "done":
        raise SystemExit(st)

    sw = call(a.url, "/swap", {"template_id": pre["template_id"], "source_face_url": a.source,
                               "quality": a.quality, "parallel_chunks": a.chunks}, a.key)
    st, t_swap = wait(a.url, sw["job_id"], a.key)
    print(f"swap: {st['status']} in {t_swap:.0f}s")
    if st["status"] != "done":
        raise SystemExit(st)

    gpu_s = (t_pre + t_swap) * a.chunks
    print("result:", st["result"].get("url"))
    print(f"rough GPU cost upper bound: ${gpu_s * a.gpu_rate:.3f} (wall time x chunks x L4 rate)")


if __name__ == "__main__":
    main()
