"""Static analyzer for the faceswap pipeline: flags settings and patterns that hurt swap quality.

Run from the repo root:  python faceswap_api/analyze_code.py
Exits non-zero when any finding is reported, so it can run in CI.
"""
import re
import sys
from pathlib import Path

PKG = Path(__file__).parent

# (id, regex, message) - each regex is matched per line against every .py file in the package.
RULES = [
    ("external-storage", r"(?i)cloudinary|boto3|upload_(video|image)",
     "external storage dependency; results are served from the Modal Volume"),
    ("jpeg-source", r"source_\{?i\}?\.jpe?g",
     "source photos re-saved as JPEG; use lossless PNG so identity detail is not degraded"),
    ("loose-match", r"match_threshold[:=]\s*(?:float\s*=\s*Field\()?0\.(0\d?|1\d?|2\d?|3[0-4]?)\b(?!\d)",
     "match_threshold below 0.35 lets FaceFusion swap bystanders and wrong faces"),
    ("enhancer-default-on", r"enhance(?::\s*bool)?\s*=\s*True",
     "face enhancers lowered identity similarity in benchmarks; keep them opt-in"),
    ("low-quality-intermediate", r"--output-video-quality\W+\s*\"?([0-7]\d?)\"?\b",
     "lossy intermediate chunk encode; chunks are re-encoded again when assembled"),
    ("hard-blur", r"mask_blur\b[^=\n]*=\s*(?:Field\()?(0\.[6-9]|1)\b",
     "heavy mask blur smears the face edge and shows a halo"),
    ("high-det-score", r"detector_score\b[^=\n]*=\s*(?:Field\()?0\.[6-9]",
     "high detector score drops faces in profile/motion blur, causing swap flicker"),
]

CHECKS_PRESENT = [
    ("--output-video-quality", "engine.py", "FaceFusion intermediate encode quality is not set (default is lossy)"),
    ("--face-mask-types", "engine.py", "no face mask types passed to FaceFusion"),
    ("occlusion", "engine.py", "occlusion mask unused: hands/glasses in front of the face will be painted over"),
]


def main() -> int:
    findings = []
    for path in sorted(PKG.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for rid, rx, msg in RULES:
                if re.search(rx, line):
                    findings.append((path.name, n, rid, msg))
    for needle, fname, msg in CHECKS_PRESENT:
        if needle not in (PKG / fname).read_text(encoding="utf-8"):
            findings.append((fname, 0, "missing", msg))

    for f, n, rid, msg in findings:
        print(f"{f}:{n}: [{rid}] {msg}")
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
