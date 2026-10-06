import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

import httpx

MAX_DOWNLOAD_MB = 500
MAX_REDIRECTS = 5


def check_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts resolving to private/loopback addresses (SSRF guard)."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError(f"invalid url: {url}")
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror:
        raise ValueError(f"cannot resolve host: {p.hostname}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"url resolves to a non-public address: {p.hostname}")


def download(url: str, dest: Path) -> Path:
    """Download `url` to `dest`. Redirects are followed by hand so every hop passes check_url
    (follow_redirects=True would let a public URL bounce us to 169.254.169.254 or an internal host)."""
    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    with httpx.Client(follow_redirects=False, timeout=120) as client:
        for _ in range(MAX_REDIRECTS + 1):
            check_url(url)
            with client.stream("GET", url) as r:
                if r.is_redirect:
                    location = r.headers.get("location")
                    if not location:
                        raise ValueError("redirect without a location")
                    url = str(r.url.join(location))
                    continue
                r.raise_for_status()
                size = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(f"download exceeds {MAX_DOWNLOAD_MB} MB")
                        f.write(chunk)
                return dest
    raise ValueError(f"too many redirects (> {MAX_REDIRECTS})")
