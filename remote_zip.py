"""Read members of a (multi-part) remote ZIP via HTTP Range requests - fetches only the bytes needed (central directory + chosen members)."""
import io, zipfile, urllib.request
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) kraken-bridge-db/1.0"}
class RangeFile(io.RawIOBase):
    def __init__(self, urls):
        self.urls, self.sizes = urls, []
        for u in urls:
            r = urllib.request.urlopen(urllib.request.Request(u, method="HEAD", headers=UA), timeout=60); self.sizes.append(int(r.headers["Content-Length"]))
        self.size, self.pos = sum(self.sizes), 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def _get(self, i, a, b):
        req = urllib.request.Request(self.urls[i], headers={"Range": f"bytes={a}-{b}", **UA})
        for k in range(5):
            try: return urllib.request.urlopen(req, timeout=120).read()
            except Exception:
                if k == 4: raise
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos); out = bytearray(); p = self.pos
        while len(out) < n:
            off = p; i = 0
            while off >= self.sizes[i]: off -= self.sizes[i]; i += 1
            take = min(n - len(out), self.sizes[i] - off); out += self._get(i, off, off + take - 1); p += take
        self.pos = p; return bytes(out)
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
def open_zip(urls):
    return zipfile.ZipFile(io.BufferedReader(RangeFile(urls), buffer_size=1 << 20))
