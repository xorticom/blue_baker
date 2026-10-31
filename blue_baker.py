#!/usr/bin/env python3
# Blue Baker -- text in, narration out.
# Copyright (C) 2026 XortiCom LLC
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details. You should have received a copy of the GNU General Public
# License along with this program (LICENSE). If not, see
# <https://www.gnu.org/licenses/>.
"""
Blue Baker turns text into a WAV file with a neural voice. It is the part of
XortiCom BLUE that speaks, kept as a program of its own: it uses eSpeak NG
and phonemizer (both GPL v3) to work out how words are said, and Kokoro
(Apache 2.0) or Piper (MIT) to say them.

    Blue Baker --say "Hello there." --voice am_michael --out hello.wav
    Blue Baker --phonemize "produce" --lang en-us
    Blue Baker --serve         (JSON lines on stdin / stdout -- how BLUE uses it)
    Blue Baker --about

Model files are not inside Blue Baker; pass where they are with --models
(the folder holding kokoro/ and piper/), or in each --serve request.

THE SERVE PROTOCOL. One JSON object per line on stdin, one reply per line on
stdout. Nothing else is ever written to stdout.

  {"op": "ping"}
      -> {"ok": true, "version": "...", "name": "Blue Baker"}
  {"op": "bake", "id": 7, "engine": "kokoro", "voice": "am_michael",
   "text": "...", "out": "C:/.../clip.wav", "lang": "en-us", "speed": 1.0,
   "kokoro_onnx": ".../kokoro-v1.0.onnx", "kokoro_voices": ".../voices-v1.0.bin",
   "threads": 4}
  {"op": "bake", "id": 8, "engine": "piper", "model": ".../en_GB-alba-medium.onnx",
   "text": "...", "out": "..."}
      -> {"id": 7, "ok": true}   or   {"id": 7, "ok": false, "error": "..."}
  {"op": "phonemize", "id": 9, "text": "produce", "lang": "en-us"}
      -> {"id": 9, "ok": true, "phonemes": "..."}
  {"op": "quit"}
"""
import argparse
import json
import os
import subprocess
import sys
import threading

NAME = "Blue Baker"
VERSION = "1.0.0"
SOURCE_URL = "https://github.com/xorticom/blue_baker"

ABOUT = """Blue Baker %s -- text in, narration out.
Copyright (C) 2026 XortiCom LLC. Free software under the GNU General Public
License version 3 or later; see LICENSE. Source: %s

Built with:
  Kokoro (Kokoro-82M, hexgrad)          Apache License 2.0
  kokoro-onnx                            MIT License
  Piper (Rhasspy)                        MIT License
  phonemizer                             GNU GPL v3
  eSpeak NG                              GNU GPL v3   https://github.com/espeak-ng/espeak-ng
  ONNX Runtime (Microsoft)               MIT License
Voices carry the licence of the recordings they were trained on.
""" % (VERSION, SOURCE_URL)


def _log(msg):
    try:
        sys.stderr.write(str(msg) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


# ---------------------------------------------------------------- Kokoro
class _Kokoro:
    """One loaded model, kept between requests (loading takes seconds)."""

    def __init__(self):
        self.obj = None
        self.key = None
        self.lock = threading.Lock()

    def _build(self, onnx, voices, threads):
        from kokoro_onnx import Kokoro
        if threads:
            # A capped thread pool can only be set when the session is made.
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = int(threads)
                so.inter_op_num_threads = 1
                sess = ort.InferenceSession(onnx, sess_options=so,
                                            providers=["CPUExecutionProvider"])
                for name in ("from_session", "from_inference_session"):
                    f = getattr(Kokoro, name, None)
                    if callable(f):
                        return f(sess, voices)
                _log("This kokoro-onnx cannot take a capped session; running uncapped.")
            except Exception as e:
                _log("Could not cap threads (%s); running uncapped." % e)
        return Kokoro(onnx, voices)

    def get(self, onnx, voices, threads):
        key = (onnx, voices, int(threads or 0))
        if self.obj is None or self.key != key:
            if not (onnx and os.path.isfile(onnx)):
                raise RuntimeError("Kokoro model not found: %s" % onnx)
            if not (voices and os.path.isfile(voices)):
                raise RuntimeError("Kokoro voices not found: %s" % voices)
            self.obj = self._build(onnx, voices, threads)
            self.key = key
        return self.obj


_KOKORO = _Kokoro()


def bake_kokoro(text, voice, out, lang="en-us", speed=1.0,
                onnx=None, voices=None, threads=None):
    import soundfile as sf
    with _KOKORO.lock:
        k = _KOKORO.get(onnx, voices, threads)
        samples, rate = k.create(text, voice=voice, speed=float(speed or 1.0),
                                 lang=lang or "en-us")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # 16-bit PCM: the stdlib wave module (and every browser) reads it.
    sf.write(out, samples, rate, subtype="PCM_16")


# ---------------------------------------------------------------- Piper
def bake_piper(text, model, out):
    if not (model and os.path.isfile(model)):
        raise RuntimeError("Piper voice not found: %s" % model)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    try:
        from piper import PiperVoice           # piper-tts 1.2+
        import wave
        v = PiperVoice.load(model)
        with wave.open(out, "wb") as w:
            if hasattr(v, "synthesize_wav"):
                v.synthesize_wav(text, w)
            else:
                v.synthesize(text, w)
        return
    except ImportError:
        pass
    # An older piper-tts: its command line, run as a separate process.
    p = subprocess.run([sys.executable, "-m", "piper", "-m", model, "-f", out],
                       input=text.encode("utf-8"), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or b"").decode("utf-8", "ignore")[-400:]
                           or "Piper failed.")


# ---------------------------------------------------------------- phonemes
def phonemize(text, lang="en-us"):
    from kokoro_onnx.tokenizer import Tokenizer
    return Tokenizer().phonemize(text, lang or "en-us")


# ---------------------------------------------------------------- serve
def _handle(req):
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "name": NAME, "version": VERSION}
    if op == "bake":
        eng = req.get("engine") or "kokoro"
        text = (req.get("text") or "").strip()
        if not text:
            raise RuntimeError("No text to bake.")
        if eng == "kokoro":
            bake_kokoro(text, req.get("voice") or "am_michael", req["out"],
                        lang=req.get("lang") or "en-us", speed=req.get("speed") or 1.0,
                        onnx=req.get("kokoro_onnx"), voices=req.get("kokoro_voices"),
                        threads=req.get("threads"))
        elif eng == "piper":
            bake_piper(text, req.get("model"), req["out"])
        else:
            raise RuntimeError("Unknown engine: %s" % eng)
        return {"ok": True}
    if op == "phonemize":
        return {"ok": True, "phonemes": phonemize(req.get("text") or "",
                                                  req.get("lang") or "en-us")}
    raise RuntimeError("Unknown op: %s" % op)


def serve():
    # stdout carries replies and nothing else: anything a library prints
    # goes to stderr instead.
    out = sys.stdout
    sys.stdout = sys.stderr
    try:
        out.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            out.write(json.dumps({"ok": False, "error": "bad request: %s" % e}) + "\n")
            out.flush()
            continue
        if req.get("op") == "quit":
            break
        try:
            rep = _handle(req)
        except Exception as e:
            rep = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        if "id" in req:
            rep["id"] = req["id"]
        out.write(json.dumps(rep) + "\n")
        out.flush()


def _models_paths(models):
    m = models or os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "tts_models")
    return (os.path.join(m, "kokoro", "kokoro-v1.0.onnx"),
            os.path.join(m, "kokoro", "voices-v1.0.bin"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="Blue Baker",
                                 description="Text in, narration out (WAV).")
    ap.add_argument("--serve", action="store_true", help="JSON lines on stdin/stdout")
    ap.add_argument("--about", action="store_true", help="licence and source")
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--say", help="text to speak")
    ap.add_argument("--voice", default="am_michael")
    ap.add_argument("--lang", default="en-us")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", help="WAV file to write")
    ap.add_argument("--models", help="folder holding kokoro/ (and piper/)")
    ap.add_argument("--piper", help="a Piper .onnx voice, instead of Kokoro")
    ap.add_argument("--phonemize", help="print how the text is said")
    a = ap.parse_args(argv)
    if a.about:
        print(ABOUT)
        return 0
    if a.version:
        print("%s %s" % (NAME, VERSION))
        return 0
    if a.serve:
        serve()
        return 0
    if a.phonemize is not None:
        print(phonemize(a.phonemize, a.lang))
        return 0
    if a.say is not None:
        if not a.out:
            ap.error("--say needs --out")
        if a.piper:
            bake_piper(a.say, a.piper, a.out)
        else:
            onnx, voices = _models_paths(a.models)
            bake_kokoro(a.say, a.voice, a.out, a.lang, a.speed, onnx, voices)
        print(a.out)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
