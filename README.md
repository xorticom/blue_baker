# Blue Baker

**Text in, narration out.** Blue Baker turns text into a WAV file with a
neural voice. It is the part of [XortiCom BLUE](https://xorticom.com) that
speaks, kept as a program of its own so that it can be free software.

```
Blue Baker --say "Hello there." --voice am_michael --out hello.wav --models <folder>
Blue Baker --phonemize "produce" --lang en-us
Blue Baker --about
Blue Baker --serve          (one JSON request per line on stdin, one reply per line on stdout)
```

`--models` is the folder that holds `kokoro/kokoro-v1.0.onnx` and
`kokoro/voices-v1.0.bin` (from the kokoro-onnx project).

## Licence

Blue Baker is free software under the **GNU General Public License version 3
or later** (see `LICENSE`). XortiCom BLUE itself is a separate program that
runs Blue Baker and reads the WAV files it writes.

Blue Baker is built with:

| Part | Licence |
|---|---|
| Kokoro (Kokoro-82M, hexgrad) | Apache License 2.0 |
| kokoro-onnx | MIT |
| Piper (Rhasspy) | MIT |
| phonemizer | GNU GPL v3 |
| eSpeak NG — https://github.com/espeak-ng/espeak-ng | GNU GPL v3 |
| ONNX Runtime (Microsoft) | MIT |

Each voice carries the licence of the recordings it was trained on. BLUE
offers only voices cleared for business use: the Kokoro voices (Apache 2.0)
and Piper's Alba (CC BY 4.0).

## Building

Windows, Python 3.11 or later:

```
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --clean "Blue Baker.spec"
```

The program is written to `dist\Blue Baker\`.
