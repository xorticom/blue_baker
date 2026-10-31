# -*- mode: python ; coding: utf-8 -*-
# Blue Baker -- a console program, started hidden by BLUE (no window appears).
from PyInstaller.utils.hooks import collect_all

datas = [('LICENSE', '.'), ('README.md', '.')]
binaries = []
hiddenimports = ['onnxruntime']
for pkg in ('kokoro_onnx', 'onnxruntime', 'espeakng_loader', 'phonemizer',
            'segments', 'csvw', 'language_tags', 'soundfile', 'piper'):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

a = Analysis(['blue_baker.py'], pathex=[], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, hookspath=[], runtime_hooks=[],
             excludes=['tkinter'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Blue Baker',
          debug=False, strip=False, upx=True, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name='Blue Baker')
