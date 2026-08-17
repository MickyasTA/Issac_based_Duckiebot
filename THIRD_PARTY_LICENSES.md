# Third-party licenses

This file reproduces the license texts of third-party material that may be vendored under
`assets/`. See `NOTICE` for the clean-room position and for upstream software that is imported
but never vendored.

Every binary asset under `assets/` must have an entry in `assets/MANIFEST.yaml` naming either the
generator script that produced it or the third-party source below. `scripts/check_clean_room.py`
fails the build otherwise.

---

## AprilTag marker images (`apriltag-imgs`)

Source: <https://github.com/AprilRobotics/apriltag-imgs>
Used for: 36h11 tag PNGs upscaled onto sign cards as visual distractors.

```
Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed or
implied, of The Regents of The University of Michigan.
```

---

## HDRI environment maps (CC0 1.0 Universal)

Source: CC0 asset libraries only, for example <https://polyhaven.com>.
Used for: dome-light background imagery (six 2K files maximum, per the VRAM budget).
The files themselves are gitignored; `assets/hdri/SOURCES.yaml` records URL and sha256 per file.

CC0 1.0 Universal is a public domain dedication: the rights holder has waived all copyright and
related rights worldwide. No attribution is legally required; it is given anyway in
`assets/hdri/SOURCES.yaml`. Full text: <https://creativecommons.org/publicdomain/zero/1.0/legalcode>

---

## Upstream Python packages

`torch`, `numpy`, `onnx`, `onnxruntime`, `opencv-python-headless`, `pillow`, `pyyaml`,
`tensorboard`, `mujoco` and `usd-core` are installed from PyPI and are not vendored. Their
licenses (BSD-3-Clause, Apache-2.0, MIT, and similar permissive terms) ship inside their
respective distributions.

`isaacsim` and `isaaclab` are NOT dependencies of this package. They are installed separately by
the user under NVIDIA license terms and imported behind guarded imports.
