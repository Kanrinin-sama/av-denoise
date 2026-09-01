#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["vapoursynth", "vapoursynth-bestsource"]
# ///
import argparse
import os
import pathlib
import sys

os.environ["AV_DENOISE_COMPILATION_CACHE"] = "off"

import vapoursynth as vs  # noqa: E402

core = vs.core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="target/release/libav_denoise_vs.so")
    ap.add_argument("--input", required=True)
    ap.add_argument("--algo", choices=["nl4d", "nlmeans"], required=True)
    args = ap.parse_args()

    core.std.LoadPlugin(str(pathlib.Path(args.plugin).resolve()))

    src = core.bs.VideoSource(args.input)

    if args.algo == "nl4d":
        out = core.avd.NL4D(src)
    else:
        out = core.avd.NLMeans(src)

    out.output(sys.stdout.buffer, y4m=True, prefetch=0)


if __name__ == "__main__":
    main()
