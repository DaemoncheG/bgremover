import shutil
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_video_processing_creates_webm_output(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not available")

    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.webm"

    create_video = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:d=1",
            "-pix_fmt",
            "yuv420p",
            str(input_path),
        ],
        text=True,
        capture_output=True,
    )
    assert create_video.returncode == 0, create_video.stderr

    result = subprocess.run(
        [sys.executable, "main.py", str(input_path), str(output_path)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ],
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr
