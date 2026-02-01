import subprocess
import sys

import pytest
from PIL import Image, ImageDraw


@pytest.mark.slow
def test_image_processing_creates_output(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([32, 32, 96, 96], fill=(0, 0, 0))
    img.save(input_path)

    result = subprocess.run(
        [sys.executable, "main.py", str(input_path), str(output_path)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    out_img = Image.open(output_path)
    assert out_img.mode == "RGBA"
