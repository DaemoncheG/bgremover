import main


def test_parse_bg_color_known_values():
    assert main.parse_bg_color("black") == (0, 0, 0)
    assert main.parse_bg_color("white") == (255, 255, 255)
    assert main.parse_bg_color("green") == (0, 255, 0)
    assert main.parse_bg_color("blue") == (255, 0, 0)


def test_parse_bg_color_defaults_to_black():
    assert main.parse_bg_color("unknown") == (0, 0, 0)


def test_force_ext_replaces_extension():
    assert main.force_ext("input.jpg", "png") == "input.png"
    assert main.force_ext("input.jpg", ".webm") == "input.webm"


def test_is_image_and_video_detection():
    assert main.is_image("photo.png")
    assert main.is_image("photo.jpeg")
    assert not main.is_image("video.mp4")
    assert main.is_video("clip.mp4")
    assert main.is_video("clip.webm")
    assert not main.is_video("image.png")
