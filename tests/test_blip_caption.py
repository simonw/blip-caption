import pytest
from click.testing import CliRunner
from unittest.mock import ANY
import json
from PIL import UnidentifiedImageError
from blip_caption import cli

# Mocked data for transformers.pipeline
CAPTION_MOCK = [{"generated_text": "This is a test caption"}]
ERROR_MOCK = "Unidentified image error"


PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc\xfa\x0f\x00\x01"
    b"\x05\x01\x02\xcf\xa0.\xcd\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def image_paths(tmpdir):
    image = tmpdir / "test_image.jpg"
    image.write(PNG_1x1, mode="wb")
    image2 = tmpdir / "test_image2.jpg"
    image2.write(PNG_1x1, mode="wb")
    return (str(image), str(image2))


def test_single_image_caption(image_paths, mocker):
    mock_pipeline = mocker.patch("blip_caption.pipeline")

    mock_captioner = mocker.Mock()
    mock_captioner.return_value = [{"generated_text": "This is a test caption"}]
    mock_pipeline.return_value = mock_captioner

    runner = CliRunner()
    result = runner.invoke(cli, [image_paths[0]], catch_exceptions=False)
    assert result.exit_code == 0
    assert "This is a test caption" in result.output


def test_multiple_image_captions(image_paths, mocker):
    mock_pipeline = mocker.patch("blip_caption.pipeline")

    mock_captioner = mocker.Mock()
    mock_captioner.side_effect = [
        [{"generated_text": "Caption 1"}],
        [{"generated_text": "Caption 2"}],
    ]
    mock_pipeline.return_value = mock_captioner

    runner = CliRunner()

    result = runner.invoke(cli, image_paths)
    assert result.exit_code == 0
    assert "test_image.jpg" in result.output
    assert "test_image2.jpg" in result.output
    assert "Caption 1" in result.output
    assert "Caption 2" in result.output


def test_error_image(mocker, image_paths):
    mock_pipeline = mocker.patch("blip_caption.pipeline")

    # Create a mock captioner function that raises an exception
    mock_captioner = mocker.Mock()
    mock_captioner.side_effect = UnidentifiedImageError("Error bad image")
    mock_pipeline.return_value = mock_captioner

    runner = CliRunner()
    result = runner.invoke(cli, [image_paths[0]])
    assert result.exit_code == 0
    assert result.output == "Error: Error bad image\n"


def test_json_output(mocker, image_paths):
    mock_pipeline = mocker.patch("blip_caption.pipeline")

    mock_captioner = mocker.Mock()
    mock_captioner.side_effect = [
        [{"generated_text": "Caption 1"}],
        [{"generated_text": "Caption 2"}],
    ]
    mock_pipeline.return_value = mock_captioner

    runner = CliRunner()
    result = runner.invoke(cli, list(image_paths) + ["--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == [
        {
            "path": ANY,
            "caption": "Caption 1",
        },
        {
            "path": ANY,
            "caption": "Caption 2",
        },
    ]
