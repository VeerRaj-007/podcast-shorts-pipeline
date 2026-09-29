from dataclasses import dataclass


@dataclass
class EditOptions:
    """
    User-controlled editing preferences.

    Important:
    Important-moment detection is always enabled.

    Everything else is optional.
    """

    captions_enabled: bool = False

    caption_style: str = "dynamic"

    camera_enabled: bool = False

    camera_style: str = "auto"


VALID_CAPTION_STYLES = {
    "dynamic",
    "clean",
    "minimal",
}


VALID_CAMERA_STYLES = {
    "auto",
    "podcast",
    "streamer",
}


def validate_options(options: EditOptions) -> EditOptions:

    if (
        options.caption_style
        not in VALID_CAPTION_STYLES
    ):
        raise ValueError(
            f"Invalid caption style: "
            f"{options.caption_style}"
        )

    if (
        options.camera_style
        not in VALID_CAMERA_STYLES
    ):
        raise ValueError(
            f"Invalid camera style: "
            f"{options.camera_style}"
        )

    return options