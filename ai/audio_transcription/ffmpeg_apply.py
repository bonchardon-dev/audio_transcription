from os import environ, pathsep
from pathlib import Path
from os import path
from shutil import which
from sys import platform, exit

from loguru import logger

from did_api.ai.audio_transcription.consts import FFMPEG_PATH


class FfmpegApp:
    @staticmethod
    async def ffmpeg_apply() -> str | None:
        ffmpeg_path: str | None = which('ffmpeg')

        if not ffmpeg_path and platform.startswith("win"):
            if path.isfile(FFMPEG_PATH):
                ffmpeg_path: str | None = FFMPEG_PATH
                environ["PATH"] += pathsep + str(Path(FFMPEG_PATH).parent)

        if not ffmpeg_path:
            logger.error('ffmpeg not found in PATH or fallback location.')
            exit(1)
        return ffmpeg_path
