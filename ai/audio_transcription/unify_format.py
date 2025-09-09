from pathlib import Path
from subprocess import call

from loguru import logger

from did_api.ai.audio_transcription.ffmpeg_apply import FfmpegApp


class WavTransform:
    def __init__(self, m4a_path: str, folder_name: str) -> None:
        self.folder_name: Path = Path(folder_name).resolve()
        self.m4a_path: Path = Path(m4a_path).resolve()
        self.wav_path: Path = self.folder_name / self.m4a_path.with_suffix(".wav").name

    async def wav2m4a(self) -> Path | None:
        ffmpeg: FfmpegApp = FfmpegApp()
        ffmpeg_path: str | None = await ffmpeg.ffmpeg_apply()

        logger.info(f'Using ffmpeg from: {ffmpeg_path}')

        try:
            logger.info(f'Conversion completed successfully. The files has been saved here --> {self.wav_path}')
            call([ffmpeg_path, '-y', '-i', self.m4a_path, self.wav_path])
            return self.wav_path
        except Exception as exc:
            logger.error(f'Error during conversion: {exc!r}')
            return None
