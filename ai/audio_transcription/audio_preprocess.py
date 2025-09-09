from loguru import logger

from pyannote.audio import Pipeline

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from did_api.ai.audio_transcription.unify_format import WavTransform
from did_api.ai.audio_transcription.consts import PYANNOTE_MODEL

from config import settings


class AudioPreprocessor(WavTransform):
    def __init__(self, m4a_path: str, folder_name: str):
        super().__init__(m4a_path, folder_name)
        self.pipeline: Pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=settings.huggingface_token)
        self.m4a_path: str = m4a_path

    async def load_audio(self) -> AudioSegment | None:
        try:
            audio: AudioSegment = AudioSegment.from_file(self.wav_path)
            logger.success(f"Loaded audio file: {self.wav_path}")
            return audio
        except Exception as exc:
            logger.error(f"Failed to load audio: {exc!r}")
            return None

    @staticmethod
    async def merge_close_ranges(ranges, gap=250) -> list | None:
        if not ranges:
            return []
        merged: list = [ranges[0]]
        for current in ranges[1:]:
            prev = merged[-1]
            if current[0] - prev[1] <= gap:
                merged[-1] = [prev[0], max(prev[1], current[1])]
            else:
                merged.append(current)
        return merged

    async def delete_silent_part(self, audio_data: AudioSegment) -> AudioSegment | None:
        silence_thresh: float = audio_data.dBFS - 14
        min_silence_len: int = 400

        if not (non_silent_ranges := detect_nonsilent(
            audio_data,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )):
            logger.warning("Audio is completely silent")
            return None

        if non_silent_ranges[0][0] > 300:
            non_silent_ranges.insert(0, [0, non_silent_ranges[0][0]])

        non_silent_ranges: list[list] | None = await self.merge_close_ranges(non_silent_ranges)
        logger.debug(f"Detected non-silent ranges: {non_silent_ranges}")

        cleaned_audio: AudioSegment = AudioSegment.empty()
        for start, end in non_silent_ranges:
            cleaned_audio += audio_data[start:end]

        logger.success("Silences removed, cleaned audio generated.")
        return cleaned_audio

    # async def diarization(self, original_audio: AudioSegment) -> list[str] | None:
    #     diarization_result: Pipeline = self.pipeline(self.wav_path)
    #
    #     ordered_segments: list = []
    #     for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    #         start_ms: int = int(turn.start * 1000)
    #         end_ms: int = int(turn.end * 1000)
    #         segment: AudioSegment = original_audio[start_ms:end_ms]
    #         ordered_segments.append((start_ms, speaker, segment))
    #         logger.debug(f"Added segment for {speaker}: {start_ms}ms - {end_ms}ms")
    #
    #     ordered_segments.sort(key=lambda x: x[0])  # Sort by start time
    #
    #     output_dir: Path = Path("did_api/data/diarized_segments")
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #
    #     for i, (start_ms, speaker, segment) in enumerate(ordered_segments):
    #         filename: Path = output_dir / f"{i:04d}_{speaker}.wav"
    #         segment.export(filename, format="wav")
    #         logger.success(f"Exported: {filename}")
    #
    #     return ordered_segments

    async def diarization(self, original_audio: AudioSegment) -> list[str] | None:
        diarization_result: Pipeline = self.pipeline(self.wav_path)

        ordered_segments: list = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            start_ms: int = int(turn.start * 1000)
            end_ms: int = int(turn.end * 1000)
            segment: AudioSegment = original_audio[start_ms:end_ms]
            ordered_segments.append((start_ms, speaker, segment))
            logger.debug(f"Added segment for {speaker}: {start_ms}ms - {end_ms}ms")

        ordered_segments.sort(key=lambda x: x[0])
        return ordered_segments

    # async def run(self) -> None:
    #     audio_data: AudioSegment | None = await self.load_audio()
    #     cleaned_audio: AudioSegment | None = await self.delete_silent_part(audio_data)
    #     await self.diarization(cleaned_audio)

    async def run(self) -> list | None:
        # Step 1: Convert M4A to WAV
        wav_path = await self.wav2m4a()
        if not wav_path:
            logger.error("WAV conversion failed. Aborting preprocessing.")
            return None

        # Step 2: Load the converted WAV audio
        audio_data: AudioSegment | None = await self.load_audio()
        if not audio_data:
            logger.error("Audio loading failed. Aborting preprocessing.")
            return None

        # Step 3: Remove silence
        cleaned_audio: AudioSegment | None = await self.delete_silent_part(audio_data)
        if not cleaned_audio:
            logger.error("Silence removal failed. Aborting preprocessing.")
            return None

        # Step 4: Diarization (no saving segments)
        diarized_segments: list | None = await self.diarization(cleaned_audio)
        return diarized_segments
