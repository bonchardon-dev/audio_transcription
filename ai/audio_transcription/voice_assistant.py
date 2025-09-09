from pathlib import Path
from os import getenv
from typing import List

from loguru import logger
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydub import AudioSegment

load_dotenv()


class AudioTranscription:
    def __init__(self, audio_path: str, model_size: str = "whisper-1") -> None:
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=getenv("OPENAI_API_KEY"))
        self.audio_path: Path = Path(audio_path).resolve()
        self.model_size: str = model_size
        self.diarized_dir: Path = Path("did_api/data/diarized_segments").resolve()
        self.max_mb: float = 25.0  # Whisper API limit

    def split_audio_by_duration(self, chunk_length_ms: int = 4 * 60 * 1000) -> List[AudioSegment]:
        """
        Splits audio into chunks (default: 4 minutes) that are likely under the Whisper 25MB limit.
        """
        audio = AudioSegment.from_file(self.audio_path)
        chunks = []
        for start in range(0, len(audio), chunk_length_ms):
            end = min(start + chunk_length_ms, len(audio))
            chunks.append(audio[start:end])
        logger.info(f"Split audio into {len(chunks)} chunks (~{chunk_length_ms / 1000}s each).")
        return chunks

    async def trans_audio(self, file_path: Path) -> str | None:
        try:
            with open(file_path, "rb") as f:
                transcription: str = await self.client.audio.transcriptions.create(
                    model=self.model_size,
                    file=f,
                    response_format="text",
                    prompt=(
                        "Це аудіо-файл розмови кількох людей українською (можливий суржик). "
                        "Будь ласка, транскрибуй текст і познач кожного мовця у форматі Speaker_0:, Speaker_1:, тощо. "
                        "Пиши лише репліки, без додаткових описів чи перекладів."
                    ),
                    language="uk",
                )
                return transcription
        except Exception as e:
            logger.error(f"Error transcribing {file_path.name}: {e}")
            return None

    async def transcribe_chunk(self, chunk: AudioSegment, index: int) -> str | None:
        temp_path: Path = Path(f"temp_chunk_{index}.wav")

        # Export chunk to temp WAV file
        chunk.export(temp_path, format="wav")

        # Check size in MB
        size_mb = temp_path.stat().st_size / (1024 * 1024)
        logger.info(f"[Chunk {index}] Size: {size_mb:.2f} MB")

        if size_mb > self.max_mb:
            logger.warning(f"[Chunk {index}] Skipped: exceeds 25 MB limit")
            return None

        # Transcribe
        transcription = await self.trans_audio(temp_path)
        logger.debug(f"[Chunk {index}] Transcription: {transcription}")

        # Clean up
        temp_path.unlink(missing_ok=True)

        return transcription

    async def transcribe_all(self) -> str | None:
        chunks = self.split_audio_by_duration()

        full_transcript_lines: List[str] = []

        for index, chunk in enumerate(chunks, start=1):
            logger.info(f"Transcribing chunk {index}/{len(chunks)}...")
            transcription = await self.transcribe_chunk(chunk, index)
            if transcription:
                full_transcript_lines.append(transcription.strip())

        final_transcript: str = "\n".join(full_transcript_lines)

        # Save transcript
        txt_path = self.audio_path.with_suffix(".txt")
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(final_transcript)

        logger.success(f"Transcript saved to: {txt_path}")
        return final_transcript
