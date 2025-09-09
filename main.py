from asyncio import run
from ai.audio_transcription.audio_preprocess import AudioPreprocessor

async def main() -> None:
    await AudioPreprocessor(m4a_path='', folder_name='response').run()


if __name__ == '__main__':
    run(main())
