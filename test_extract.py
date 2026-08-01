import asyncio
from pathlib import Path
import traceback
from vaaniflow.audio.extractor import AudioExtractor

async def test():
    try:
        await AudioExtractor().extract(Path('README.md'))
    except Exception as e:
        traceback.print_exc()

asyncio.run(test())
