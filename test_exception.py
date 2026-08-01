import asyncio
from pathlib import Path

async def test():
    try:
        await asyncio.create_subprocess_exec('does_not_exist', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError:
        print("Caught FileNotFoundError")
    except Exception as e:
        print(f"Caught Exception: type={type(e)}, str='{e}'")

asyncio.run(test())
