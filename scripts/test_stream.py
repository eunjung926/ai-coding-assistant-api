"""Local streaming endpoint benchmark script."""

import asyncio
import json
import time

import httpx

API_URL = "http://localhost:5059/autocompleteNLtoCode/vul/stream"
TEST_INPUT = "Generate an encrypt function for sensitive user data using AES.new"


async def test_streaming_speed() -> None:
    start = time.time()
    first_token_time = None
    last_token_time = None
    total_chars = 0
    token_count = 0
    generated_code = ""
    ttft = None

    print("=" * 50)
    print("Streaming request sent")
    print("=" * 50)

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "GET", API_URL, params={"input": TEST_INPUT}
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])

                if "token" in data:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        ttft = first_token_time - start
                        print(f"\nTTFT: {ttft:.3f}s\n")
                        print("--- Generated code ---")

                    last_token_time = now
                    token_count += 1
                    total_chars += len(data["token"])
                    generated_code += data["token"]
                    print(data["token"], end="", flush=True)

                if "done" in data:
                    end = time.time()
                    generation_time = (
                        (last_token_time - first_token_time)
                        if (last_token_time and first_token_time)
                        else 0
                    )
                    total_time = end - start
                    chars_per_sec = (
                        total_chars / generation_time if generation_time > 0 else 0
                    )

                    print("\n\n" + "=" * 50)
                    print("Benchmark results")
                    print("=" * 50)
                    print(f"TTFT              : {ttft:.3f}s")
                    print(f"Total time        : {total_time:.3f}s")
                    print(f"Generation time   : {generation_time:.3f}s")
                    print(f"Chunks received   : {token_count}")
                    print(f"Characters        : {total_chars}")
                    print(f"Chars/sec         : {chars_per_sec:.1f}")

                    if "stats" in data:
                        stats = data["stats"]
                        print(f"Tokens            : {stats['total_tokens']}")
                        print(f"Tokens/sec        : {stats['tokens_per_sec']:.2f}")
                        print(f"Server total time : {stats['total_time']:.3f}s")
                    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_streaming_speed())
