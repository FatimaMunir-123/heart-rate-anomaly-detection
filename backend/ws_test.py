import asyncio
import json
import websockets


async def test_websocket():
    uri = "ws://127.0.0.1:8000/ws/heart-rate"

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")

        for _ in range(10):
            message = await websocket.recv()
            data = json.loads(message)
            print(data)


if __name__ == "__main__":
    asyncio.run(test_websocket())
