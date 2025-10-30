import os
from redis import asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class MessageBus:
    def __init__(self):
        self.redis = None

    async def init(self):
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)

    async def publish(self, channel: str, message: str):
        await self.redis.publish(channel, message)

    async def subscribe(self, channel: str):
        pub = self.redis.pubsub()
        await pub.subscribe(channel)
        return pub