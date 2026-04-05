import json
from channels.generic.websocket import AsyncWebsocketConsumer


class PredictConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.request_id = self.scope["url_route"]["kwargs"]["request_id"]
        self.room_group_name = f"predict_{self.request_id}"

        print(f"[WS CONNECT] request_id={self.request_id} room={self.room_group_name}")

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name,
        )
        await self.accept()

    async def disconnect(self, close_code):
        print(f"[WS DISCONNECT] request_id={self.request_id}")

        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name,
        )

    async def send_draft(self, event):
        print(f"[WS SEND_DRAFT] event={event}")

        await self.send(text_data=json.dumps({
            "event": "draft_ready",
            "request_id": event["request_id"],
            "draft": event["draft"],
        }, ensure_ascii=False))

    async def send_error(self, event):
        print(f"[WS SEND_ERROR] event={event}")

        await self.send(text_data=json.dumps({
            "event": "draft_error",
            "request_id": event["request_id"],
            "mcId": event["mcId"],
            "error": event["error"],
        }, ensure_ascii=False))

    async def send_done(self, event):
        print(f"[WS SEND_DONE] event={event}")

        await self.send(text_data=json.dumps({
            "event": "predict_done",
            "request_id": event["request_id"],
        }, ensure_ascii=False))