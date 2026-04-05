import asyncio
from channels.layers import get_channel_layer

from .llm_service import generate_draft_with_llm


async def generate_drafts(request_id, item_data, split_categories):
    channel_layer = get_channel_layer()

    print(f"[TASK START] request_id={request_id}")
    print(f"[TASK SPLIT CATEGORIES] {split_categories}")


    for category in split_categories:
        try:
            print(f"[TASK GENERATE] request_id={request_id} category={category}")

            text = await asyncio.to_thread(
                generate_draft_with_llm,
                item_data,
                category,
            )

            draft = {
                "mcId": category["mcId"],
                "mcTitle": category["mcTitle"],
                "text": text,
            }

            print(f"[TASK SEND] room=predict_{request_id} draft={draft}")

            await channel_layer.group_send(
                f"predict_{request_id}",
                {
                    "type": "send_draft",
                    "request_id": request_id,
                    "draft": draft,
                },
            )

        except Exception as e:
            print(f"[TASK ERROR] request_id={request_id} category={category} error={e}")

            await channel_layer.group_send(
                f"predict_{request_id}",
                {
                    "type": "send_error",
                    "request_id": request_id,
                    "mcId": category["mcId"],
                    "error": str(e),
                },
            )

    print(f"[TASK DONE] request_id={request_id}")

    await channel_layer.group_send(
        f"predict_{request_id}",
        {
            "type": "send_done",
            "request_id": request_id,
        },
    )