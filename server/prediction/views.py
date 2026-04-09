import threading
import uuid

from asgiref.sync import async_to_sync
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import PredictRequestSerializer
from .tasks import generate_drafts
from .model_service import run_prediction
from .storage import REQUEST_STORE
from .llm_service import generate_draft_with_llm



class PredictView(APIView):
    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        request_id = str(uuid.uuid4())

        prediction = run_prediction(data)


        split_categories = prediction["splitCategories"]

        REQUEST_STORE[request_id] = {
            "item_data": data,
            "split_categories": split_categories,
        }

        detected_mc_ids = prediction["detectedMcIds"]
        should_split = prediction["shouldSplit"]
        split_categories = prediction["splitCategories"]

        print(f"[PREDICT] request_id={request_id}")
        print(f"[PREDICT] detected_mc_ids={detected_mc_ids}")
        print(f"[PREDICT] should_split={should_split}")
        print(f"[PREDICT] split_categories={split_categories}")

        return Response(
            {
                "request_id": request_id,
                "detectedMcIds": detected_mc_ids,
                "shouldSplit": should_split,
                "drafts": [],
            },
            status=status.HTTP_202_ACCEPTED,
        )


class PredictSyncView(APIView):
    """Predict and return all drafts in a single HTTP response (no websocket streaming)."""

    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        prediction = run_prediction(data)
        split_categories = prediction["splitCategories"]

        drafts = []
        for category in split_categories:
            try:
                text = generate_draft_with_llm(data, category)
            except Exception as exc:
                # Keep endpoint resilient even if LLM provider fails for one draft.
                text = (
                    f"Черновик для категории '{category.get('mcTitle', '')}'. "
                    f"Описание: {str(data.get('description', ''))[:160]}"
                )
                print(f"[PREDICT_SYNC] draft generation error mcId={category.get('mcId')} error={exc}")

            drafts.append(
                {
                    "mcId": category["mcId"],
                    "mcTitle": category["mcTitle"],
                    "text": text,
                }
            )

        return Response(
            {
                "detectedMcIds": prediction["detectedMcIds"],
                "shouldSplit": prediction["shouldSplit"],
                "drafts": drafts,
            },
            status=status.HTTP_200_OK,
        )