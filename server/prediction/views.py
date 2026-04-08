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