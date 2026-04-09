from rest_framework import serializers


class PredictRequestSerializer(serializers.Serializer):
    sourceMcId = serializers.IntegerField()
    sourceMcTitle = serializers.CharField()
    description = serializers.CharField()