from rest_framework import serializers


class PredictRequestSerializer(serializers.Serializer):
    itemId = serializers.IntegerField()
    sourceMcId = serializers.IntegerField()
    sourceMcTitle = serializers.CharField()
    description = serializers.CharField()