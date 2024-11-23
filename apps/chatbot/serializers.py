from rest_framework import serializers
from apps.chatbot.models import Document, PermissionRequest
from apps.user.serializers import UserSerializer
from apps.user.models import BaseUser
from rna_utils import debug_print


class PermissionRequestSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = PermissionRequest
        fields = ["id", "user", "request_status", "created_at", "updated_at", "status"]
        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
        ]


class DocumentSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "user",
            "document",
            "name",
            "type",
            "vectorstore_ids",
            "embeddings_status",
            "created_at",
            "updated_at",
            "status",
        ]
        read_only_fields = [
            "id",
            "vectorstore_ids",
            "created_at",
            "updated_at",
            "status",
        ]
