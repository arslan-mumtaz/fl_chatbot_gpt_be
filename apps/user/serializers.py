from rest_framework import serializers
from rna_utils import debug_print
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from apps.user.models import BaseUser


class UserSerializer(serializers.ModelSerializer):

    class Meta:
        model = BaseUser
        fields = [
            "id",
            "email",
            "first_name",
            "last_name",
            "full_name",
            "password",
            "is_active",
            "is_staff",
            "is_superuser",
            "is_chatbot_user",
            "date_joined",
            "last_login",
            "created_at",
            "updated_at",
            "status",
            # "chatbot_requests",
        ]
        read_only_fields = [
            "id",
            "full_name",
            "created_at",
            "updated_at",
            "is_active",
            "is_staff",
            "is_superuser",
            "is_chatbot_user",
            "date_joined",
            "last_login",
            # "chatbot_requests",
        ]
        extra_kwargs = {"password": {"write_only": True}}

    def create(self, validated_data):
        user = BaseUser.objects.create(**validated_data)
        user.set_password(validated_data["password"])
        user.save()
        return user

    def update(self, instance, validated_data):
        for attr, value in validated_data.items():
            if attr == "password":
                instance.set_password(value)
            else:
                setattr(instance, attr, value)
        instance.save()
        return instance


class LoginSerializer(TokenObtainPairSerializer):

    @classmethod
    def get_token(cls, user):
        token = super(TokenObtainPairSerializer, cls).get_token(user)

        token["username"] = user.email
        token["email"] = user.email
        token["is_staff"] = user.is_staff
        token["is_superuser"] = user.is_superuser
        token["is_chatbot_user"] = user.is_chatbot_user

        return token
