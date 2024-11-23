from rest_framework import viewsets
from rest_framework.request import Request
from apps.user.serializers import LoginSerializer, UserSerializer
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.decorators import action
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenBlacklistView,
    TokenRefreshView,
)
from custom.permissions.permissions import IsSuperAdmin
from .models import BaseUser


# ---------------------------------------------------------------------------- #
#                                     AUTH                                     #
# ---------------------------------------------------------------------------- #
class LoginApiView(TokenObtainPairView):
    permission_classes = [AllowAny]
    serializer_class = LoginSerializer

    def post(self, request, *args, **kwargs):
        return super().post(request, *args, **kwargs)


class LogoutApiView(TokenBlacklistView):
    permission_classes = [AllowAny]

    def post(self, request: Request, *args, **kwargs) -> Response:
        return super().post(request, *args, **kwargs)


class TokenRefreshApiView(TokenRefreshView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs) -> Response:
        return super().post(request, *args, **kwargs)


# ---------------------------------------------------------------------------- #
#                                     USER                                     #
# ---------------------------------------------------------------------------- #


class UserViewSet(viewsets.ModelViewSet):
    queryset = BaseUser.objects.all()
    serializer_class = UserSerializer
    permission_classes = (permissions.IsAuthenticated,)
    filterset_fields = (
        "email",
        "is_active",
        "is_staff",
        "is_superuser",
        "is_chatbot_user",
    )
    USER_NOT_FOUND = {"error": "User not found"}
    USER_STATUSES = ["active", "inactive", "deleted"]

    def get_permissions(self):
        if self.action == "create":
            return (AllowAny(),)
        elif self.action in [
            "restore",
            "destroy",
            "allow_chatbot_access",
            "revoke_chatbot_access",
        ]:
            return (IsSuperAdmin(),)
        return super().get_permissions()

    def get_object(self, id: int | None = None):
        try:
            return BaseUser.objects.get(pk=id or self.kwargs.get("pk"))
        except BaseUser.DoesNotExist:
            return None

    # ------------------------------------ API ----------------------------------- #

    def list(self, request, *args, **kwargs):
        self.queryset = self.queryset.filter(status="active")
        return super().list(request, *args, **kwargs)

    def create(self, request, *args, **kwargs):
        request.data["is_staff"] = True
        request.data["is_superuser"] = False
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        if "email" in serializer.errors:
            return Response(
                {"error": "User with this email already exists"}, status=400
            )
        return Response(serializer.errors, status=400)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object(id=kwargs.get("pk"))
        if not instance:
            return Response(self.USER_NOT_FOUND, status=404)
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object(id=kwargs.get("pk"))
        if not instance:
            return Response({"error": "User not found"}, status=404)
        instance.deactivate()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def restore(self, request, *args, **kwargs):
        instance = self.get_object(id=kwargs.get("pk"))
        if not instance:
            return Response(self.USER_NOT_FOUND, status=404)
        instance.activate()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def allow_chatbot_access(self, request, *args, **kwargs):
        instance = self.get_object(id=kwargs.get("pk"))
        if not instance:
            return Response(self.USER_NOT_FOUND, status=404)
        instance.allow_chatbot_access()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def revoke_chatbot_access(self, request, *args, **kwargs):
        instance = self.get_object(id=kwargs.get("pk"))
        if not instance:
            return Response(self.USER_NOT_FOUND, status=404)
        instance.revoke_chatbot_access()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    @action(detail=False, methods=["post"], url_path="bulk-delete")
    def bulk_delete(self, request):
        user_ids = request.data.get("users", [])
        users = BaseUser.objects.filter(id__in=user_ids)
        for user in users:
            user.delete()
        return Response({"status": "deleted", "message": "Users deleted!"})
