from django.urls import path
from rest_framework import routers
from django.urls import include
from apps.chatbot.views import ChatBotApiView, DocumentViewSet, PermissionRequestViewSet

router = routers.DefaultRouter()


router.register(r"permission-requests", PermissionRequestViewSet)
router.register(r"documents", DocumentViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("chatbot/", ChatBotApiView.as_view()),
]


urlpatterns += router.urls
