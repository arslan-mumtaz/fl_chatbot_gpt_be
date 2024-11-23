from django.urls import path
from rest_framework import routers
from django.urls import include
from apps.user.views import (
    LoginApiView,
    LogoutApiView,
    TokenRefreshApiView,
    UserViewSet,
)

router = routers.DefaultRouter()


router.register(r"users", UserViewSet)

urlpatterns = [
    path("", include(router.urls)),
    # ? AUTHENTICATION
    path("login/", LoginApiView.as_view(), name="token_obtain_pair"),
    path("refresh/", TokenRefreshApiView.as_view(), name="token_refresh"),
    path("logout/", LogoutApiView.as_view(), name="token_blacklist"),
]


urlpatterns += router.urls
