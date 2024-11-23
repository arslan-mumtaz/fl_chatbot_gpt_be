from rest_framework.permissions import BasePermission


class IsSuperAdmin(BasePermission):
    """
    Allows access only to super admins.
    """

    def has_permission(self, request, view):
        return request.user.is_superuser


class IsChatbotUser(BasePermission):
    """
    Allows access only to chatbot users.
    """

    def has_permission(self, request, view):
        return request.user.is_chatbot_user or request.user.is_superuser
