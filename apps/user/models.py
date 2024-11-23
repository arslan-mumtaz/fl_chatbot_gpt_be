from typing import Any
from django.db import models

from django.contrib.auth.models import AbstractUser, UserManager
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from django.db import models
from core.models import BaseModel

DEFAULT_TOKEN_STOCK = 1000


class CustomUserManager(UserManager):
    """
    Custom user manager where email is the unique identifier, inherited from UserManager provided by auth
    """

    def create_superuser(
        self,
        email: str,
        password: str | None,
        **extra_fields: Any,
    ) -> Any:
        username = email
        return super().create_superuser(username, email, password, **extra_fields)


class BaseUser(AbstractUser, BaseModel):
    """
    Custom user model where email is the unique identifier, inhertied from abstract user provided by auth
    """

    username = models.CharField(_("username"), max_length=150, blank=True, null=True)
    email = models.EmailField(_("email address"), unique=True)
    first_name = models.CharField(_("first name"), max_length=30, blank=True)
    last_name = models.CharField(_("last name"), max_length=30, blank=True)
    password = models.CharField(_("password"), max_length=128, blank=True)

    token_stock = models.IntegerField(_("token stock"), default=DEFAULT_TOKEN_STOCK)
    # The total amount of tokens used by the user for chatbot api
    tokens_used = models.IntegerField(_("tokens used"), default=0)

    is_superuser = models.BooleanField(_("superuser"), default=False)
    is_staff = models.BooleanField(_("staff status"), default=True)
    is_chatbot_user = models.BooleanField(_("chatbot user"), default=False)

    date_joined = models.DateTimeField(_("date joined"), auto_now_add=True)
    last_login = models.DateTimeField(_("last login"), blank=True, null=True)

    is_active = models.BooleanField(("active"), default=True)

    objects = CustomUserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    class Meta:
        app_label = "user"

    def __str__(self):
        return self.email

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @classmethod
    def get_user_by_email(cls, email: str):
        return cls.objects.filter(email=email).first()

    def get_last_chat_date(self):
        return self.chat_history_set.last().created_at  # type: ignore

    def activate(self, *args, **kwargs):
        self.is_active = True
        return super().activate(*args, **kwargs)

    def deactivate(self, *args, **kwargs):
        self.is_active = False
        return super().deactivate(*args, **kwargs)

    def delete(self, *args, **kwargs):
        self.is_active = False
        return super().delete(*args, **kwargs)

    def allow_chatbot_access(self, *args, **kwargs):
        self.is_chatbot_user = True
        self.save()

    def revoke_chatbot_access(self, *args, **kwargs):
        self.is_chatbot_user = False
        self.save()

    def reset_token_stock(self):
        self.token_stock = DEFAULT_TOKEN_STOCK
        self.save()

    def use_tokens(self, tokens: int):
        last_chat_date = self.get_last_chat_date()
        # ? if its the next day, reset the tokens used
        if last_chat_date.date() < timezone.now().date():
            self.reset_token_stock()
        if self.token_stock < tokens:
            raise ValueError("Not enough tokens available")
        self.tokens_used += tokens
        self.token_stock -= tokens
        self.save()

    def set_token_stock(self, tokens: int):
        self.token_stock = tokens
        self.save()
