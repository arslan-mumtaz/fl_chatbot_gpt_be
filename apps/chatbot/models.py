from django.db import models
from core.models import BaseModel


class PermissionRequest(BaseModel):
    """
    This model is used to store the permission requests made by the users to access the chatbot.
    """

    user = models.ForeignKey("user.BaseUser", on_delete=models.CASCADE)

    # ? The user can request to be allowed access to the chatbot
    REQUEST_STATUS_CHOICES = (
        ("pending", "Pending"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
    )
    request_status = models.CharField(
        max_length=10, choices=REQUEST_STATUS_CHOICES, default="pending"
    )

    def approve_request(self):
        self.request_status = "approved"
        self.user.allow_chatbot_access()
        self.save()

    def reject_request(self):
        self.request_status = "rejected"
        self.save()

    class Meta:
        app_label = "chatbot"
        ordering = ["-created_at"]


class Document(BaseModel):
    """
    This model is used to store the documents uploaded by the users.
    """

    user = models.ForeignKey("user.BaseUser", on_delete=models.SET_NULL, null=True)

    document = models.FileField(upload_to="uploaded_documents/")
    name = models.CharField(max_length=255)
    type = models.CharField(max_length=255)
    embeddings_status = models.CharField(max_length=255, default="pending")

    vectorstore_ids = models.TextField(null=True)

    def __str__(self):
        return self.name

    def set_embeddings_uploaded(self):
        self.embeddings_status = "uploaded"
        self.save()

    def set_embeddings_failed(self):
        self.embeddings_status = "failed"
        self.save()

    def delete(self, *args, **kwargs):
        self.embeddings_status = "deleted"
        return super().delete(*args, **kwargs)

    class Meta:
        app_label = "chatbot"
        ordering = ["-created_at"]


class ChatHistory(BaseModel):
    """
    This model is used to store the chat history of the users.
    """

    user = models.ForeignKey("user.BaseUser", on_delete=models.SET_NULL, null=True)
    message = models.TextField()
    response = models.TextField()
    tokens_used = models.IntegerField(default=0)

    def __str__(self):
        return self.message

    class Meta:
        app_label = "chatbot"
        ordering = ["-created_at"]
