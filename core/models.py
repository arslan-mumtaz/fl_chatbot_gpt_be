import uuid
from django.db import models
from hashids import Hashids

hashids = Hashids(min_length=8, salt="your_salt_here")


class BaseModel(models.Model):
    id = models.CharField(max_length=32, primary_key=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    STATUS_CHOICES = (
        ("active", "Active"),
        ("inactive", "Inactive"),
        ("deleted", "Deleted"),
    )
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="active")

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if not self.id:
            uuid_hex = uuid.uuid4().hex
            self.id = hashids.encode(int(uuid_hex, 16))
        super().save(*args, **kwargs)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_inactive(self) -> bool:
        return self.status == "inactive"

    @property
    def is_deleted(self) -> bool:
        return self.status == "deleted"

    def delete(self, *args, **kwargs):
        self.status = "deleted"
        self.save()

    def activate(self, *args, **kwargs):
        self.status = "active"
        self.save()

    def deactivate(self, *args, **kwargs):
        self.status = "inactive"
        self.save()
