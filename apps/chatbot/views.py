from apps.chatbot.models import Document, PermissionRequest
from apps.chatbot.serializers import DocumentSerializer, PermissionRequestSerializer
from rest_framework import permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from custom.permissions.permissions import IsChatbotUser, IsSuperAdmin
from apps.chatbot.utils.chatbot.chatbot import ChatBot
from apps.user.models import BaseUser
from rest_framework import viewsets, views
from django.shortcuts import get_object_or_404
from django.db import transaction
from rest_framework.response import Response
from rest_framework import status
import tempfile
import os
from reportlab.pdfgen import canvas
from rna_utils import debug_print
from io import BytesIO
from reportlab.lib.pagesizes import letter
from django.core.files.uploadedfile import InMemoryUploadedFile
from tasks.embeddings_tasks import upload_document_to_vectorstore


class PermissionRequestViewSet(viewsets.ModelViewSet):
    http_method_names = ["get", "post", "delete"]
    queryset = PermissionRequest.objects.all().select_related("user")
    serializer_class = PermissionRequestSerializer
    permission_classes = (permissions.IsAuthenticated,)
    filterset_fields = ["request_status", "status", "user"]

    def get_permissions(self):
        if self.action in (
            "approve",
            "reject",
        ):
            return (IsSuperAdmin(),)
        return super().get_permissions()

    def create(self, request, *args, **kwargs):
        request.data["user"] = request.user.id
        user = get_object_or_404(BaseUser, id=request.user.id)
        if user.is_chatbot_user:
            return Response({"info": "You already have access to the chatbot! 🤖"})

        permission = PermissionRequest.objects.create(user=user)
        serializer = PermissionRequestSerializer(permission)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def approve(self, request, pk=None):
        permission_request = self.get_object()
        permission_request.approve_request()
        return Response({"status": "approved", "message": "Request approved!"})

    @action(detail=True, methods=["post"])
    def reject(self, request, pk=None):
        permission_request = self.get_object()
        permission_request.reject_request()
        return Response({"status": "rejected", "message": "Request rejected!"})


class ChatBotApiView(views.APIView):
    permission_classes = (IsChatbotUser | IsSuperAdmin,)

    def post(self, request):
        prompt = request.data.get("prompt", None)
        history = request.data.get("history", [])
        if not prompt:
            return Response(
                {"detail": "Prompt is required!"}, status=status.HTTP_400_BAD_REQUEST
            )
        completion = ChatBot(history=history).get_completion(prompt=prompt)
        return Response(data=completion)


class DocumentViewSet(viewsets.ModelViewSet):
    http_method_names = ["get", "post"]
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    permission_classes = (IsSuperAdmin,)

    def save_as_pdf(self, document):
        temp_pdf = BytesIO()
        c = canvas.Canvas(temp_pdf, pagesize=letter)
        c.drawString(100, 750, document.read().decode("utf-8"))
        c.showPage()
        c.save()
        temp_pdf.seek(0)
        return InMemoryUploadedFile(
            temp_pdf, None, "document.pdf", "application/pdf", temp_pdf.tell(), None
        )

    def list(self, request, *args, **kwargs):
        self.queryset = self.queryset.filter(status="active").order_by("-created_at")
        return super().list(request, *args, **kwargs)

    @transaction.atomic
    def create(self, request, *args, **kwargs):
        user_id = request.user.id
        documents = request.FILES.getlist("document")
        if not documents:
            return Response(
                {"detail": "No documents provided."}, status=status.HTTP_400_BAD_REQUEST
            )

        results = []
        for document in documents:
            original_name = document.name.rsplit(".", 1)[0]
            document_name = document.name
            document_type = document.content_type

            # Reading document content into memory
            document_content = document.read()

            if document.content_type != "application/pdf":
                # Converting to PDF
                pdf_file = self.save_as_pdf(BytesIO(document_content))
                document_content = pdf_file.read()
                document_type = "application/pdf"
                document_name = f"{document_name}.pdf"

            # Create a new data dictionary for each document
            data = {
                "user": user_id,
                "name": document_name,
                "type": document_type,
                "document": document,
            }

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_file:
                    temp_file.write(document_content)
                    temp_file_path = temp_file.name

                # Create the document record in the database
                serializer = self.get_serializer(data=data)
                serializer.is_valid(raise_exception=True)
                self.perform_create(serializer)
                res = serializer.data

                if res and serializer.instance:
                    upload_document_to_vectorstore(serializer.instance.id)
                    results.append(res)

                os.remove(temp_file_path)
            except Exception as e:
                return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(results, status=status.HTTP_201_CREATED)

    @action(
        detail=False,
        methods=["post"],
        url_path="recreate-embeddings",
    )
    def recreate_embeddings(self, request):
        ChatBot().recreate_embeddings()
        return Response({"message": "Embeddings recreated!"})

    @action(
        detail=False,
        methods=["get"],
        url_path="vectorstore",
    )
    def vectorstore(self, request):
        try:
            vectorstore_documents = ChatBot().get_vectorstore_documents()
            return Response(vectorstore_documents)
        except Exception as e:
            return Response(
                {"message": "No vectorstore found!"}, status=status.HTTP_400_BAD_REQUEST
            )

    @action(
        detail=False,
        methods=["post"],
        url_path="delete",
    )
    def delete_docs(self, request):
        documents_list = request.data.get("documents", [])
        if not documents_list:
            return Response(
                {"detail": "No documents provided."}, status=status.HTTP_400_BAD_REQUEST
            )
        ChatBot().delete_documents(documents_list)
        return Response({"message": "Documents deleted!"})
