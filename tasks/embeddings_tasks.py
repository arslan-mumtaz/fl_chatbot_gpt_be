# your_app/tasks.py

import logging
from rna_utils import run_in_background
from apps.chatbot.models import Document
from apps.chatbot.utils.chatbot.chatbot import ChatBot


logger = logging.getLogger(__name__)


@run_in_background
def upload_document_to_vectorstore(document_id):
    logger.info(
        f"Starting upload_document_to_vectorstore task for document_id: {document_id}"
    )
    document = Document.objects.get(id=document_id)
    original_name = document.name.rsplit(".", 1)[0]
    document_path = document.document.path

    try:
        vectorstore_ids = ChatBot().upload_file_to_vectorstore(
            path=document_path, file_name=original_name
        )
        if vectorstore_ids:
            document.vectorstore_ids = vectorstore_ids  # type: ignore
            document.set_embeddings_uploaded()
            logger.info(
                f"Successfully uploaded document {document_id} to vectorstore with IDs: {vectorstore_ids}"
            )
        else:
            document.set_embeddings_failed()
            logger.warning(f"No vectorstore IDs returned for document {document_id}")
    except Exception as e:
        document.set_embeddings_failed()
        logger.error(f"Error uploading document {document_id} to vectorstore: {e}")
