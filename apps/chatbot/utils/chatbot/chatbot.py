import io
import json
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import pdfplumber
import ast
from typing import Optional
import numpy as np
from operator import itemgetter
from langchain_core.documents import Document as DocumentType
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import shutil
from rna_utils import color_print, debug_print
from langchain_community.callbacks import get_openai_callback
from apps.chatbot.models import Document
from django.core.files.base import ContentFile
from core.settings import OPENAI_API_KEY
import re
import camelot.io as camelot
import pandas as pd
import csv
from langchain.prompts import PromptTemplate
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field


GPT_4o = "gpt-4o-2024-05-13"
VECTORESTORE_PATH = "chatbot/vectorstore"
IMAGES_PATH = "extracted_images"
TABLES_PATH = "extracted_tables"

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

BACKEND_URL = "http://localhost:8000"


class ChatBot:
    def __init__(self, test_doc=None, history: list = []) -> None:
        if not os.path.exists(IMAGES_PATH):
            os.makedirs(IMAGES_PATH)

        if not os.path.exists(TABLES_PATH):
            os.makedirs(TABLES_PATH)

        self.OPENAI_API_KEY = OPENAI_API_KEY  # type: ignore
        self.test_doc = test_doc

        if not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required")

        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY  # type: ignore

        self.llm = ChatOpenAI(api_key=self.OPENAI_API_KEY, model=GPT_4o, temperature=0)  # type: ignore

        self.embeddings = OpenAIEmbeddings()

        self.history = ChatMessageHistory()

        for obj in history:
            self.history.add_user_message(obj["prompt"])
            self.history.add_ai_message(obj["completion"])

    def load_document(
        self, path: str, file_name: str | None = None
    ) -> list[DocumentType]:
        pages = []
        pdf = fitz.open(path)  # type: ignore
        pdf_reader = PdfReader(path)
        document_name = (
            os.path.basename(path).replace(".pdf", "") if not file_name else file_name
        )

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            # Extract tables and images
            images_text, image_paths = self._extract_images_text(
                pdf, page_num, document_name
            )
            tables_text, table_paths = self._extract_tables_text(
                path, page_num, document_name
            )

            combined_text = text + "\n" + tables_text + "\n" + images_text
            page_metadata = {"images": image_paths, "tables": table_paths}
            pages.append(
                DocumentType(page_content=combined_text, metadata=page_metadata)
            )

        return pages

    def _extract_images_text(self, pdf, page_num, document_name):
        images_text = ""
        image_paths = []
        try:
            page = pdf.load_page(page_num)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                # Pre-process the image
                pre_processed_image = self._preprocess_image(image)
                images_text += pytesseract.image_to_string(pre_processed_image)

                # Save the image for verification
                image_path = f"{IMAGES_PATH}/{document_name}_page_{page_num + 1}_img_{img_index + 1}.png"
                image.save(image_path)
                image_paths.append(image_path)
                color_print(f"Saved extracted image to {image_path}", "green")

            color_print(
                f"Successfully extracted images from page {page_num + 1}", "green"
            )
        except Exception as e:
            color_print(
                f"Failed to extract images from page {page_num + 1}: {e}", "red"
            )
        return images_text, image_paths

    def _preprocess_image(self, image):
        # Convert image to grayscale and apply thresholding for better OCR accuracy
        image = image.convert("L")
        image = image.point(lambda x: 0 if x < 128 else 255, "1")
        return image

    def _extract_tables_text(self, path, page_num, document_name):
        tables_html = ""
        table_paths = []
        try:
            # Read tables from the specific page using camelot
            tables = camelot.read_pdf(path, pages=str(page_num + 1), flavor="stream")
            if tables:
                for table_index, table in enumerate(tables):  # type: ignore
                    tables_html += table.df.to_html(index=False, header=False)

                    # Save the table for verification
                    table_path = f"{TABLES_PATH}/{document_name}_page_{page_num + 1}_table_{table_index + 1}.csv"
                    table.df.to_csv(table_path, index=False)
                    table_paths.append(table_path)
                    color_print(f"Saved extracted table to {table_path}", "green")

                color_print(
                    f"Successfully extracted tables from page {page_num + 1} using Camelot",
                    "green",
                )
            else:
                color_print(
                    f"No tables found with Camelot on page {page_num + 1}", "yellow"
                )
        except Exception as e:
            color_print(
                f"Camelot failed on page {page_num + 1}: {e}. Trying pdfplumber...",
                "yellow",
            )
            tables_html, table_paths = self._extract_tables_with_pdfplumber(
                path, page_num, document_name
            )
        return tables_html, table_paths

    def _extract_tables_with_pdfplumber(self, path, page_num, document_name):
        tables_html = ""
        table_paths = []
        try:
            with pdfplumber.open(path) as pdf:
                page = pdf.pages[page_num]
                tables = page.extract_tables()
                if tables:
                    for table_index, table in enumerate(tables):
                        tables_html += pd.DataFrame(table).to_html(
                            index=False, header=False
                        )

                        # Save the table for verification
                        table_path = f"{TABLES_PATH}/{document_name}_page_{page_num + 1}_table_{table_index + 1}.csv"
                        with open(table_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerows(table)
                        table_paths.append(table_path)
                        color_print(f"Saved extracted table to {table_path}", "green")

                    color_print(
                        f"Successfully extracted tables from page {page_num + 1} using pdfplumber",
                        "green",
                    )
                else:
                    color_print(
                        f"No tables found on page {page_num + 1} using pdfplumber",
                        "yellow",
                    )
        except Exception as e:
            color_print(f"pdfplumber also failed on page {page_num + 1}: {e}", "red")
        return tables_html, table_paths

    def _format_docs(self, docs: list[DocumentType]) -> str:
        formatted_docs = ""
        for doc in docs:
            doc_content = doc.page_content
            metadata = doc.metadata
            formatted_doc = f"<p>{doc_content}</p>"

            if metadata.get("tables"):
                for table_path in metadata["tables"]:
                    full_table_url = f"{BACKEND_URL}/{table_path}"
                    with open(table_path, "r") as table_file:
                        formatted_doc += pd.read_csv(table_file).to_html(index=False)

            if metadata.get("images"):
                for image_path in metadata["images"]:
                    full_image_url = f"{BACKEND_URL}/{image_path}"
                    formatted_doc += f'<img src="{full_image_url}" alt="Image"/>'

            formatted_docs += formatted_doc
        return formatted_docs

    def split_documents(self, pages: list[DocumentType]) -> list[DocumentType]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(
            pages
        )  # Ensure split_documents gets a list of DocumentType objects
        return splits

    def get_or_create_vectorstore(
        self, split_documents: list[DocumentType] | None = None
    ) -> FAISS:
        try:

            vectorstore = FAISS.load_local(
                VECTORESTORE_PATH, self.embeddings, allow_dangerous_deserialization=True
            )
            color_print("Loaded existing vectorstore", "green")
        except Exception as e:
            color_print(f"Creating new vectorstore: {str(e)}", "cyan")
            # if self.test_doc:
            #     loaded_documents = self.load_document(f"documents/{self.test_doc}")
            #     print(loaded_documents)
            #     split_documents = self.split_documents(loaded_documents)
            # else:
            # if not split_documents:
            #     loaded_documents = []
            # for _, document in enumerate(os.listdir("documents")):
            #     color_print(f"Processing document: {document}", "yellow")
            #     loaded_doc = self.load_document(f"documents/{document}")
            #     loaded_documents.extend(loaded_doc)
            #     with open(f"documents/{document}", "rb") as f:
            #         file = f.read()
            #         document_file = ContentFile(file, name=document)
            #         Document.objects.create(
            #             document=document_file,
            #             name=document,
            #             type="application/pdf",
            #         )

            # split_documents = self.split_documents(loaded_documents)
            # else:
            #     color_print("Using provided documents", "yellow")

            vectorstore = FAISS.from_documents(split_documents, self.embeddings)  # type: ignore
            vectorstore.save_local(VECTORESTORE_PATH)
            color_print("Saved vectorstore", "green")
        return vectorstore

    def get_retriever(self, vectorstore: FAISS) -> VectorStoreRetriever:
        return vectorstore.as_retriever()

    def get_rag_chain(self, retriever: VectorStoreRetriever) -> Runnable:
        # prompt_template = PromptTemplate(
        #     template="""
        #     Please answer the following question using the provided context. Include any relevant tables, images, or figures within the response. Format the tables and images properly for rendering in a browser. Provide the response in HTML format suitable for web rendering. If you do not know the answer, please state so and provide answer only from the embeddings documents provided to you. The response has to be rendered in a browser, so make sure to include the necessary HTML tags for formatting. Also apply some styling as you see fit to make it please UI wise. Also please do not include the question/prompt in the response. Make sure to cross check the dates and values so that they are accurate.

        #     You are also provided with the history of the current chat session. Please use this information to provide a more accurate response if available. Otherwise, you can ignore it.

        #     Question: {question}
        #     Context: {context}
        #     """,
        #     input_variables=[
        #         "question",
        #         "context",
        #     ],
        # )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Please answer the following question using the provided context. Include any relevant tables, images, or figures within the response. Format the tables and images properly for rendering in a browser. Provide the response in HTML format suitable for web rendering. If you do not know the answer, please state so and provide answer only from the embeddings documents provided to you. The response has to be rendered in a browser, so make sure to include the necessary HTML tags for formatting. Also apply some styling as you see fit to make it please UI wise. Also please do not include the question/prompt in the response. Make sure to cross check the dates and values so that they are accurate.

                    You are also provided with the history of the current chat session. Please use this information to provide a more accurate response if available. Otherwise, you can ignore it.

                    Question: {question}
                    Context: {context}
                """,
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | retriever | self._format_docs,
            )
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        chain_with_history = RunnableWithMessageHistory(
            chain,  # type: ignore
            get_session_history=self._get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        return chain_with_history

    def get_completion(
        self,
        prompt: str,
    ) -> str:
        with get_openai_callback() as cb:
            vectorstore = self.get_or_create_vectorstore()
            retriever = self.get_retriever(vectorstore)
            rag_chain = self.get_rag_chain(
                retriever,
            )
            res = rag_chain.invoke(
                input={
                    "question": prompt,
                    "history": self.history.messages,
                },
                config={"configurable": {"session_id": "test"}},
                # config={"callbacks": [ConsoleCallbackHandler()]},
            )
            color_print(f"Completion: {res}", "yellow")
            color_print(cb, "cyan")
            return res

    def upload_file_to_vectorstore(self, path: str, file_name: str) -> list[str]:
        document = self.load_document(path, file_name)
        split_documents = self.split_documents(document)
        try:
            vectorstore = FAISS.load_local(
                VECTORESTORE_PATH, self.embeddings, allow_dangerous_deserialization=True
            )
            vectorstore_ids = vectorstore.add_documents(split_documents)
            vectorstore.save_local(VECTORESTORE_PATH)
            color_print(f"uploading from existing: {len(vectorstore_ids)}", "green")
            return vectorstore_ids
        except:
            vectorstore = FAISS.from_documents(split_documents, self.embeddings)
            vectorstore.save_local(VECTORESTORE_PATH)
            vectorstore_ids = vectorstore.index_to_docstore_id.values()
            vectorstore_ids = list(vectorstore_ids)
            color_print(f"uploading from scratch: {len(vectorstore_ids)}", "yellow")
            return vectorstore_ids

    def get_vectorstore_documents(self) -> dict:
        vectorstore = self.get_or_create_vectorstore()
        return vectorstore.docstore.__dict__

    def _delete_vectorstore(self) -> None:
        try:
            shutil.rmtree(VECTORESTORE_PATH)
        except (FileNotFoundError, PermissionError) as e:
            color_print(f"Failed to delete vectorstore: {str(e)}", "red")

    def _delete_documents(self) -> None:
        Document.objects.all().delete()
        path = "uploaded_documents"
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except (FileNotFoundError, PermissionError) as e:
            color_print(f"Failed to delete documents: {str(e)}", "red")

    def recreate_embeddings(self) -> None:
        color_print("Deleting documents", "red")
        self._delete_documents()

        color_print("Removing vectorstore", "red")
        self._delete_vectorstore()

        color_print("Recreating embeddings", "green")
        self.get_or_create_vectorstore()

    def _get_session_history(self, session_id) -> BaseChatMessageHistory:
        return self.history

    def delete_documents(self, documents_list: list):
        """
        Delete the documents and associated chatbot data which includes:
        - Embeddings in vectorstore
        - Documents in the database
        - Extracted images and tables
        - Document from uploaded_documents
        """
        # Get existing vectorstore
        vectorstore = self.get_or_create_vectorstore()
        total_embeddings = vectorstore.index.ntotal
        color_print(f"Total embeddings before deletion: {total_embeddings}", "red")

        # Collect all document IDs to be deleted
        for document in documents_list:
            delete_vectorstore_ids = set()
            document_id = document["id"]

            try:
                # Parse the string literal to a list
                vectorstore_ids_str = document.get("vectorstore_ids", "[]")
                try:
                    vectorstore_ids = ast.literal_eval(vectorstore_ids_str)
                    if not isinstance(vectorstore_ids, list):
                        raise ValueError("vectorstore_ids is not a list")
                    delete_vectorstore_ids.update(vectorstore_ids)
                except (ValueError, SyntaxError) as e:
                    color_print(
                        f"Failed to parse vectorstore_ids for document ID {document_id}: {e}",
                        "red",
                    )
                    continue

                # Log the IDs being processed
                color_print(f"Processing document ID: {document_id}", "cyan")

                # Delete the embeddings from the vectorstore using the remove function
                if delete_vectorstore_ids:
                    try:
                        vectorstore.delete(list(delete_vectorstore_ids))
                        vectorstore.save_local(VECTORESTORE_PATH)
                        color_print(
                            f"Deleted { len(delete_vectorstore_ids)} embeddings from the vectorstore",
                            "green",
                        )
                    except ValueError as e:
                        color_print(f"Failed to delete embeddings: {str(e)}", "red")
                else:
                    color_print("No valid IDs found to delete", "yellow")

                document_obj = Document.objects.get(id=document_id)
                document_path = document_obj.document.path

                # Delete document from the database
                document_obj.delete()
                color_print(
                    f"Deleted document {document_id} from the database", "green"
                )

                # Delete document file from the file system
                if os.path.exists(document_path):
                    os.remove(document_path)
                    color_print(
                        f"Deleted document file {document_path} from the file system",
                        "green",
                    )
                else:
                    color_print(
                        f"Document file {document_path} not found in the file system",
                        "yellow",
                    )

                # Delete extracted images and tables
                images = document.get("metadata", {}).get("images", [])
                tables = document.get("metadata", {}).get("tables", [])

                for image_path in images:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        color_print(f"Deleted extracted image {image_path}", "green")

                for table_path in tables:
                    if os.path.exists(table_path):
                        os.remove(table_path)
                        color_print(f"Deleted extracted table {table_path}", "green")

            except Document.DoesNotExist:
                color_print(
                    f"Document with ID {document_id} does not exist in the database",
                    "yellow",
                )
            except Exception as e:
                color_print(f"Failed to delete document {document_id}: {str(e)}", "red")

        # Reload vectorstore to verify that embeddings are actually deleted
        try:
            vectorstore = FAISS.load_local(
                VECTORESTORE_PATH, self.embeddings, allow_dangerous_deserialization=True
            )
            color_print("Reloaded vectorstore to verify deletions", "green")
            remaining_embeddings = vectorstore.index.ntotal
            color_print(
                f"Remaining embeddings after deletion: {remaining_embeddings}", "red"
            )
        except Exception as e:
            color_print(f"Failed to reload vectorstore: {e}", "red")

        color_print("Completed deletion process", "green")
