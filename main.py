import os
import sys

from dotenv import load_dotenv

load_dotenv()

import requests
import getpass

from user_auth import UserAuth
from vector_db import NutritionHealthVectorDB, KNOWLEDGE_DOC_TYPES
from llm_processor import LLMProcessor, EMERGENCY_NOTICE, APP_LLM_VERSION
from medical_report_pipeline import MedicalReportPipeline

MEDRAG_VERSION = "2.0.0"

DOC_TYPE_LABELS = {
    "clinical": "Clinical guidelines & medical reference",
    "nutrition": "Nutrition science & macronutrients",
    "diet": "Diet plans & meal guides",
    "general": "General wellness & lifestyle",
}


class MedicalHealthApp:
    def __init__(self):
        self.user_auth = UserAuth()
        self.diagnosis_db = NutritionHealthVectorDB(db_path="./diagnosis_vectordb")
        self.nutrition_db = NutritionHealthVectorDB(db_path="./nutrition_vectordb")
        self.llm_processor = LLMProcessor()
        self.pipeline = MedicalReportPipeline(self.diagnosis_db, self.nutrition_db, self.llm_processor)
        self.current_user = None

    def login_or_register(self):
        print("MedRAG - Medical Health Assistant")
        print(f"Build {MEDRAG_VERSION} | LLM module {APP_LLM_VERSION}")
        print(f"Gemini model: {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-lite')}")
        print("Knowledge library (vector DB) + medical report analysis (Docling)")
        print("=" * 60)

        while True:
            print("\n1. Login")
            print("2. Register")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                if self.login():
                    return True
            elif choice == "2":
                self.register()
            elif choice == "3":
                print("Goodbye!")
                return False
            else:
                print("Invalid choice. Please try again.")

    def login(self):
        print("\n--- Login ---")
        user_id = input("Enter User ID: ").strip()
        password = getpass.getpass("Enter Password: ")

        if self.user_auth.authenticate_user(user_id, password):
            self.current_user = user_id
            print(f"Welcome, {user_id}!")
            return True
        print("Login failed. Please check your credentials.")
        return False

    def register(self):
        print("\n--- Register ---")
        user_id = input("Enter User ID: ").strip()

        if self.user_auth.user_exists(user_id):
            print("User ID already exists. Please choose a different one.")
            return

        password = getpass.getpass("Enter Password: ")
        confirm_password = getpass.getpass("Confirm Password: ")

        if password != confirm_password:
            print("Passwords do not match.")
            return

        if self.user_auth.register_user(user_id, password):
            print("Registration successful! You can now login.")
        else:
            print("Registration failed. Please try again.")

    def download_file(self, url, filename):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False

    def _resolve_file_path(self, purpose: str) -> str | None:
        print(f"\nEnter {purpose} source:")
        print("1. Local file path")
        print("2. URL link")
        choice = input("Enter choice (1-2): ").strip()

        if choice == "1":
            path = input("Enter local file path: ").strip().strip('"')
            if not os.path.exists(path):
                print("File not found.")
                return None
            return path
        if choice == "2":
            url = input("Enter document URL: ").strip()
            filename = f"temp_{self.current_user}_{os.path.basename(url.split('?')[0])}"
            if not self.download_file(url, filename):
                return None
            return filename
        print("Invalid choice.")
        return None

    def _prompt_knowledge_type(self) -> str:
        print("\nKnowledge categories (stored for RAG lookup only):")
        for key in KNOWLEDGE_DOC_TYPES:
            print(f"  - {key}: {DOC_TYPE_LABELS.get(key, key)}")
        doc_type = input("Enter type (default: nutrition): ").strip() or "nutrition"
        if doc_type not in KNOWLEDGE_DOC_TYPES:
            print(f"Unknown type '{doc_type}', using 'nutrition'.")
            doc_type = "nutrition"
        return doc_type

    def add_knowledge_document(self):
        """Add reference material to vector DB (not patient reports)."""
        print("\n--- Add to Knowledge Library ---")
        print("Upload nutrition guides, diet plans, or clinical reference PDFs.")
        print("Patient medical reports are analyzed via menu 4 (not stored here).")
        
        print("\nWhich database should this be stored in?")
        print("1. Diagnosis DB (medical reference, clinical guidelines)")
        print("2. Nutrition DB (diet plans, fitness, nutrition science)")
        db_choice = input("Enter choice (1-2): ").strip()
        
        if db_choice == "1":
            target_db = self.diagnosis_db
        else:
            target_db = self.nutrition_db

        file_path = self._resolve_file_path("knowledge document")
        if not file_path:
            return

        doc_type = self._prompt_knowledge_type()
        access_level = input("Access level (private/public): ").strip() or "private"
        temp_file = file_path.startswith(f"temp_{self.current_user}_")

        print(f"Indexing: {file_path}")
        document_id = target_db.add_document(
            file_path, self.current_user, access_level, doc_type
        )

        if document_id:
            self.user_auth.add_user_document(
                self.current_user,
                document_id,
                os.path.basename(file_path),
                doc_type,
                access_level,
            )
            print("Knowledge document indexed for RAG.")
        else:
            print("Failed to index document.")

        if temp_file:
            try:
                os.remove(file_path)
            except OSError:
                pass

    def list_documents(self):
        print("\n--- Knowledge Library ---")
        documents = self.user_auth.get_user_documents(self.current_user)

        if not documents:
            print("No knowledge documents. Add nutrition/diet guides (menu 1).")
            return

        print(f"{'ID':<36} {'Filename':<30} {'Type':<12} {'Access':<8} {'Added'}")
        print("-" * 105)
        for doc in documents:
            print(
                f"{doc['document_id']:<36} {doc['filename']:<30} {doc['doc_type']:<12} "
                f"{doc['access_level']:<8} {doc['added_at'].strftime('%Y-%m-%d')}"
            )

    def remove_document(self):
        print("\n--- Remove Knowledge Document ---")
        documents = self.user_auth.get_user_documents(self.current_user)

        if not documents:
            print("No documents to remove.")
            return

        for i, doc in enumerate(documents, 1):
            print(f"{i}. {doc['filename']} ({doc['doc_type']})")

        try:
            choice = int(input("Document number to remove: ")) - 1
            if 0 <= choice < len(documents):
                doc = documents[choice]
                if input(f"Remove '{doc['filename']}'? (y/N): ").strip().lower() == "y":
                    self.diagnosis_db.delete_user_documents(
                        self.current_user, doc["document_id"]
                    )
                    self.nutrition_db.delete_user_documents(
                        self.current_user, doc["document_id"]
                    )
                    self.user_auth.remove_user_document(
                        self.current_user, doc["document_id"]
                    )
                    print("Removed.")
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input.")

    def _check_api_before_query(self) -> bool:
        print("\n--- Gemini API check ---")
        self.llm_processor._rate_status = None
        status = self.llm_processor.check_api_rate(force_refresh=True)
        self.llm_processor.print_rate_status(status)
        if status.get("ok"):
            return True
        if not self.llm_processor.is_available():
            return False
        return input("\nContinue without Gemini (limited output)? (y/N): ").strip().lower() == "y"

    def analyze_medical_report(self):
        """
        Main query flow: feed a medical report → Docling → Gemini findings →
        RAG on knowledge library → structured diet plan + advice.
        """
        print("\n--- Analyze Medical Report ---")
        print("Reports are read with Docling and are NOT saved to the vector database.")
        print("Diet and lifestyle answers come from your knowledge library (menu 1).")

        documents = self.user_auth.get_user_documents(self.current_user)
        if not documents:
            print("\nWarning: No knowledge documents in library. Add diet/nutrition PDFs first (menu 1).")

        if not self._check_api_before_query():
            print("Gemini is required for report analysis.")
            return

        file_path = self._resolve_file_path("medical report")
        if not file_path:
            return

        print("\nOptional patient context (age, symptoms, allergies, medications):")
        patient_notes = input("> ").strip()
        temp_file = file_path.startswith(f"temp_{self.current_user}_")

        try:
            analysis = self.pipeline.run(
                self.current_user,
                file_path,
                patient_notes=patient_notes,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return
        finally:
            if temp_file:
                try:
                    os.remove(file_path)
                except OSError:
                    pass

        print("\n" + "=" * 80)
        print("STRUCTURED CARE PLAN")
        print("=" * 80)
        print(analysis.care_plan)
        print("=" * 80)

    def main_menu(self):
        while True:
            print(f"\n--- MedRAG | User: {self.current_user} ---")
            print("1. Add knowledge document (nutrition/diet guides → vector DB)")
            print("2. List knowledge library")
            print("3. Remove knowledge document")
            print("4. Analyze medical report (Docling + RAG + Gemini)")
            print("5. Logout")

            choice = input("\nEnter your choice (1-5): ").strip()

            actions = {
                "1": self.add_knowledge_document,
                "2": self.list_documents,
                "3": self.remove_document,
                "4": self.analyze_medical_report,
            }

            if choice in actions:
                actions[choice]()
            elif choice == "5":
                print("Logging out...")
                self.current_user = None
                break
            else:
                print("Invalid choice.")


def main():
    app = MedicalHealthApp()
    if app.login_or_register():
        app.main_menu()


if __name__ == "__main__":
    main()
