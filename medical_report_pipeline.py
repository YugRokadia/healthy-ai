"""
Report-driven care plan pipeline:
  1. Docling extracts the medical report (not stored in vector DB)
  2. Gemini lists findings / potential issues from the report
  3. Gemini plans what to look up in the knowledge library
  4. RAG searches vector DB (nutrition/diet/reference docs only)
  5. Gemini produces structured diagnosis hints, diet plan, and general advice
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from docreader import read_medical_report
from llm_processor import LLMProcessor, MEDICAL_DISCLAIMER, EMERGENCY_NOTICE
from vector_db import NutritionHealthVectorDB


@dataclass
class ReportAnalysisResult:
    report_excerpt: str
    findings: list[str] = field(default_factory=list)
    rag_queries: list[str] = field(default_factory=list)
    knowledge_hits: list[dict] = field(default_factory=list)
    care_plan: str = ""


def _parse_json_array(text: str) -> list[str]:
    """Parse a JSON string array from model output; fallback to bullet lines."""
    text = (text or "").strip()
    if not text:
        return []
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-\*\d.)]+", "", line).strip()
        if len(line) > 3:
            lines.append(line)
    return lines[:12]


class MedicalReportPipeline:
    def __init__(self, diagnosis_db: NutritionHealthVectorDB, nutrition_db: NutritionHealthVectorDB, llm: LLMProcessor):
        self.diagnosis_db = diagnosis_db
        self.nutrition_db = nutrition_db
        self.llm = llm

    def extract_report_text(self, file_path: str) -> str | None:
        print("  Reading report with Docling...")
        return read_medical_report(
            file_path,
            fallback_extractor=self.diagnosis_db.extract_document_content,
        )

    def run(
        self,
        user_id: str,
        file_path: str,
        patient_notes: str = "",
    ) -> ReportAnalysisResult:
        result = ReportAnalysisResult(report_excerpt="")

        report_text = self.extract_report_text(file_path)
        if not report_text or len(report_text.strip()) < 30:
            raise ValueError("Could not extract text from the medical report.")

        result.report_excerpt = report_text[:2000]
        max_report = int(__import__("os").getenv("REPORT_MAX_CHARS", "24000"))
        report_for_llm = report_text[:max_report]

        if self.llm._needs_emergency_notice(patient_notes + " " + report_text[:5000]):
            print(f"\n*** {EMERGENCY_NOTICE} ***\n")

        print("\n  Step 1/4 — Extract findings and potential issues from report...")
        result.findings = self.llm.extract_report_findings(report_for_llm, patient_notes)
        self._print_numbered("Findings / potential issues", result.findings)

        print("\n  Step 2/4 — Evaluating Diagnosis & Gathering Context...")
        loop_count = 0
        current_context = patient_notes
        
        while loop_count < 10:
            evaluation = self.llm.evaluate_and_ask(report_for_llm, result.findings, current_context)
            if evaluation.strip().startswith("[READY]"):
                print("  Gemini: I have enough information to proceed.")
                break
            else:
                print(f"\nGemini asks: {evaluation}")
                user_answer = input("Your answer: ").strip()
                current_context += f"\nQ: {evaluation}\nA: {user_answer}"
                loop_count += 1
                
        print("\n  Step 3/4 — Plan knowledge-base lookups for diagnosis and nutrition...")
        result.rag_queries = self.llm.plan_rag_queries(result.findings, current_context)
        self._print_numbered("RAG search topics", result.rag_queries)

        print("\n  Step 4/4 — Search knowledge libraries (Diagnosis & Nutrition DBs)...")
        diag_hits = self._search_knowledge(self.diagnosis_db, user_id, result.rag_queries, result.findings)
        nutri_hits = self._search_knowledge(self.nutrition_db, user_id, result.rag_queries, result.findings)
        result.knowledge_hits = diag_hits + nutri_hits
        print(f"  Retrieved {len(result.knowledge_hits)} reference passages.")

        print("\n  Step 5/5 — Build structured diagnosis, diet, and fitness plan...")
        result.care_plan = self.llm.synthesize_care_plan(
            report_for_llm,
            result.findings,
            result.knowledge_hits,
            current_context,
        )
        return result

    def _search_knowledge(
        self, db: NutritionHealthVectorDB, user_id: str, rag_queries: list[str], findings: list[str]
    ) -> list[dict]:
        queries = list(rag_queries)
        for f in findings[:4]:
            if f and f not in queries:
                queries.append(f)

        seen_content: set[str] = set()
        hits: list[dict] = []
        per_query = int(__import__("os").getenv("RAG_HITS_PER_QUERY", "3"))

        for q in queries[:10]:
            if not q:
                continue
            rows = db.search_knowledge(q, user_id, limit=per_query)
            for row in rows:
                content = (row.get("content") or "").strip()
                key = content[:120]
                if content and key not in seen_content:
                    seen_content.add(key)
                    hits.append(row)
        return hits[:20]

    @staticmethod
    def _print_numbered(title: str, items: list[str]) -> None:
        print(f"  {title}:")
        if not items:
            print("    (none)")
            return
        for i, item in enumerate(items, 1):
            print(f"    {i}. {item}")
