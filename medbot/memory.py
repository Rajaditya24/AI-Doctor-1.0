# --- memory.py ---
# Compatible with LangChain v1.0+ (community split)

try:
    # New import path for LangChain >= 1.0
    from langchain_community.memory import ConversationBufferWindowMemory
except ImportError:
    # Fallback for older versions (< 1.0)
    from langchain.memory import ConversationBufferWindowMemory

from langchain.schema import HumanMessage, AIMessage
from datetime import datetime
import json
import re


class MedicalMemoryManager:
    """Manages conversational context and extracts medical information."""

    def __init__(self, k: int = 10):
        self.conversation_memory = ConversationBufferWindowMemory(k=k, return_messages=True)
        self.reset_session()

    def add_interaction(self, human_input: str, ai_response: str) -> None:
        """Adds a user/AI exchange and extracts information."""
        self.conversation_memory.chat_memory.add_user_message(human_input)
        self.conversation_memory.chat_memory.add_ai_message(ai_response)
        self._extract_medical_info(human_input)

    def _extract_medical_info(self, user_input: str) -> None:
        """Extracts relevant medical details from user input."""
        user_lower = user_input.lower()

        # --- Symptom extraction ---
        symptom_keywords = [
            "pain", "ache", "hurt", "sore", "cough", "fever", "nausea",
            "headache", "dizzy", "tired", "fatigue", "vomit", "swollen",
            "rash", "itch", "burn", "cramp", "bleed", "shortness of breath"
        ]
        if any(k in user_lower for k in symptom_keywords):
            if user_input not in self.patient_context["symptoms"]:
                self.patient_context["symptoms"].append(user_input)

        # --- Timeline extraction ---
        time_keywords = ["days", "weeks", "months", "hours", "yesterday", "today", "started", "began"]
        if any(k in user_lower for k in time_keywords):
            self.patient_context["timeline"].append(user_input)

        # --- Severity score extraction ---
        severity_match = re.search(r'\b([1-9]|10)\b.*(?:pain|severity|scale)', user_lower)
        if severity_match:
            self.patient_context["severity_scores"][datetime.now().isoformat()] = severity_match.group(1)

        # --- Medication extraction ---
        med_keywords = ["taking", "medication", "medicine", "pills", "prescribed", "drug"]
        if any(k in user_lower for k in med_keywords):
            self.patient_context["medications"].append(user_input)

        # --- Allergy extraction ---
        allergy_keywords = ["allergic", "allergy", "allergies", "reaction"]
        if any(k in user_lower for k in allergy_keywords):
            self.patient_context["allergies"].append(user_input)

    def get_memory_context(self) -> str:
        """Returns the recent conversation context."""
        messages = self.conversation_memory.chat_memory.messages[-6:]
        context = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context.append(f"Patient: {msg.content}")
            elif isinstance(msg, AIMessage):
                context.append(f"Doctor: {msg.content}")
        return "\n".join(context)

    def get_patient_summary(self) -> str:
        """Returns a JSON summary of extracted patient data."""
        summary = {
            "conversation_turns": len(self.conversation_memory.chat_memory.messages) // 2,
            "key_symptoms": self.patient_context["symptoms"][-3:],
            "timeline_info": self.patient_context["timeline"][-2:],
            "medications": self.patient_context["medications"],
            "allergies": self.patient_context["allergies"],
            "severity_scores": self.patient_context["severity_scores"]
        }
        return json.dumps(summary, indent=2)

    def reset_session(self) -> None:
        """Resets the conversation and patient context."""
        if hasattr(self, "conversation_memory"):
            self.conversation_memory.clear()

        self.patient_context = {
            "symptoms": [],
            "medications": [],
            "allergies": [],
            "timeline": [],
            "severity_scores": {},
            "session_start": datetime.now().isoformat()
        }
