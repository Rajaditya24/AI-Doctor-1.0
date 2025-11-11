import logging
from .model import ModelManager
from .memory import MedicalMemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

model_manager = ModelManager()
memory_manager = MedicalMemoryManager()
conversation_turns = 0


def build_gemini_prompt(system_prompt, history, user_input):
    """
    Build a comprehensive prompt for Gemini API that includes system instructions,
    conversation history, and memory context.
    """
    memory_context = memory_manager.get_memory_context()
    
    # Build the full prompt with context
    prompt = f"{system_prompt}\n\n"
    
    if memory_context:
        prompt += f"Previous conversation context:\n{memory_context}\n\n"
    
    # Add recent history (last 3 exchanges for context)
    recent_history = history[-3:] if len(history) > 3 else history
    
    if recent_history:
        prompt += "Recent conversation:\n"
        for user_msg, assistant_msg in recent_history:
            prompt += f"Patient: {user_msg}\n"
            if assistant_msg:
                prompt += f"Doctor: {assistant_msg}\n"
        prompt += "\n"
    
    # Add current user input
    prompt += f"Patient: {user_input}\n\nDoctor:"
    
    return prompt


def respond(message, chat_history):
    """
    Main response handler that manages the consultation flow
    """
    global conversation_turns
    conversation_turns += 1
    
    logging.info(f"Turn {conversation_turns} - User input: {message}")
    
    try:
        # Import prompts here to avoid circular imports
        from .prompts import CONSULTATION_PROMPT, MEDICINE_PROMPT
        
        # Phase 1: Information Gathering (first 3 turns)
        if conversation_turns < 4:
            logging.info("Phase 1: Information gathering with CONSULTATION_PROMPT")
            
            prompt = build_gemini_prompt(CONSULTATION_PROMPT, chat_history, message)
            response = model_manager.generate(prompt, max_new_tokens=256, temperature=0.7)
            
            logging.info(f"Model response: {response[:100]}...")
            
            # Store interaction in memory
            memory_manager.add_interaction(message, response)
            chat_history.append((message, response))
            
            return "", chat_history
        
        # Phase 2: Summary and Recommendations (turn 4 onwards)
        else:
            logging.info("Phase 2: Generating comprehensive summary and recommendations")
            
            # Get patient summary from memory
            patient_summary = memory_manager.get_patient_summary()
            memory_context = memory_manager.get_memory_context()
            
            # Generate comprehensive summary
            summary_prompt = build_gemini_prompt(
                CONSULTATION_PROMPT + "\n\nNow provide a comprehensive summary of all the information gathered. Include assessment of severity and when professional care may be needed.",
                chat_history,
                message
            )
            
            logging.info("Generating summary...")
            summary = model_manager.generate(summary_prompt, max_new_tokens=500, temperature=0.7)
            logging.info(f"Summary generated: {summary[:100]}...")
            
            # Generate medicine and care suggestions
            full_patient_info = f"Patient Summary: {patient_summary}\n\nDetailed Assessment: {summary}"
            
            med_prompt = f"{MEDICINE_PROMPT.format(patient_info=full_patient_info, memory_context=memory_context)}"
            
            logging.info("Generating medicine and care suggestions...")
            medicine_suggestions = model_manager.generate(med_prompt, max_new_tokens=400, temperature=0.7)
            logging.info(f"Medicine suggestions generated: {medicine_suggestions[:100]}...")
            
            # Combine all responses
            final_response = (
                f"**📋 COMPREHENSIVE MEDICAL SUMMARY:**\n{summary}\n\n"
                f"**💊 MEDICATION AND HOME CARE SUGGESTIONS:**\n{medicine_suggestions}\n\n"
                f"**📊 PATIENT CONTEXT SUMMARY:**\n{patient_summary}\n\n"
                f"**⚠️ IMPORTANT DISCLAIMER:** This is AI-generated advice for informational purposes only. "
                f"This assessment is not a substitute for professional medical advice, diagnosis, or treatment. "
                f"Please consult a licensed healthcare provider for proper medical evaluation and personalized care."
            )
            
            memory_manager.add_interaction(message, final_response)
            chat_history.append((message, final_response))
            
            return "", chat_history
    
    except Exception as e:
        logging.error(f"Error in respond function: {e}")
        error_message = (
            "I apologize, but I encountered an error processing your request. "
            "This could be due to API connectivity issues. Please try again in a moment. "
            f"Error details: {str(e)}"
        )
        chat_history.append((message, error_message))
        return "", chat_history


def reset_chat():
    """
    Reset the chat session and start a new consultation
    """
    global conversation_turns
    conversation_turns = 0
    memory_manager.reset_session()
    
    reset_msg = (
        "🔄 **New consultation started.**\n\n"
        "Hello! I'm your virtual medical assistant with voice capabilities. I'm here to help gather information about your health concerns.\n\n"
        "Please tell me:\n"
        "- Your name and age\n"
        "- What symptoms or health concerns you're experiencing\n\n"
        "💡 **Voice Features Available:**\n"
        "- 🎤 Click the microphone to speak your symptoms\n"
        "- 🔊 Click any message in the chat to hear it aloud\n"
        "- 🗣️ Use 'Conversation Mode' for hands-free chat\n\n"
        "I'll ask follow-up questions to better understand your situation."
    )
    
    logging.info("Session reset. New consultation started.")
    return [(None, reset_msg)], ""