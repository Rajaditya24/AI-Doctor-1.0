import google.generativeai as genai
from .config import GEMINI_API_KEY, GEMINI_MODEL
import logging

class ModelManager:
    def __init__(self):
        self.model = None
        
    def load(self):
        """Initialize Gemini model"""
        if self.model is not None:
            return
        
        try:
            # Configure Gemini API
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Initialize the model
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            
            logging.info(f"Successfully loaded Gemini model: {GEMINI_MODEL}")
            
        except Exception as e:
            logging.error(f"Error loading Gemini model: {e}")
            raise Exception(f"Failed to initialize Gemini API. Please check your API key. Error: {e}")
    
    def generate(self, prompt, max_new_tokens=1000, temperature=0.7, top_p=0.9):
        """
        Generate response using Gemini API
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum tokens to generate (Gemini uses max_output_tokens)
            temperature: Controls randomness (0.0 to 1.0)
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text response
        """
        self.load()
        
        try:
            # Configure generation parameters
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_new_tokens,
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error processing your request. Please try again or rephrase your question."