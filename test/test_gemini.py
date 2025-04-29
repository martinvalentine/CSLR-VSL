# Using the standard library import
import google.generativeai as genai
import os
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your actual API key or ensure it's set as an environment variable
GOOGLE_API_KEY = ""

if not GOOGLE_API_KEY:
    logging.error("Google API Key not found. Set the GOOGLE_API_KEY environment variable or replace the placeholder.")
    # Raise an error to stop execution if the key is missing.
    raise ValueError("Google API Key not configured.")

# Configure the library globally (standard approach)
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Initialize the model globally
    # Using a suitable model for text generation
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    logging.info("Google Generative AI configured successfully with gemini-1.5-flash-latest.")
except Exception as e:
    logging.error(f"Failed to configure Google Generative AI or initialize model: {e}")
    # Re-raise the exception after logging
    raise

def build_vsl_to_vietnamese_prompt(vsl_gloss_tuple):
    """
    Constructs the few-shot prompt for VSL to Vietnamese conversion,
    using insights from linguistic analysis of VSL structures.
    The prompt content is directly included here, matching the latest version
    from the 'vsl_translation_prompt' immersive artifact.
    """
    vsl_input_string = " ".join(vsl_gloss_tuple)
    logging.info(f"Building prompt for VSL gloss: {vsl_input_string}")

    # Few-Shot Examples & Instructions
    prompt = f"""
You are an expert assistant translating Vietnamese Sign Language (VSL) gloss sequences into natural, grammatically correct Vietnamese sentences. You understand key structural differences based on linguistic analysis, recognizing that VSL is a distinct language with its own grammar, not simply a manual representation of spoken Vietnamese. You are aware of typical VSL structures such as S-O-V (Subject-Object-Verb) order, the common final position of negation and question words, and the use of topic-comment structures. You also understand that VSL can sometimes simplify or synthesize multiple concepts into a single sign compared to spoken Vietnamese.

**Important Note on Tense:** VSL often relies heavily on context (like time adverbials - HÔM-QUA, NGÀY-MAI) or explicit tense glosses ('SẼ', 'ĐÃ'). Explicit tense glosses may be present, often at the end of a sentence, but infer the appropriate tense for the Vietnamese translation based on the available glosses (especially time words like HÔM-QUA, NGÀY-MAI, BÂY-GIỜ). If a tense gloss *is* present, use it accordingly.

**Important Note on Questions without Question Words:** For questions that do not have an explicit question word (like GÌ, BAO_NHIÊU, ĐÂU), the interrogative nature is often indicated by the structure of the gloss sequence. The presence of a subject like "BẠN" (You) or a final particle like "KHÔNG" (which can function as a yes/no question marker at the end, distinct from its use in negation within the sentence) or a question mark (?) at the end of the sequence should be strong indicators that the sentence is a question.

**Important Note on Specificity:** Understand that some actions or concepts that might be represented by a single verb in spoken Vietnamese may require more specific signs in VSL depending on the object or context (e.g., different signs for "to open" depending on what is being opened).

Convert the provided VSL gloss sequence into Vietnamese, applying your knowledge of VSL structure and nuances.

Here are some examples reflecting VSL structure:

# Basic S+O+V (common in VSL)

VSL Glosses: TÔI MÈO THÍCH
Vietnamese: Tôi thích mèo.

VSL Glosses: CON_GÀ THÓC ĂN
Vietnamese: Con gà ăn thóc.

# Negation particle at the end

VSL Glosses: CON SỮA UỐNG CHƯA
Vietnamese: Con chưa uống sữa.
(Note: 'CHƯA' at the end functions as negation in this context)

VSL Glosses: TÔI CAM ĂN THÍCH KHÔNG
Vietnamese: Tôi không thích ăn cam.

# Question word at the end

VSL Glosses: BẠN TÊN GÌ
Vietnamese: Bạn tên gì?

VSL Glosses: BẠN TUỔI BAO_NHIÊU
Vietnamese: Bạn bao nhiêu tuổi?

VSL Glosses: BẠN CÙNG ĐI AI
Vietnamese: Bạn đi cùng với ai?

VSL Glosses: BẠN GIA\_ĐÌNH THÀNH_VIÊN BAO_NHIÊU?
Vietnamese: Gia đình bạn có bao nhiêu thành viên?

VSL Glosses: BẠN KẸO BAO_NHIÊU
Vietnamese: Bạn có bao nhiêu cái kẹo?

# Questions without explicit question words (indicated by structure/context in glosses)

VSL Glosses: BẠN ĂN RỒI_CHƯA
Vietnamese: Bạn ăn chưa?

VSL Glosses: BẠN BẬN
Vietnamese: Bạn bận không?

VSL Glosses: BẠN KHỎE
Vietnamese: Bạn khỏe không?

VSL Glosses: BẠN TRÀ THÍCH
Vietnamese: Bạn có thích trà không?

# Tense implied by time adverbial

VSL Glosses: HÔM-QUA TÔI PHIM XEM
Vietnamese: Hôm qua tôi xem phim.

VSL Glosses: NGÀY-MAI BẠN ĐI ĐÂU
Vietnamese: Ngày mai bạn đi đâu?

# Place adverbial often at the beginning

VSL Glosses: Ở-TRƯỜNG TÔI BẠN-THÂN CÓ
Vietnamese: Ở trường tôi có bạn thân.

# Other examples

VSL Glosses: CHA CON THƯƠNG
Vietnamese: Cha thương con.

VSL Glosses: CHÚC NGỦ
Vietnamese: Chúc ngủ ngon.

VSL Glosses: HẸN GẶP Ở\_ĐÂY
Vietnamese: Hẹn gặp lại ở đây.
(Note: "Ở\_ĐÂY" is included in the translation as it's in the gloss, but "Hẹn gặp lại" is a common Vietnamese phrase).

VSL Glosses: HÔM_NAY TÔI VUI Ở\_ĐÂY
Vietnamese: Hôm nay tôi vui khi ở đây.

VSL Glosses: MỌI_NGƯỜI GIA\_ĐÌNH YÊU_THƯƠNG
Vietnamese: Mọi người trong gia đình yêu thương nhau.

Now, convert the following VSL gloss sequence:

VSL Glosses: {vsl_input_string}
Vietnamese:"""
    return prompt.strip()

def convert_vsl_to_vietnamese(vsl_gloss_tuple):
    """
    Connects to Gemini API (using the pre-configured model)
    and converts VSL gloss tuple to Vietnamese sentence.
    Includes error handling and response validation.
    """
    # Validate input type and content
    if not isinstance(vsl_gloss_tuple, tuple) or not all(isinstance(item, str) for item in vsl_gloss_tuple):
        logging.error(f"Invalid input type: Expected tuple of strings, got {type(vsl_gloss_tuple)}")
        return "Error: Input must be a tuple of strings."

    if not vsl_gloss_tuple:
        logging.warning("Input tuple is empty.")
        return "Error: Input tuple cannot be empty."

    # Build the prompt using the helper function
    prompt = build_vsl_to_vietnamese_prompt(vsl_gloss_tuple)

    logging.info("Sending prompt to Gemini API...")
    try:
        # Define safety settings for the generation
        # These settings are applied to the *response* content.
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        # Use the globally initialized model object to generate content
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            # Optional: generation_config allows controlling output parameters like temperature, max_output_tokens, etc.
            # For translation, a lower temperature might be preferred for more deterministic output.
            # generation_config=genai.types.GenerationConfig(temperature=0.5, max_output_tokens=100)
        )

        # --- Response Validation and Extraction ---
        # Check if the response contains any candidates
        if not response.candidates:
            logging.warning(f"Response generation failed or was blocked. Prompt feedback: {response.prompt_feedback}")
            # Safely extract finish_reason and safety_ratings if available
            finish_reason = 'N/A'
            if response.candidates: # Check again if candidates exist before accessing
                 if hasattr(response.candidates[0], 'finish_reason'):
                     finish_reason = response.candidates[0].finish_reason
                 else:
                     logging.warning("Finish reason attribute not found in candidate.")
            safety_ratings = response.prompt_feedback.safety_ratings if response.prompt_feedback else 'N/A'
            return f"Error: Response generation failed. Finish reason: {finish_reason}. Safety feedback: {safety_ratings}"
        # Check if the first candidate's content has parts and if those parts are not empty
        elif not hasattr(response.candidates[0].content, 'parts') or not response.candidates[0].content.parts:
             logging.warning(f"Response generated but content parts are missing or empty. Finish Reason: {response.candidates[0].finish_reason}")
             return f"Error: Response missing content. Finish Reason: {response.candidates[0].finish_reason}"

        # Extract the text from the response
        generated_text = ""
        try:
             # Access the text attribute directly from the response object
             generated_text = response.text.strip()
             logging.info(f"Translation successful: {generated_text}")
        except Exception as text_extract_error:
             logging.error(f"Failed to extract text from response: {text_extract_error}. Full response object: {response}")
             return f"Error: Could not extract text from response."

        return generated_text

    except Exception as e:
        # Catching potential API errors (authentication, network issues, rate limits)
        logging.error(f"API call failed: {e}")
        # Check for specific API key error (often raises google.api_core.exceptions.PermissionDenied)
        if "PERMISSION_DENIED" in str(e) or "API key not valid" in str(e):
             return "Error: Invalid API Key. Please check your configuration."
        # Return a generic error message for other API issues
        return f"Error: Failed to generate content due to an API issue - {e}"

if __name__ == "__main__":
    # Define a list of test VSL gloss sequences as tuples
    test_inputs = [
        ('BẠN', 'KHỎE'), # Example of a question without a question word
        ('BẠN', 'LỚP', 'BAO_NHIÊU'), # Example of a question with a question word
        ('TÔI', 'MÈO', 'THÍCH'), # Example of S-O-V structure
        ('HÔM-QUA', 'TÔI', 'PHIM', 'XEM'), # Example with time adverbial
        ('CON', 'SỮA', 'UỐNG', 'CHƯA'), # Example with negation at the end
        ('BẠN', 'BẬN'), # Another example of a question without a question word
        ('BẠN', 'TRÀ', 'THÍCH'), # Another example of a question without a question word
        ('BẠN', 'CƠM_GÀ', 'THÍCH'), # Another example of a question without a question word
    ]

    # Loop through test inputs and call the conversion function
    for idx, vsl_tuple in enumerate(test_inputs, 1):
        logging.info(f"--- [Prediction {idx}] ---")
        logging.info(f"Input VSL Gloss: {vsl_tuple}")
        result = convert_vsl_to_vietnamese(vsl_tuple)
        # Log the result whether it's a translation or an error message
        logging.info(f"Output Vietnamese: {result}")
        print("-" * 40) # Separator in console output

