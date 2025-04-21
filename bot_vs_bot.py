import streamlit as st
import os
import io
import base64
import json
from dotenv import load_dotenv
from PIL import Image
import requests # To fetch images from URLs if needed (though not currently used)
import traceback # For printing full tracebacks on unexpected errors

# --- Page Config (MUST be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="Bot vs. Bot")

# --- Configuration and Initialization ---
# Load API keys from .env file
load_dotenv()

# --- Attempt to Import API Client Libraries ---
try:
    from anthropic import Anthropic, APIError as AnthropicAPIError
except ImportError:
    Anthropic = None; AnthropicAPIError = None
try:
    from openai import OpenAI, APIError as OpenAIAPIError
except ImportError:
    OpenAI = None; OpenAIAPIError = None
try:
    # Use google-genai package
    from google import genai
    from google.genai import types as google_types
    from google.api_core.exceptions import GoogleAPIError
except ImportError:
    genai = None; google_types = None; GoogleAPIError = None


# --- Helper Functions --- (Remain the same)

def get_image_bytes(uploaded_file):
    """Reads bytes from Streamlit UploadedFile."""
    if uploaded_file:
        return uploaded_file.getvalue()
    return None

def image_to_base64(image_bytes, mime_type):
    """Converts image bytes to base64 string, handling OpenAI's data URI format."""
    if image_bytes and mime_type:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        if mime_type.startswith("openai_"):
             original_mime = mime_type.split('_', 1)[1]
             return f"data:{original_mime};base64,{encoded}"
        else:
             return encoded
    return None

def get_mime_type(uploaded_file):
    """Gets the MIME type of the uploaded file."""
    if uploaded_file:
        return uploaded_file.type
    return None

# --- API Client Initialization (Cached) ---

@st.cache_resource # Cache the clients for the session duration
def initialize_clients():
    """Initializes API clients based on available libraries and .env keys."""
    clients = {"anthropic": None, "openai": None, "gemini": None}
    keys = {
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "vertex_project": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "vertex_location": os.getenv("GOOGLE_CLOUD_LOCATION"),
        "use_vertex": os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true",
    }
    missing_keys = []
    error_messages = []
    # (Initialization logic remains the same - uses genai.Client correctly)
    if Anthropic:
        if keys["anthropic"]:
            try: clients["anthropic"] = Anthropic(api_key=keys["anthropic"])
            except Exception as e: error_messages.append(f"Anthropic Init Error: {e}")
        else: missing_keys.append("ANTHROPIC_API_KEY")
    else:
        if keys["anthropic"]: error_messages.append("Anthropic lib not found (pip install anthropic)")
    if OpenAI:
        if keys["openai"]:
            try: clients["openai"] = OpenAI(api_key=keys["openai"])
            except Exception as e: error_messages.append(f"OpenAI Init Error: {e}")
        else: missing_keys.append("OPENAI_API_KEY")
    else:
        if keys["openai"]: error_messages.append("OpenAI lib not found (pip install openai)")
    if genai:
        try:
            if keys["use_vertex"]:
                if keys["vertex_project"] and keys["vertex_location"]:
                    clients["gemini"] = genai.Client(project=keys["vertex_project"], location=keys['vertex_location'])
                else:
                    missing_keys.append("GOOGLE_CLOUD_PROJECT/LOCATION (Vertex)")
                    clients["gemini"] = None
            elif keys["google"]:
                clients["gemini"] = genai.Client(api_key=keys["google"])
            else:
                missing_keys.append("GOOGLE_API_KEY (or Vertex config)")
                clients["gemini"] = None
        except Exception as e:
             error_messages.append(f"Google GenAI Init Error: {e}")
             clients["gemini"] = None
    elif "GOOGLE_API_KEY" not in missing_keys and "GOOGLE_CLOUD_PROJECT" not in missing_keys:
         error_messages.append("Google GenAI lib not found (pip install google-genai)")

    return clients, missing_keys, error_messages

# --- Initialize Clients and Display Init Status ---
_clients, _missing_keys, _error_messages = initialize_clients()
# Display init errors/warnings after set_page_config
if _missing_keys: st.warning(f"Missing API keys in .env: {', '.join(_missing_keys)}. Corresponding models may be unavailable.")
if _error_messages:
    for err in _error_messages: st.error(err)


# --- API Request Formatting Functions --- (Remain the same)

def format_anthropic_messages(history, system_prompt, user_message, image_bytes, mime_type):
    messages = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg: messages.append(msg)
    content_blocks = []
    if user_message: content_blocks.append({"type": "text", "text": user_message})
    if image_bytes and mime_type:
        base64_image = image_to_base64(image_bytes, mime_type)
        if base64_image: content_blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": base64_image}})
    if content_blocks: messages.append({"role": "user", "content": content_blocks})
    elif not history or history[-1].get("role") != "user": messages.append({"role": "user", "content": ""})
    return messages, system_prompt

def format_openai_messages(history, system_prompt, user_message, image_bytes, mime_type):
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    for msg in history:
        if isinstance(msg, dict): messages.append(msg)
    content_parts = []
    if user_message: content_parts.append({"type": "text", "text": user_message})
    if image_bytes and mime_type:
         base64_data_uri = image_to_base64(image_bytes, f"openai_{mime_type}")
         if base64_data_uri: content_parts.append({"type": "image_url", "image_url": {"url": base64_data_uri, "detail": "auto"}})
    if content_parts: messages.append({"role": "user", "content": content_parts})
    elif not history or history[-1]['role'] != 'user': messages.append({"role": "user", "content": ""})
    return messages

def format_gemini_messages(history, system_prompt, user_message, image_bytes, mime_type):
    """Formats messages for Google Gemini API generate_content method."""
    contents = []
    if google_types:
        for msg_data in history:
            if isinstance(msg_data, google_types.Content):
                contents.append(msg_data)
            elif isinstance(msg_data, dict) and "role" in msg_data and "parts" in msg_data:
                try:
                    reconstructed_parts = []
                    for part_data in msg_data["parts"]:
                         if isinstance(part_data, dict) and "text" in part_data:
                              reconstructed_parts.append(google_types.Part(text=part_data["text"]))
                    if reconstructed_parts:
                         contents.append(google_types.Content(role=msg_data["role"], parts=reconstructed_parts))
                except Exception as e: print(f"Warning: Could not reconstruct Gemini history item: {e} - {msg_data}")
            else: print(f"Warning: Skipping invalid Gemini history item: {type(msg_data)}")

        user_parts = []
        if user_message: user_parts.append(google_types.Part(text=user_message))
        if image_bytes and mime_type:
            try: user_parts.append(google_types.Part.from_data(data=image_bytes, mime_type=mime_type))
            except Exception as e: st.warning(f"Could not format image for Gemini API call: {e}")
        if user_parts: contents.append(google_types.Content(role="user", parts=user_parts))
        elif not user_parts and (not contents or contents[-1].role != "user"): contents.append(google_types.Content(role="user", parts=[]))
    return contents, system_prompt


# --- Streamlit App UI Setup ---
st.title("🤖 Bot vs. Bot Arena 🥊")
st.caption("Compare responses from different AI models side-by-side.")

# --- Model Selection & Config ---
# *** UPDATED Model Lists based on user table ***
ANTHROPIC_MODELS = [
    "claude-3-7-sonnet-20250219",
    "claude-3.5-haiku-20240620", # Assuming standard ID for 3.5 Haiku
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
] if _clients.get("anthropic") else []
OPENAI_MODELS = [
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-3.5-turbo",
    "gpt-3.5-instruct", "o1", "o1-mini", "o1-pro", "o3", "o3-mini",
    "o3-mini-high", "o4-mini", "o4-mini-high"
] if _clients.get("openai") else []
GEMINI_MODELS = [
    # Use simpler/latest tags first for broader compatibility
    "gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro",
    # Add specific/preview/vertex tags from table
    "gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17",
    "models/gemini-2.0-flash", "models/gemini-2.0-flash-lite",
    "models/gemini-1.5-pro", "models/gemini-1.5-flash"
] if _clients.get("gemini") else []

ALL_MODELS = {"Anthropic": ANTHROPIC_MODELS, "OpenAI": OPENAI_MODELS, "Google Gemini": GEMINI_MODELS}
PROVIDER_KEY_MAP = {"Anthropic": "anthropic", "OpenAI": "openai", "Google Gemini": "gemini"}

# --- Session State Initialization ---
default_provider1 = next(iter(PROVIDER_KEY_MAP.keys()), None)
default_provider2 = list(PROVIDER_KEY_MAP.keys())[1] if len(PROVIDER_KEY_MAP) > 1 else default_provider1
if "history1" not in st.session_state: st.session_state.history1 = []
if "provider1" not in st.session_state: st.session_state.provider1 = default_provider1
if "model1" not in st.session_state: st.session_state.model1 = ALL_MODELS.get(st.session_state.provider1, [None])[0]
if "history2" not in st.session_state: st.session_state.history2 = []
if "provider2" not in st.session_state: st.session_state.provider2 = default_provider2
if "model2" not in st.session_state: st.session_state.model2 = ALL_MODELS.get(st.session_state.provider2, [None])[0]
if "user_input1_text" not in st.session_state: st.session_state.user_input1_text = ""
if "user_input2_text" not in st.session_state: st.session_state.user_input2_text = ""
if "system_prompt" not in st.session_state: st.session_state.system_prompt = ""
if "temperature" not in st.session_state: st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state: st.session_state.max_tokens = 1024

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    st.subheader("Bot 1 Setup")
    provider_options = list(ALL_MODELS.keys())
    available_provider_options = [p for p in provider_options if _clients.get(PROVIDER_KEY_MAP.get(p))]
    if not available_provider_options: st.error("No API clients initialized."); st.stop()
    try: provider1_idx = available_provider_options.index(st.session_state.provider1)
    except ValueError: provider1_idx = 0
    provider1_select = st.selectbox("Select Bot 1 Provider", available_provider_options, key="provider1_select_widget", index=provider1_idx)
    model1_select_options = ALL_MODELS.get(provider1_select, [])
    model1_select = None
    if model1_select_options:
        try: current_model1_index = model1_select_options.index(st.session_state.model1)
        except ValueError: current_model1_index = 0
        model1_select = st.selectbox("Select Bot 1 Model", model1_select_options, key="model1_select_widget", index=current_model1_index)
    else: st.warning(f"No models available for {provider1_select}")

    st.subheader("Bot 2 Setup")
    provider2_default_index = (available_provider_options.index(provider1_select) + 1) % len(available_provider_options) if len(available_provider_options) > 1 else 0
    try: provider2_idx = available_provider_options.index(st.session_state.provider2)
    except ValueError: provider2_idx = provider2_default_index
    provider2_select = st.selectbox("Select Bot 2 Provider", available_provider_options, key="provider2_select_widget", index=provider2_idx)
    model2_select_options = ALL_MODELS.get(provider2_select, [])
    model2_select = None
    if model2_select_options:
         try: current_model2_index = model2_select_options.index(st.session_state.model2)
         except ValueError: current_model2_index = 0
         model2_select = st.selectbox("Select Bot 2 Model", model2_select_options, key="model2_select_widget", index=current_model2_index)
    else: st.warning(f"No models available for {provider2_select}")

    st.divider()
    st.subheader("System Prompt")
    system_prompt_input = st.text_area("System Prompt (common)", value=st.session_state.system_prompt, key="system_prompt_widget", height=100)
    st.divider()
    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, value=st.session_state.temperature, key="temperature_widget")
    max_tokens = st.number_input("Max Tokens", min_value=50, max_value=8192, value=st.session_state.max_tokens, step=128, key="max_tokens_widget")
    st.divider()
    if st.button("Clear All Histories"):
        st.session_state.history1 = []; st.session_state.history2 = []
        st.session_state.user_input1_text = ""; st.session_state.user_input2_text = ""
        st.rerun()

# --- Update session state based on sidebar selections ---
history_cleared = False
if st.session_state.provider1 != provider1_select or st.session_state.model1 != model1_select:
    st.session_state.history1 = []
    st.session_state.provider1 = provider1_select; st.session_state.model1 = model1_select
    if model1_select: st.toast(f"Bot 1: {provider1_select}/{model1_select}. History cleared.", icon="🧹"); history_cleared = True
if st.session_state.provider2 != provider2_select or st.session_state.model2 != model2_select:
    st.session_state.history2 = []
    st.session_state.provider2 = provider2_select; st.session_state.model2 = model2_select
    if model2_select: st.toast(f"Bot 2: {provider2_select}/{model2_select}. History cleared.", icon="🧹"); history_cleared = True
st.session_state.system_prompt = st.session_state.system_prompt_widget
st.session_state.temperature = st.session_state.temperature_widget
st.session_state.max_tokens = st.session_state.max_tokens_widget
if history_cleared: st.rerun()


# --- Main Area Layout ---
col1, col2 = st.columns(2)

# --- API Call and Streaming Logic (Function Definition) ---
def call_api_and_stream(bot_index, client, provider, model_id, history, system_prompt, user_message, image_bytes, mime_type, temp, max_tok, column_display):
    """Calls the appropriate API and streams the response TO A SPECIFIC COLUMN."""
    with column_display: placeholder = st.empty()
    full_response = ""; error_message = None; assistant_response_obj = None

    _AnthropicAPIError = AnthropicAPIError if AnthropicAPIError else Exception
    _OpenAIAPIError = OpenAIAPIError if OpenAIAPIError else Exception
    _GoogleAPIError = GoogleAPIError if GoogleAPIError else Exception

    try:
        # --- API Call Logic ---
        if provider == "Anthropic":
            if not client: raise ValueError("Anthropic client not initialized.")
            messages, sys_prompt = format_anthropic_messages(history, system_prompt, user_message, image_bytes, mime_type)
            with client.messages.stream(model=model_id,max_tokens=max_tok,temperature=temp,messages=messages,system=sys_prompt if sys_prompt else None) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            assistant_response_obj = stream.get_final_message()

        elif provider == "OpenAI":
            if not client: raise ValueError("OpenAI client not initialized.")
            messages = format_openai_messages(history, system_prompt, user_message, image_bytes, mime_type)
            stream = client.chat.completions.create(model=model_id,messages=messages,temperature=temp,max_tokens=max_tok,stream=True)
            collected_chunks = []; finish_reason_val = "unknown"
            for chunk in stream:
                collected_chunks.append(chunk)
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content; full_response += content_piece
                    placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                if chunk.choices and chunk.choices[0].finish_reason: finish_reason_val = chunk.choices[0].finish_reason
            assistant_response_obj = {"role": "assistant", "content": full_response, "finish_reason": finish_reason_val}

        elif provider == "Google Gemini":
             if not client or not google_types: raise ValueError("Google GenAI client/types not initialized.")
             contents, sys_prompt_arg = format_gemini_messages(history, system_prompt, user_message, image_bytes, mime_type)

             # NOTE: Config is temporarily removed from streaming call due to previous TypeError.
             # Default settings will be used by the API for temp, max_tokens, safety. System prompt ignored for streaming.
             # safety_settings = { ... }
             # generation_config = google_types.GenerateContentConfig(...)

             print(f"\nDEBUG (Bot {bot_index}): Calling Gemini model '{model_id}' via client.models.generate_content_stream (NO config/safety)")
             stream = client.models.generate_content_stream(
                 model=model_id,
                 contents=contents
                 # generation_config=generation_config, # Removed
                 # safety_settings=safety_settings    # Removed
             )

             # *** REMOVED aggregated_response_content list - wasn't working correctly ***
             usage_metadata = None; finish_reason_final = google_types.FinishReason.FINISH_REASON_UNSPECIFIED; prompt_feedback = None
             print(f"DEBUG (Bot {bot_index}): Starting stream iteration...")

             chunk_count = 0
             for chunk in stream:
                  chunk_count += 1
                  print(f"\n--- Gemini Chunk {chunk_count} (Bot {bot_index}) ---")
                  chunk_text_found = None
                  try:
                      print(f"Chunk Type: {type(chunk)}")
                      direct_text = getattr(chunk, 'text', None)
                      print(f"Direct Text (chunk.text): {direct_text}")
                      chunk_parts = getattr(chunk, 'parts', None)
                      print(f"Parts (chunk.parts): {chunk_parts}")
                      # if chunk_parts: # Debug print parts content if needed
                      #     for i, part in enumerate(chunk_parts):
                      #         part_text = getattr(part, 'text', None); print(f"  Part {i}: Type={type(part)}, Text='{part_text}'")

                      if direct_text: chunk_text_found = direct_text
                      elif chunk_parts:
                          parts_text = "";
                          for part in chunk_parts:
                               if hasattr(part, 'text') and part.text: parts_text += part.text
                          if parts_text: chunk_text_found = parts_text

                      if chunk_text_found:
                          print(f"DEBUG: Found text in chunk: '{chunk_text_found[:50]}...'")
                          full_response += chunk_text_found # Accumulate text
                          placeholder.markdown(full_response + "▌", unsafe_allow_html=True) # Update UI
                      else: print("DEBUG: No text found in this chunk via .text or .parts[].text")

                      # --- Check for finish reason / safety ---
                      if hasattr(chunk, 'candidates') and chunk.candidates:
                           candidate = chunk.candidates[0]
                           # Commented out noisy debug print for safety ratings if None
                           # print(f"DEBUG: Candidate Info: FinishReason={getattr(candidate, 'finish_reason', 'N/A')}, Safety={getattr(candidate, 'safety_ratings', 'N/A')}")
                           print(f"DEBUG: Candidate Info: FinishReason={getattr(candidate, 'finish_reason', 'N/A')}")
                           if hasattr(candidate, 'finish_reason') and candidate.finish_reason != google_types.FinishReason.FINISH_REASON_UNSPECIFIED:
                               finish_reason_final = candidate.finish_reason
                               if finish_reason_final != google_types.FinishReason.STOP:
                                    finish_reason_name = google_types.FinishReason(finish_reason_final).name
                                    if finish_reason_final == google_types.FinishReason.SAFETY:
                                        block_reason = "Blocked: Safety";
                                        if hasattr(candidate, 'safety_ratings'): block_reason += f" ({', '.join([f'{r.category.name}: {r.probability.name}' for r in candidate.safety_ratings])})"
                                        error_message = block_reason
                                    else: error_message = f"Stopped: {finish_reason_name}"
                                    print(f"DEBUG: Stream stopped early: {error_message}")
                                    full_response += f"\n\n*[{error_message}]*"
                                    placeholder.markdown(full_response, unsafe_allow_html=True); break
                      # --- Capture usage metadata & prompt feedback ---
                      if hasattr(chunk, 'usage_metadata'):
                           usage_metadata = chunk.usage_metadata; print(f"DEBUG: Usage Metadata found: {usage_metadata}")
                      if hasattr(chunk, 'prompt_feedback'):
                           prompt_feedback = chunk.prompt_feedback
                           # Commented out noisy debug print for prompt feedback if None
                           # print(f"DEBUG: Prompt Feedback found: {prompt_feedback}")
                           if prompt_feedback: print(f"DEBUG: Prompt Feedback found: {prompt_feedback}")


                  except Exception as debug_e: print(f"ERROR during chunk processing/debugging: {debug_e}")

             print(f"DEBUG (Bot {bot_index}): Finished stream iteration after {chunk_count} chunks.")

             # *** FIXED: Construct final response object using accumulated full_response ***
             final_candidate_parts = []
             if full_response: # If we accumulated any text
                 final_candidate_parts.append(google_types.Part(text=full_response))
             # Add handling here if other part types (e.g., function calls) were aggregated

             final_candidate = None
             # Create candidate only if there are parts or a finish reason was set
             if final_candidate_parts or finish_reason_final != google_types.FinishReason.FINISH_REASON_UNSPECIFIED:
                  final_candidate = google_types.Candidate(
                      content=google_types.Content(role="model", parts=final_candidate_parts),
                      finish_reason=finish_reason_final
                  )
             assistant_response_obj = google_types.GenerateContentResponse(
                 candidates=[final_candidate] if final_candidate else [],
                 prompt_feedback=prompt_feedback,
                 usage_metadata=usage_metadata
             )
             print(f"DEBUG (Bot {bot_index}): Constructed final response object with parts derived from full_response.")

        else:
            raise NotImplementedError(f"Provider {provider} not implemented.")

        # Update placeholder only if stream finished without critical API error
        if not error_message or "Stopped:" in error_message or "Blocked:" in error_message :
             placeholder.markdown(full_response, unsafe_allow_html=True)

    except (_AnthropicAPIError, _OpenAIAPIError, _GoogleAPIError) as api_err:
        error_message = f"API Error ({provider}): {type(api_err).__name__} - {api_err}"
        print(f"ERROR (Bot {bot_index}): Caught API Error - {error_message}")
        with column_display: st.error(error_message)
        full_response = None
    except Exception as e:
        error_message = f"Unexpected Error ({provider}): {type(e).__name__} - {e}"
        print(f"ERROR (Bot {bot_index}): Caught Unexpected Error - {error_message}")
        traceback.print_exc()
        with column_display: st.error(error_message)
        full_response = None

    return full_response, assistant_response_obj, error_message


# --- Function to display history --- (Remains the same)
def display_history(history, provider):
    for i, msg in enumerate(history):
        role = "unknown"
        if hasattr(msg, 'role'): role = msg.role
        elif isinstance(msg, dict): role = msg.get('role','unknown')
        display_role = "assistant" if role == "model" or role == "assistant" else "user"
        with st.chat_message(name=display_role): # No key here
            content = None; parts = None
            if isinstance(msg, dict): content = msg.get("content")
            elif hasattr(msg, 'parts'): parts = msg.parts
            elif hasattr(msg, 'content') and isinstance(msg.content, list): parts = msg.content
            has_displayed_content = False
            if content is not None:
                if isinstance(content, str):
                    if content: st.markdown(content, unsafe_allow_html=True); has_displayed_content = True
                elif isinstance(content, list):
                    for item_idx, item in enumerate(content):
                        item_type = item.get("type")
                        if item_type == "text":
                            if item.get("text"): st.markdown(item.get("text", ""), unsafe_allow_html=True); has_displayed_content = True
                        elif item_type in ["image", "image_url"]:
                             st.markdown("🖼️ *[User image input]*"); has_displayed_content = True
                    if not has_displayed_content and any(item.get("type") in ["image", "image_url"] for item in content):
                         st.markdown("🖼️ *[Image only message]*"); has_displayed_content = True
            elif parts is not None:
                 # Add a check for empty parts list before iterating
                 if not parts:
                     print(f"DEBUG (Hist {i}, {display_role}, {provider}): Parts list is empty.")
                 for part_idx, part in enumerate(parts):
                    # Ensure part is not None and has text attribute
                    if part and hasattr(part, 'text') and part.text:
                        part_text_content = part.text
                        print(f"  DEBUG Display Part {part_idx} Text: '{part_text_content[:100]}...'") # Added print
                        is_placeholder = (part_text_content == "🖼️ *[User image input]*")
                        st.markdown(part_text_content, unsafe_allow_html=True)
                        has_displayed_content = True # Mark displayed if text part exists (even placeholder)
                    elif part and (hasattr(part, 'inline_data') or hasattr(part, 'file_data')):
                         st.markdown("🖼️ *[User image/data input]*"); has_displayed_content = True
                    elif part and hasattr(part, 'type') and part.type == 'image':
                         st.markdown("🖼️ *[User image input]*"); has_displayed_content = True
                 # Display "Image only" message only if no text parts were found but other parts existed
                 if not has_displayed_content and parts and any(not (hasattr(p,'text') and p.text) for p in parts):
                      st.markdown("🖼️ *[Non-text message content]*"); has_displayed_content = True


            if not has_displayed_content:
                print(f"DEBUG (Hist {i}, {display_role}, {provider}): No displayable content found in msg, showing [Empty message]. Msg object: {msg}")
                st.markdown("*[Empty message]*")


# --- Input Area Below Columns ---
st.divider()
input_col1, input_col2 = st.columns(2)
with input_col1:
    st.text_area("Message for Bot 1", key="user_input1_widget", height=100, value=st.session_state.user_input1_text)
    uploaded_image1 = st.file_uploader("Upload Image (Bot 1)", type=["png", "jpg", "jpeg", "webp", "gif"], key="uploader1")
with input_col2:
    st.text_area("Message for Bot 2", key="user_input2_widget", height=100, value=st.session_state.user_input2_text)
    uploaded_image2 = st.file_uploader("Upload Image (Bot 2)", type=["png", "jpg", "jpeg", "webp", "gif"], key="uploader2")

send_button_pressed = st.button("✉️ Send to Both Bots", use_container_width=True, key="send_button")
st.divider()


# --- Display Columns Setup ---
with col1:
    provider1_name = st.session_state.get('provider1','N/A')
    model1_name = st.session_state.get('model1','N/A')
    st.subheader(f"Bot 1: {provider1_name} ({model1_name})")
    current_history1 = st.session_state.get("history1", [])
    display_history(current_history1, provider1_name)
    bot1_display_area = st.container() # Area for new messages
with col2:
    provider2_name = st.session_state.get('provider2','N/A')
    model2_name = st.session_state.get('model2','N/A')
    st.subheader(f"Bot 2: {provider2_name} ({model2_name})")
    current_history2 = st.session_state.get("history2", [])
    display_history(current_history2, provider2_name)
    bot2_display_area = st.container() # Area for new messages


# --- Processing Logic (Triggered by Button Click) ---
if send_button_pressed:
    # Read text input from the widget's state via its key
    user_input1_text = st.session_state.get("user_input1_widget", "")
    user_input2_text = st.session_state.get("user_input2_widget", "")

    # Update the separate state variables (used for binding 'value')
    st.session_state.user_input1_text = user_input1_text
    st.session_state.user_input2_text = user_input2_text

    # Read images directly from uploaders
    uploader1_state = st.session_state.get("uploader1")
    uploader2_state = st.session_state.get("uploader2")
    image_bytes1 = get_image_bytes(uploader1_state)
    mime_type1 = get_mime_type(uploader1_state)
    image_bytes2 = get_image_bytes(uploader2_state)
    mime_type2 = get_mime_type(uploader2_state)

    # Read other parameters from session state
    model1_runtime = st.session_state.get('model1'); provider1_runtime = st.session_state.get('provider1')
    model2_runtime = st.session_state.get('model2'); provider2_runtime = st.session_state.get('provider2')
    system_prompt_runtime = st.session_state.get('system_prompt', "")
    temperature_runtime = st.session_state.get('temperature', 0.7)
    max_tokens_runtime = st.session_state.get('max_tokens', 1024)

    valid_input1 = user_input1_text or image_bytes1; valid_input2 = user_input2_text or image_bytes2
    models_selected = model1_runtime and model2_runtime

    if not models_selected: st.error("Please select models for both bots in the sidebar.")
    elif not valid_input1 and not valid_input2: st.warning("Please enter a message or upload an image for at least one bot.")
    else:
        history_updated = False # Flag to check if we need to rerun

        # --- Process Bot 1 ---
        if valid_input1 and model1_runtime:
            client1 = _clients.get(PROVIDER_KEY_MAP.get(provider1_runtime))
            if client1:
                with bot1_display_area:
                    with st.chat_message("user"): # No key
                        if image_bytes1: st.image(image_bytes1, width=150)
                        if user_input1_text: st.markdown(user_input1_text, unsafe_allow_html=True)
                        elif image_bytes1: st.markdown("🖼️ *[Image only]*")
                with st.spinner(f"Bot 1 ({provider1_runtime}) thinking..."):
                    response_text1, assistant_obj1, error1 = call_api_and_stream(1, client1, provider1_runtime, model1_runtime, current_history1, system_prompt_runtime, user_input1_text, image_bytes1, mime_type1, temperature_runtime, max_tokens_runtime, bot1_display_area)
                if response_text1 is not None and not error1:
                    user_msg1_to_store = None; assistant_msg1_to_store = None
                    if provider1_runtime in ["Anthropic", "OpenAI"]:
                        user_content = []
                        if user_input1_text: user_content.append({"type": "text", "text": user_input1_text})
                        if image_bytes1: user_content.append({"type": "image"}) # Placeholder
                        user_msg1_to_store = {"role": "user", "content": user_content}
                        if provider1_runtime == "Anthropic" and assistant_obj1: assistant_msg1_to_store = {"role": assistant_obj1.role, "content": [block.model_dump() for block in assistant_obj1.content]}
                        elif provider1_runtime == "OpenAI" and assistant_obj1: assistant_msg1_to_store = assistant_obj1
                    elif provider1_runtime == "Google Gemini" and google_types and assistant_obj1:
                        user_parts = []
                        if user_input1_text: user_parts.append(google_types.Part(text=user_input1_text))
                        if image_bytes1: user_parts.append(google_types.Part(text="🖼️ *[User image input]*"))
                        user_msg1_to_store = google_types.Content(role="user", parts=user_parts)
                        # *** Store the Content object from the response candidate ***
                        if assistant_obj1.candidates: assistant_msg1_to_store = assistant_obj1.candidates[0].content
                        else: print(f"Warning (Bot 1 Gemini): No candidates found in final response object: {assistant_obj1}")

                    if user_msg1_to_store and assistant_msg1_to_store:
                         # Ensure assistant message is stored correctly for Gemini
                         if provider1_runtime == "Google Gemini" and not isinstance(assistant_msg1_to_store, google_types.Content):
                              print(f"Warning (Bot 1 Gemini): Assistant message not stored as Content object: {type(assistant_msg1_to_store)}")
                         else:
                              st.session_state.history1.append(user_msg1_to_store); st.session_state.history1.append(assistant_msg1_to_store)
                              history_updated = True
            else:
                with bot1_display_area: st.error(f"Client for Bot 1 ({provider1_runtime}) not available.")

        # --- Process Bot 2 (Sequentially) ---
        if valid_input2 and model2_runtime:
            client2 = _clients.get(PROVIDER_KEY_MAP.get(provider2_runtime))
            if client2:
                with bot2_display_area:
                    with st.chat_message("user"): # No key
                        if image_bytes2: st.image(image_bytes2, width=150)
                        if user_input2_text: st.markdown(user_input2_text, unsafe_allow_html=True)
                        elif image_bytes2: st.markdown("🖼️ *[Image only]*")
                with st.spinner(f"Bot 2 ({provider2_runtime}) thinking..."):
                    response_text2, assistant_obj2, error2 = call_api_and_stream(2, client2, provider2_runtime, model2_runtime, current_history2, system_prompt_runtime, user_input2_text, image_bytes2, mime_type2, temperature_runtime, max_tokens_runtime, bot2_display_area)
                if response_text2 is not None and not error2:
                    user_msg2_to_store = None; assistant_msg2_to_store = None
                    if provider2_runtime in ["Anthropic", "OpenAI"]:
                        user_content = []
                        if user_input2_text: user_content.append({"type": "text", "text": user_input2_text})
                        if image_bytes2: user_content.append({"type": "image"}) # Placeholder
                        user_msg2_to_store = {"role": "user", "content": user_content}
                        if provider2_runtime == "Anthropic" and assistant_obj2: assistant_msg2_to_store = {"role": assistant_obj2.role, "content": [block.model_dump() for block in assistant_obj2.content]}
                        elif provider2_runtime == "OpenAI" and assistant_obj2: assistant_msg2_to_store = assistant_obj2
                    elif provider2_runtime == "Google Gemini" and google_types and assistant_obj2:
                        user_parts = []
                        if user_input2_text: user_parts.append(google_types.Part(text=user_input2_text))
                        if image_bytes2: user_parts.append(google_types.Part(text="🖼️ *[User image input]*"))
                        user_msg2_to_store = google_types.Content(role="user", parts=user_parts)
                        # *** Store the Content object from the response candidate ***
                        if assistant_obj2.candidates: assistant_msg2_to_store = assistant_obj2.candidates[0].content
                        else: print(f"Warning (Bot 2 Gemini): No candidates found in final response object: {assistant_obj2}")

                    if user_msg2_to_store and assistant_msg2_to_store:
                         # Ensure assistant message is stored correctly for Gemini
                         if provider2_runtime == "Google Gemini" and not isinstance(assistant_msg2_to_store, google_types.Content):
                              print(f"Warning (Bot 2 Gemini): Assistant message not stored as Content object: {type(assistant_msg2_to_store)}")
                         else:
                              st.session_state.history2.append(user_msg2_to_store); st.session_state.history2.append(assistant_msg2_to_store)
                              history_updated = True
            else:
                with bot2_display_area: st.error(f"Client for Bot 2 ({provider2_runtime}) not available.")

        # --- Post-processing ---
        # Clear the separate state variables, which will clear the text_area widgets via value binding
        st.session_state.user_input1_text = ""
        st.session_state.user_input2_text = ""

        if history_updated:
             # Rerun to clear text areas and update history display definitively
             st.rerun()

# Add some padding at the bottom for spacing
st.markdown("<br><br>", unsafe_allow_html=True)