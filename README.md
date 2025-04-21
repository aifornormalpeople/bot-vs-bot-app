# Bot vs. Bot Arena 🤖🥊

Welcome to the Bot vs. Bot Arena! This Streamlit application lets you pit different Large Language Models (LLMs) against each other in a side-by-side comparison. See how models from Anthropic (Claude), OpenAI (GPT), and Google (Gemini) respond to the same prompts and context.

## Features

* **Side-by-Side Comparison:** Select two different models (one for each "Bot") to compare their responses directly.
* **Provider Selection:** Choose models from Anthropic, OpenAI, and Google.
* **System Prompt:** Set a common system prompt to guide the behavior of both selected models.
* **Individual Inputs:** Provide unique text prompts and optionally upload images for each bot independently.
* **Conversation History:** Track the conversation history separately for each bot.
* **Streaming Responses:** Watch the model responses generate in real-time.
* **Parameter Tuning:** Adjust Temperature and Max Tokens via the sidebar.

## How to Use the Arena

1.  **Configure Bots (Sidebar):**
    * Use the dropdown menus under "Bot 1 Setup" and "Bot 2 Setup" to select the Provider (Anthropic, OpenAI, Google Gemini) and the specific Model you want to test for each side.
    * **Note:** Only providers/models for which you have provided valid API keys in your `.env` file and have the necessary libraries installed will be available.
2.  **Set the Scene (Sidebar):**
    * Enter instructions or context into the "System Prompt (common)" text area. This prompt will be sent to *both* models at the beginning of their (currently stateless) interaction context for each turn.
    * Adjust the "Temperature" slider (lower values = more predictable, higher values = more creative) and "Max Tokens" number input to control the response generation.
3.  **Enter Your Prompts (Main Area):**
    * Below the main chat history display, you'll find two input columns.
    * Type your message for Bot 1 in the "Message for Bot 1" text area.
    * Optionally, use the "Upload Image (Bot 1)" button to add an image to your prompt for Bot 1 (ensure the selected model supports image input).
    * Do the same for Bot 2 using its respective input area and uploader.
4.  **Send!**
    * Click the "✉️ Send to Both Bots" button below the input areas.
    * The application will display your input in each bot's chat history.
    * It will then sequentially send the request to Bot 1, stream its response, and then send the request to Bot 2 and stream its response. Watch the responses appear in real-time!
5.  **Continue the Conversation:** Enter new messages and send again. The conversation history for each bot is maintained separately during your session.
6.  **Clear Histories (Sidebar):** Use the "Clear All Histories" button to start fresh conversations. Changing a selected model will also automatically clear the history for that bot.

## Tips for Comparison

* Try the same text prompt for both bots to see differences in style, tone, and content.
* Use the same image with different text prompts.
* Use the same text prompt with different images.
* Experiment with different System Prompts to see how it influences behavior.
* Adjust the Temperature to see how creativity varies.

## Prerequisites

* Python 3.9+
* pip (Python package installer)
* Git

## Setup

1.  **Clone the repository (or download the files):**
    ```bash
    # Replace with your repository URL after pushing
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` file:** Create a file named `.env` in the project's root directory. Add your API keys like this:
    ```dotenv
    ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    OPENAI_API_KEY="your_openai_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here"

    # Optional: For Vertex AI with google-genai (uncomment and set if using Vertex)
    # GOOGLE_GENAI_USE_VERTEXAI=True
    # GOOGLE_CLOUD_PROJECT='your-gcp-project-id'
    # GOOGLE_CLOUD_LOCATION='us-central1'
    ```
    **Important:** This `.env` file is excluded by `.gitignore` and should *not* be committed to Git. Obtain keys from the respective AI providers (Anthropic, OpenAI, Google AI Studio / Google Cloud).

## Running the App

1.  Make sure your virtual environment is activated.
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run bot_vs_bot.py
    ```
3.  Open the local URL provided in your web browser (usually `http://localhost:8501`).

## Notes

* Ensure you have the necessary API keys with sufficient credits/access for the selected models.
* Some models (especially preview ones) might have specific access requirements or regional availability. The model lists in the code are based on recent information but may need updates as APIs evolve.
* Error handling is basic; complex API errors might require inspecting the console output.
