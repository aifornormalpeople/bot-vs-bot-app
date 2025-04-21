# Bot vs. Bot Arena 🤖🥊

A Streamlit application to compare responses from different Large Language Models (LLMs) side-by-side, supporting Anthropic (Claude), OpenAI (GPT), and Google (Gemini) models.

## Features

* Select two different models for comparison.
* Input a common system prompt.
* Provide separate user prompts and optional image uploads for each bot.
* View conversation history for each bot independently.
* See streaming responses from the models.

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
* Some models (especially preview ones) might have specific access requirements or regional availability.
