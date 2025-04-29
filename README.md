# PDF Analysis with Google GenAI
Extract text from PDFs and analyze content using Google's Generative AI API.

## Setup
Install: pip install gradio PyPDF2 google-generativeai kaggle-secrets

Set API key:
from kaggle_secrets import UserSecretsClient
UserSecretsClient().set_secret("mysecret", "your_api_key")

Access the Gradio interface at http://localhost:7860 to:
1. Upload PDFs
2. Get AI generated summary, Q&A
