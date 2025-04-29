import os 
import gradio as gr
import PyPDF2
from google import genai
from google.genai import types
from google.api_core import retry
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("mysecret")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Wraps the generate_content() function with a retry handler
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)

# Extract text from pdf uploaded by user
def extract_text(pdf_path): 
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Formats the first 10000 characters of the document into a prompt that instructs the model to summarize it in bullet points.
def summarize_document(document_text):
    prompt = f"""
    Summarize the following document in bullet points in a concise and easy-to-understand way:

    {document_text[:10000]}
    """
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.5,         # accuracy and creativity balance
            max_output_tokens=512    # limit output length 
        )
    )
    return response.text 

# Formats the first 10000 characters of the document and the question entered by the user into a prompt that generate the answer
def ask_document(document_text, question):
    prompt = f"""
    You are a productive and intelligent assistant that answers questions about documents.
    
    Document:
    {document_text[:10000]}
    Question: {question}
    Answer:
    """
    response = client.models.generate_content(
        model = 'gemini-2.0-flash',
        contents = prompt,
        config = genai.types.GenerateContentConfig(
            temperature = 0.5,
            max_output_tokens = 512
        )
    )
    return response.text 

# Formats the first 10000 characters of the document to a prompt to generate 3 revelant questions and brief answers
def suggested_qa(document_text):
    prompt = f"""Based on this document, suggest 3 relevant questions and answer each briefly:
    Document:
    {document_text[:10000]}

    Format exactly like this:
    Q1: [Question 1]
    A1: [Answer 1]
    
    Q2: [Question 2]
    A2: [Answer 2]
    
    Q3: [Question 3]
    A3: [Answer 3]
    """
    response = client.models.generate_content(
        model = 'gemini-2.0-flash',
        contents = prompt,
        config = {'temperature':0.3}  # better accuracy, low creativity
    )
    return response.text 

# Build a prompt for Gemini to translate the input text to target language
def translate(text, target_language):
    prompt = f"Translate the following text into {target_language}:\n\n{text}"
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents = prompt,
        config = types.GenerateContentConfig(
            temperature = 0.3,       
            max_output_tokens = 512   
        )
    )
    return response.text

# Process uploaded pdf and generate summary, answer and suggested Q&A
def handle_file(pdf_file, question, language):
    text = extract_text(pdf_file)
    summary = summarize_document(text) 
    
    question = question or "What are the main findings?" # default question if no user entry
    answer = ask_document(text, question)
    qa = suggested_qa(text) 
    
    if language.lower() != 'english':
        summary = translate(summary, language)
        answer = translate(answer, language)
        qa = translate(qa, language) 
    return summary, answer, qa 

# Define supported target languages for translation
language_options = ["English", "Chinese", "Spanish", "French", "German", "Japanese", "Korean"]

# Create the Gradio UI using Blocks 
with gr.Blocks() as demo:
    
    # Title and description of the app
    gr.Markdown("""# 📚 Ask My Docs 🌍 *Document Summary • Q&A • Multilingual Support*""")

    # File upload and language selection row
    with gr.Row():
        pdf_file = gr.File(label='Upload your PDF')  
        language = gr.Dropdown(language_options, label='Select Language', value='English')  

    question = gr.Textbox(
        label="Ask a question about the document", 
        placeholder="e.g., What are the main findings?"
    )

    submit_btn = gr.Button("Let's Read!")

    summary = gr.Textbox(label="📌 Summary", lines=8) 
    answer = gr.Textbox(label="💬 Answer", lines=6)    
    qa = gr.Textbox(label="💡 Smart Q&A", lines=8)    

    # Bind the button to the handler function with input/output mapping
    submit_btn.click(
        fn=handle_file,                            
        inputs=[pdf_file, question, language],   
        outputs=[summary, answer, qa]          
    )

# Launch the app
demo.launch(share=True)
