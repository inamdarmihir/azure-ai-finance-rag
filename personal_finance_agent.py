import os
import streamlit as st
import tempfile
from pathlib import Path
from openai import AzureOpenAI
import PyPDF2
import pandas as pd
import io

# Set page configuration
st.set_page_config(page_title="Personal Finance Agent", page_icon="💰", layout="wide")

# App title and description
st.title("Personal Finance Agent")
st.markdown("""
This application uses Azure AI Inference SDK to create an intelligent agent that can analyze and answer questions about your financial documents.
Upload your bank statements, credit card statements, or other financial documents, and ask questions about your finances.
""")

# Function to get Azure AI credentials
def get_azure_credentials():
    """Get Azure AI credentials from secrets or environment variables"""
    try:
        # Try to get from Streamlit secrets first (for Streamlit Cloud)
        endpoint = st.secrets.get("AZURE_ENDPOINT", "")
        subscription_key = st.secrets.get("AZURE_API_KEY", "")
        model_name = st.secrets.get("AZURE_MODEL_NAME", "gpt-4")
        deployment = st.secrets.get("AZURE_DEPLOYMENT_NAME", "gpt-4")
        api_version = st.secrets.get("AZURE_API_VERSION", "2024-12-01-preview")
        
        # If secrets are empty, try environment variables (for local development)
        if not endpoint:
            endpoint = os.getenv("AZURE_ENDPOINT", "")
        if not subscription_key:
            subscription_key = os.getenv("AZURE_API_KEY", "")
        if not model_name:
            model_name = os.getenv("AZURE_MODEL_NAME", "gpt-4")
        if not deployment:
            deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4")
        if not api_version:
            api_version = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
            
        return endpoint, subscription_key, model_name, deployment, api_version
    except Exception as e:
        st.error(f"Error accessing secrets: {str(e)}")
        return "", "", "gpt-4", "gpt-4", "2024-12-01-preview"

# Get credentials
endpoint, subscription_key, model_name, deployment, api_version = get_azure_credentials()

# Sidebar for Azure AI configuration
st.sidebar.header("Azure AI Configuration")

# Show configuration status
if endpoint and subscription_key:
    st.sidebar.success("✅ Azure AI credentials loaded from secrets")
    st.sidebar.info(f"**Model:** {model_name}")
    st.sidebar.info(f"**Deployment:** {deployment}")
    # Show masked endpoint for security
    masked_endpoint = endpoint[:30] + "..." if len(endpoint) > 30 else endpoint
    st.sidebar.info(f"**Endpoint:** {masked_endpoint}")
else:
    st.sidebar.error("❌ Azure AI credentials not found")
    st.sidebar.markdown("""
    **For Streamlit Cloud:**
    Add these secrets in your Streamlit Cloud app settings:
    - `AZURE_ENDPOINT`
    - `AZURE_API_KEY`
    - `AZURE_MODEL_NAME` (optional)
    - `AZURE_DEPLOYMENT_NAME` (optional)
    - `AZURE_API_VERSION` (optional)
    
    **For local development:**
    Set environment variables or create a `.streamlit/secrets.toml` file
    """)

# Function to create Azure AI client
@st.cache_resource
def create_ai_client(endpoint, api_key, api_version):
    if not endpoint or not api_key:
        return None

    try:
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        # Test the connection by trying a simple call
        try:
            test_response = client.chat.completions.create(
                messages=[{"role": "system", "content": "Test"}],
                model=deployment,
                max_tokens=1
            )
            return client
        except Exception as test_error:
            st.sidebar.warning(f"Connection created but model test failed: {test_error}")
            return client  # Return client anyway, it might work for actual requests

    except Exception as e:
        st.error(f"Error creating AI client: {str(e)}")
        return None

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Function to extract text from CSV/Excel
def extract_text_from_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file.getvalue()))
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file.getvalue()))
        else:
            return None

        # Convert DataFrame to text representation
        text = f"File: {file.name}\n\n"
        text += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        text += df.to_string(index=False)
        return text
    except Exception as e:
        st.error(f"Error extracting data from file: {str(e)}")
        return None

# Function to extract text from TXT
def extract_text_from_txt(file):
    try:
        return file.getvalue().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

# Initialize session state
if "ai_client" not in st.session_state:
    st.session_state.ai_client = None
if "document_texts" not in st.session_state:
    st.session_state.document_texts = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Auto-connect if credentials are available
if endpoint and subscription_key and not st.session_state.ai_client:
    with st.spinner("Connecting to Azure AI..."):
        st.session_state.ai_client = create_ai_client(endpoint, subscription_key, api_version)
        if st.session_state.ai_client:
            st.sidebar.success("✅ Connected automatically!")

# Manual connect button (fallback)
if st.sidebar.button("Reconnect to Azure AI"):
    with st.spinner("Reconnecting to Azure AI..."):
        # Clear cached client
        create_ai_client.clear()
        st.session_state.ai_client = create_ai_client(endpoint, subscription_key, api_version)
        if st.session_state.ai_client:
            st.sidebar.success("✅ Reconnected successfully!")
        else:
            st.sidebar.error("❌ Failed to reconnect. Please check your credentials.")

# Connection status
if st.session_state.ai_client:
    st.sidebar.success("🟢 Azure AI Connected")
else:
    st.sidebar.error("🔴 Azure AI Not Connected")

# Main content with tabs
tab1, tab2 = st.tabs(["Upload Documents", "Chat with Agent"])

# Tab 1: Upload Documents
with tab1:
    st.header("Upload Financial Documents")
    st.markdown("Upload your bank statements, credit card statements, or other financial documents.")

    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv", "xlsx"]
    )

    if uploaded_files and st.button("Process Documents"):
        if not st.session_state.ai_client:
            st.error("❌ Please ensure Azure AI is connected first")
        else:
            with st.spinner("Processing documents..."):
                # Clear previous documents
                st.session_state.document_texts = []

                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    st.info(f"Processing {uploaded_file.name}")

                    # Extract text based on file type
                    text = None
                    if uploaded_file.name.lower().endswith('.pdf'):
                        text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.name.lower().endswith(('.csv', '.xlsx', '.xls')):
                        text = extract_text_from_data(uploaded_file)
                    elif uploaded_file.name.lower().endswith('.txt'):
                        text = extract_text_from_txt(uploaded_file)

                    if text:
                        st.session_state.document_texts.append({
                            'filename': uploaded_file.name,
                            'content': text
                        })
                        st.success(f"✅ Processed {uploaded_file.name}")
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}")

                if st.session_state.document_texts:
                    # Clear previous messages and add success message
                    st.session_state.messages = []
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"🎉 Successfully processed {len(st.session_state.document_texts)} documents! You can now ask questions about your financial data."
                    })
                    st.success(f"✅ Processed {len(st.session_state.document_texts)} documents successfully!")

# Tab 2: Chat with Agent
with tab2:
    st.header("Chat with Your Personal Finance Agent")

    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
        elif message["role"] == "system":
            st.info(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your finances..."):
        if not st.session_state.ai_client:
            st.error("❌ Please ensure Azure AI is connected first")
        elif not st.session_state.document_texts:
            st.error("❌ Please upload and process documents first")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Prepare context from documents
            context = "Here are the financial documents I have access to:\n\n"
            for doc in st.session_state.document_texts:
                context += f"=== {doc['filename']} ===\n"
                context += doc['content'][:3000]  # Limit content to avoid token limits
                if len(doc['content']) > 3000:
                    context += "\n... (content truncated) ..."
                context += "\n\n"

            # Create system message with instructions
            system_instructions = """You are a helpful financial assistant. Your job is to analyze financial documents like bank statements and credit card statements.
            Provide clear, accurate information about transactions, spending patterns, and financial insights.
            When asked about specific transactions or financial details, refer to the document content provided.
            Always be respectful of privacy and maintain confidentiality of financial information.
            If you can't find specific information in the documents, clearly state that."""

            # Prepare user message with context
            user_message_content = f"Context from financial documents:\n{context}\n\nUser question: {prompt}"

            # Prepare messages for Azure AI
            messages = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_message_content}
            ]

            # Get response from Azure AI
            with st.spinner("🤔 Analyzing your financial documents..."):
                try:
                    response = st.session_state.ai_client.chat.completions.create(
                        messages=messages,
                        model=deployment,
                        temperature=0.1,
                        max_tokens=1000
                    )

                    assistant_message = response.choices[0].message.content

                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                    st.chat_message("assistant").write(assistant_message)

                except Exception as e:
                    st.error(f"❌ Error getting response: {str(e)}")
                    if "quota" in str(e).lower():
                        st.error("You may have exceeded your API quota. Please check your Azure AI usage.")
                    elif "authentication" in str(e).lower() or "401" in str(e):
                        st.error("Authentication failed. Please check your API key in secrets.")
                    elif "404" in str(e):
                        st.error("❌ **Model not found!** Please check your deployment name in secrets.")
                        st.error("**How to fix:**")
                        st.error("1. Go to Azure Portal → Your OpenAI Resource → Model deployments")
                        st.error("2. Copy the exact **deployment name** (not model name)")
                        st.error("3. Update AZURE_DEPLOYMENT_NAME in your secrets")
                    elif "429" in str(e):
                        st.error("Rate limit exceeded. Please wait a moment and try again.")

# Sidebar with document info
if st.session_state.document_texts:
    st.sidebar.header("📄 Uploaded Documents")
    for i, doc in enumerate(st.session_state.document_texts):
        with st.sidebar.expander(f"{doc['filename']}"):
            preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            st.text(preview)

# Add debug info
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.markdown("### Debug Information")
    st.sidebar.markdown(f"**Endpoint:** {endpoint[:30]}..." if endpoint else "Not set")
    st.sidebar.markdown(f"**Model:** {model_name}")
    st.sidebar.markdown(f"**Deployment:** {deployment}")
    st.sidebar.markdown(f"**API Version:** {api_version}")
    st.sidebar.markdown(f"**Documents Loaded:** {len(st.session_state.document_texts)}")
    st.sidebar.markdown(f"**Messages:** {len(st.session_state.messages)}")
    st.sidebar.markdown(f"**Client Status:** {'Connected' if st.session_state.ai_client else 'Not Connected'}")

# Footer
st.markdown("---")
st.markdown("### 🔧 Setup Instructions")

# Streamlit Cloud setup
with st.expander("🌐 Streamlit Cloud Setup"):
    st.markdown("""
    **For Streamlit Cloud deployment:**
    
    1. **Deploy your app** to Streamlit Cloud
    2. **Go to app settings** → Advanced settings → Secrets
    3. **Add the following secrets** in TOML format:
    
    ```toml
    [secrets]
    AZURE_ENDPOINT = "https://your-resource.cognitiveservices.azure.com/"
    AZURE_API_KEY = "your-api-key-here"
    AZURE_MODEL_NAME = "gpt-4"  # optional
    AZURE_DEPLOYMENT_NAME = "your-deployment-name"  # optional
    AZURE_API_VERSION = "2024-12-01-preview"  # optional
    ```
    
    4. **Save and restart** your app
    
    **Security Note:** Never commit secrets to your repository!
    """)

# Local development setup
with st.expander("💻 Local Development Setup"):
    st.markdown("""
    **For local development, you have two options:**
    
    **Option 1: Create `.streamlit/secrets.toml` file:**
    ```toml
    [secrets]
    AZURE_ENDPOINT = "https://your-resource.cognitiveservices.azure.com/"
    AZURE_API_KEY = "your-api-key-here"
    AZURE_MODEL_NAME = "gpt-4"
    AZURE_DEPLOYMENT_NAME = "your-deployment-name"
    AZURE_API_VERSION = "2024-12-01-preview"
    ```
    
    **Option 2: Set environment variables:**
    ```bash
    export AZURE_ENDPOINT="https://your-resource.cognitiveservices.azure.com/"
    export AZURE_API_KEY="your-api-key-here"
    export AZURE_MODEL_NAME="gpt-4"
    export AZURE_DEPLOYMENT_NAME="your-deployment-name"
    export AZURE_API_VERSION="2024-12-01-preview"
    ```
    
    **Note:** Add `.streamlit/secrets.toml` to your `.gitignore` file!
    """)

# Azure credentials setup
with st.expander("🔑 How to get your Azure AI credentials"):
    st.markdown("""
    1. **Go to Azure Portal**: https://portal.azure.com
    2. **Find your Azure OpenAI resource**: Navigate to your Azure OpenAI service
    3. **Get the endpoint**: Copy the endpoint URL (e.g., https://your-resource.openai.azure.com/)
    4. **Get API key**: Go to "Keys and Endpoint" section and copy one of the keys
    5. **Check deployment**: Go to "Model deployments" to see your deployed models and their names

    **Required Python packages:**
    ```bash
    pip install streamlit openai PyPDF2 pandas openpyxl
    ```
    """)

with st.expander("💬 Sample Questions to Ask"):
    st.markdown("""
    - "What was my total spending last month?"
    - "Show me all transactions above $100"
    - "What are my biggest expense categories?"
    - "How much did I spend on groceries?"
    - "What was my account balance at the end of the statement period?"
    - "Are there any unusual or suspicious transactions?"
    - "What's my spending pattern over time?"
    """)

st.markdown("Powered by Azure OpenAI | Personal Finance Agent | Streamlit Cloud Ready ☁️")
