# Personal Finance Agent with Azure AI Foundry

This application creates an intelligent agent using Azure AI Foundry that can analyze and answer questions about your financial documents. Upload bank statements, credit card statements, or other financial documents, and chat with the agent to get insights about your finances.

## Features

- Upload multiple financial documents (PDF, TXT, CSV, XLSX)
- Process documents using Azure AI Foundry's vector store for semantic search
- Chat with an AI agent that can analyze your financial documents
- Get insights about transactions, spending patterns, and financial details

## Prerequisites

1. An Azure account with an Azure AI Foundry project
2. The following roles assigned to your account:
   - **Storage Blob Data Contributor** on your project's storage account
   - **Azure AI Developer** on your project

## Setup Instructions

1. Clone this repository:
   ```
   git clone <repository-url>
   cd personal-finance-agent
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   
   Create a `.env` file in the project directory with the following variables:
   ```
   PROJECT_ENDPOINT=https://<AIFoundryResourceName>.services.ai.azure.com/api/projects/<ProjectName>
   MODEL_DEPLOYMENT_NAME=gpt-4o
   ```

   Alternatively, you can provide these values directly in the Streamlit UI.

4. Run the Streamlit app:
   ```
   streamlit run personal_finance_agent.py
   ```

## Usage

1. **Connect to Azure AI Foundry**:
   - Enter your Project Endpoint and Model Deployment Name in the sidebar
   - Click "Connect to Azure AI Foundry"

2. **Upload Documents**:
   - Go to the "Upload Documents" tab
   - Upload your financial documents (PDF, TXT, CSV, XLSX)
   - Click "Process Documents"

3. **Chat with the Agent**:
   - Go to the "Chat with Agent" tab
   - Ask questions about your finances, such as:
     - "What was my highest expense last month?"
     - "How much did I spend on groceries?"
     - "Show me all transactions over $100"
     - "What are my recurring subscriptions?"

## Security and Privacy

- All documents are processed within your Azure AI Foundry project
- The application automatically cleans up resources (vector stores, files, agents) when closed
- No financial data is stored outside of your Azure environment

## Customization

You can customize the agent's instructions in the sidebar to focus on specific financial analysis tasks or to change the agent's tone and behavior.

## Troubleshooting

- If you encounter authentication issues, ensure you're logged in with `az login`
- For file processing errors, check that your documents are in a supported format
- If the agent doesn't find information in your documents, try rephrasing your question or check if the document contains the information you're looking for

## License

This project is licensed under the MIT License - see the LICENSE file for details.

