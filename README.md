Of course. A great README is crucial for any project. Here is a professionally written English version based on the advanced features of your IntelliFlow application.

---

# IntelliFlow: Advanced RAG & Agentic AI Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Streamlit](https://img.shields.io/badge/Streamlit-ready-brightgreen.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-powered-orange.svg)](https://langchain.ai/docs/langgraph)
[![PostgreSQL + pgvector](https://img.shields.io/badge/Database-PostgreSQL%20%26%20pgvector-blue.svg)](https://github.com/pgvector/pgvector)

**IntelliFlow** is a powerful and scalable framework for Retrieval-Augmented Generation (RAG) and Agentic AI. This is not just a simple Q&A bot; it's a sophisticated system featuring a **dynamic workflow, self-correction capabilities, and the ability to use tools**.

The project is powered by **LangGraph**, implementing a complex, cyclical reasoning process that allows the agent to **analyze, rewrite, route, retrieve, generate, and finally, reflect** on the quality of its output. If the answer is subpar, it automatically retries to improve accuracy.

![IntelliFlow UI Demo](https://path-to-your-demo-gif.gif)
*(A short GIF demonstrating the UI and features is highly recommended here)*

## ✨ Core Features

- **🧠 Agentic Workflow (Powered by LangGraph)**:
  - **Query Analysis**: Automatically identifies the user's intent (e.g., knowledge query, weather request, or general chat).
  - **Query Rewriting**: Optimizes the user's input for better retrieval and generation results.
  - **Dynamic Routing**: Intelligently dispatches tasks to the document retriever, external tools, or a direct generation module based on the query type.
  - **Reflection & Self-Correction**: After generating an answer, the agent evaluates its quality. If deemed insufficient, it enters a retry loop, adjusting its strategy to produce a better response.

- **📚 Intelligent RAG Pipeline**:
  - **Threshold + Top-K Hybrid Retrieval**: First, it attempts to retrieve documents above a specific similarity threshold. If none are found, it automatically **falls back** to returning the top-K most similar documents, ensuring you always get the most relevant context available.
  - **Persistent Vector Store**: Utilizes **PostgreSQL + pgvector** as an enterprise-grade vector database that is stable, reliable, and scalable.
  - **Multi-Format Support**: Natively handles PDF and TXT document uploads.

- **🛠️ External Tool Integration**:
  - Comes with a built-in **real-time weather query tool** (using the Amap API) to showcase the agent's ability to interact with the outside world.
  - The framework is easily extensible, allowing you to integrate more custom tools (e.g., calculators, search engines, API callers).

- **🔍 Transparent Reasoning Process**:
  - Users can opt to view the AI's "thinking process" in real-time on the interface. This enhances trust and provides invaluable insights for debugging and optimization.

- **🚀 Modern & Scalable Tech Stack**:
  - **Backend**: Python, LangChain, LangGraph
  - **Frontend**: Streamlit
  - **Database**: Dockerized PostgreSQL + pgvector
  - **Configuration**: Pydantic
  - **Dependency Management**: `pyproject.toml`

## 🏛️ System Architecture

IntelliFlow's workflow is designed as a cyclical graph rather than a linear chain. This enables complex reasoning and self-healing capabilities.

```mermaid
graph TD
    A[User Input] --> B{IntelliFlow Agent (LangGraph)};
    B --> C[Step 1: Analyze Query];
    C --> D[Step 2: Rewrite Query];
    D --> E{Step 3: Intelligent Routing};
    E -- Knowledge Query --> F[📚 Smart Retrieval];
    E -- Weather Query --> G[🛠️ Weather Tool];
    E -- General Query --> H[🧠 Direct Generation];
    F --> H;
    G --> H;
    H --> I[Step 4: Generate Response];
    I --> J[Step 5: Quality Reflection];
    J -- Poor Quality & Retries Left --> K[Prepare for Retry];
    K --> D;
    J -- Good Quality or No Retries Left --> L[✅ Final Output];
    L --> M[User Interface];
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Git

### 1. Clone the Repository
```bash
git clone https://your-repository-url.git
cd IntelliFlow
```

### 2. Configure Environment Variables
Copy the example environment file. You will need to fill in your API keys.
```bash
cp .env.example .env
```
Now, edit the `.env` file with your Alibaba Cloud (for DashScope models) and Amap API keys:
```env
# .env
ALI_API_KEY="sk-your-alibaba-cloud-api-key"
AMAP_API_KEY="your-amap-api-key"
# Other settings can be left as default
```

### 3. Launch the Database
Start the PostgreSQL + pgvector service in the background using Docker Compose.
```bash
docker-compose up -d
```
On the first run, this command will automatically create the database, user, and initialize the tables and extensions according to `init.sql`.

### 4. Install Dependencies
It is highly recommended to use a virtual environment.
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install all dependencies in editable mode
pip install -e .
```
*(Using `-e .` installs the project in editable mode, which is convenient for development.)*

### 5. Run the Application
You're all set! Launch the Streamlit application.
```bash
streamlit run src/app.py```
Open your web browser and navigate to `http://localhost:8501` to start interacting with IntelliFlow.

## 📖 Project Structure
```
IntelliFlow/
├── .env              # Stores API keys and config (created by you)
├── docker-compose.yml # Docker config for launching the database
├── init.sql          # Database initialization script
├── pyproject.toml    # Project metadata and dependency management
├── src/
│   ├── agents/       # LangGraph agent definitions (graph, nodes)
│   ├── config/       # Application configuration (Pydantic)
│   ├── database/     # DB connection, models, and vector store logic
│   ├── tools/        # External tools (weather, document processor)
│   ├── utils/        # Helper utilities (decorators, UI, chat history)
│   └── app.py        # Streamlit application main entry point
└── README.md         # The file you are reading
```

## 💡 Future Directions
- [ ] **Integrate More Tools**: Add a web search tool (e.g., Tavily), a calculator, a code interpreter, etc.
- [ ] **Support More Document Types**: Add support for Word, Markdown, and HTML files.
- [ ] **Session Management & Multi-tenancy**: Provide isolated sessions and document stores for different users.
- [ ] **Frontend Enhancements**: Explore more complex UI interactions and visualizations.
- [ ] **Model Evaluation**: Build an evaluation framework to test the performance of different models and prompting strategies.


if test:
pip install -r requirements_test.txt
# 完整测试
python test_graph.py
# 调试模式
python debug_graph.py