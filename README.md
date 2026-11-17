# LegalIQ – AI-Powered Legal Research & Document Assistant

An intelligent legal research and document analysis application designed to simplify access to legal information through Natural Language Processing, document extraction, information retrieval and AI-powered assistance.

## Overview

LegalIQ is a Python-based legal assistance system that helps users interact with legal documents, retrieve relevant case information and obtain simplified insights from legal content.

The project combines document processing, NLP-based information retrieval, question answering and text summarization into a unified application.

The long-term objective is to evolve LegalIQ into an evidence-grounded legal research platform powered by advanced Retrieval-Augmented Generation (RAG).

## Key Features

- **Legal Document Processing:** Extract and process text from supported PDF and DOCX documents.
- **Legal Information Retrieval:** Retrieve relevant legal information using text-based similarity techniques.
- **Question Answering:** Answer questions using available document context and question-answering models.
- **Text Summarization:** Generate concise summaries of lengthy legal content.
- **Legal Case Search:** Identify relevant case information from the available dataset.
- **Document Analysis:** Simplify the exploration and understanding of legal documents.
- **Web Interface:** Interact with the application's legal assistance features through a Flask-based interface.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| NLP | Hugging Face Transformers |
| Language Models | BART, DistilBERT |
| Embeddings | Sentence Transformers |
| Information Retrieval | TF-IDF, Cosine Similarity |
| Document Processing | PyPDF2, python-docx |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Data Storage | SQLite, CSV |

*Technology descriptions reflect the existing project implementation and should be updated if the underlying code changes.*

## System Workflow

1. User uploads a legal document or enters a legal query.
2. The application processes the provided input.
3. Text extraction and preprocessing prepare the content for analysis.
4. Relevant information is identified through the existing retrieval mechanisms.
5. NLP models perform question answering or summarization.
6. Results are presented through the web interface.

## Project Architecture

The application currently follows a Flask-based architecture:

- **Presentation Layer:** User interface and document interaction.
- **Application Layer:** Flask routes and request handling.
- **Processing Layer:** Document extraction and text preprocessing.
- **NLP Layer:** Embedding generation, retrieval, question answering and summarization.
- **Data Layer:** Legal datasets and application storage.

See [Architecture Documentation](docs/architecture.md).

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Git

### Installation

```bash
git clone https://github.com/Ash-Geek8/LegalIQ-AI-System.git
cd LegalIQ-AI-System

python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application using the entry point and configuration supported by the repository.

## Current Scope and Future Development

LegalIQ is being developed towards a more advanced legal research architecture.

Planned improvements include:

- Advanced Retrieval-Augmented Generation.
- Hybrid retrieval combining dense and sparse search.
- Cross-encoder reranking.
- Source-level document citations.
- Metadata-aware document ingestion.
- Retrieval and generation evaluation.
- Persistent vector indexing.
- Evidence-grounded response generation.

See [RAG Development Roadmap](docs/rag_roadmap.md).

## Project Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | Application architecture and workflow |
| [Features](docs/features.md) | Existing capabilities |
| [Installation](docs/installation.md) | Setup instructions |
| [Technical Documentation](docs/technical_documentation.md) | Technology and implementation details |
| [Project Structure](docs/project_structure.md) | Repository organization |
| [RAG Roadmap](docs/rag_roadmap.md) | Planned technical improvements |
| [Evaluation Strategy](docs/evaluation_strategy.md) | Proposed evaluation methodology |

## Disclaimer

LegalIQ is an educational and technical project intended to assist with legal information exploration and document analysis. It does not replace qualified legal professionals, legal advice or official legal sources. Users should independently verify legal information before relying on it.

## Author

**Ashfaq Ahamed**

AI Engineer | Generative AI | NLP | Python

- GitHub: [Ash-Geek8](https://github.com/Ash-Geek8)

---

*Building towards intelligent, evidence-grounded legal research.*
