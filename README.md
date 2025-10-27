# RAGit

GitHub Repository: https://github.com/seanmaman/ragit

## Overview
RAGit is a document processing and retrieval system that uses RAG (Retrieval-Augmented Generation) technology.

## Important: Documents Folder Setup

The `documents/` folder needs to be populated with files for the system to work properly. You can add any type of files including:

- Text documents (.txt)
- PDF files (.pdf)
- Word documents (.docx, .doc)
- Markdown files (.md)
- Code files (.py, .js, .html, .css, etc.)
- JSON files (.json)
- CSV files (.csv)
- Any other text-based or document files

**Note:** The system will automatically process and index all files placed in the `documents/` folder for retrieval and analysis.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/seanmaman/ragit.git
cd ragit
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Important:** Place your documents in the `documents/` folder
2. Run the GUI application:
```bash
python gui.py
```

Or use the batch files:
- Windows: `run_gui.bat`
- Monitor: `run_monitor.bat`

## Project Structure
- `gui.py` - Main GUI application
- `monitor.py` - System monitor
- `modules/` - Core modules for RAG functionality
- `documents/` - **Place your documents here for processing (required)**
- `db/` - Database storage for indexed documents
- `competitors/` - Competitor analysis data
- `settings.json` - Configuration settings

## Getting Started

1. Ensure the `documents/` folder contains the files you want to process
2. Run the application using one of the methods above
3. The system will automatically index and process your documents

## License
See repository for license information.
