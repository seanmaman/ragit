# ragIt™ - Your Personal AI Assistant - 2024

ragIt™ is an AI-powered assistant that uses a Retrieval-Augmented Generation (RAG) architecture to answer questions based on a local document knowledge base. It features a user-friendly web interface and can monitor a directory to automatically keep its knowledge up-to-date.

## Features

-   **AI Assistant**: A powerful AI assistant that can answer questions about your documents.
-   **Web Interface**: A simple and intuitive web interface for interacting with the assistant.
-   **File Monitoring**: Monitors a directory for new or updated files and automatically adds them to the knowledge base.
-   **Source Citing**: Cites the sources of its answers, so you can easily verify the information.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

-   [Python 3.x](https://www.python.org/downloads/)
-   [Git](https://git-scm.com/downloads)

### Cloning the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone <repository-url>
```

### Installation

1.  Navigate to the project directory:

    ```bash
    cd ragit
    ```

2.  Create a virtual environment:

    ```bash
    python -m venv venv
    ```

3.  Activate the virtual environment:

    -   On Windows:

        ```bash
        venv\Scripts\activate
        ```

    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

4.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use ragIt™, you need to run two separate scripts: `monitor.py` and `gui.py`.

### Running the Monitor

The `monitor.py` script monitors the `documents` directory for any file changes and updates the knowledge base accordingly. To run the monitor, use the following command:

```bash
python monitor.py
```

### Running the GUI

The `gui.py` script runs the web interface, which you can use to interact with the AI assistant. To run the GUI, use the following command:

```bash
streamlit run gui.py
```

Once the GUI is running, you can access it in your web browser at `http://localhost:8501`.
