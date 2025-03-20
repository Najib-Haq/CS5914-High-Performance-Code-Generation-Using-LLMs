# CS5914: High-Performance Code Generation Using LLMs

**By:** Prayash Joshi  
**Email:** [prayash@vt.edu](mailto:prayash@vt.edu), [prayashjoshi@hotmail.com](mailto:prayashjoshi@hotmail.com)



This project is designed to evaluate code generation using Large Langauge Models (LLMs) for High Performance Computing tasks.
It integrates multiple providers:
- **ChatOpenAI** (for OpenAI models)
- **ChatGroq** (for Groq models, for open-source models like Deepseek, Mistral and llama)
- **ChatGoogleGenerativeAI** (for Google Gemini models)

## Requirements

1. **Python 3.12**  
   This project requires Python 3.12. You can verify your version by running:
   ```sh
   python --version
   ```

2. **Poetry**  
   Install Poetry for dependency management:
   ```sh
   pip install poetry
   ```

## Setup

### 1. Clone the repository:
```sh
git clone https://github.com/Najib-Haq/CS3914-High-Performance-Code-Generation-Using-LLMs
cd llm_blue
```

### 2. Install dependencies:
If you have Poetry installed, run:
```sh
poetry install
```

### 3. Configure environment variables:
Go to `.env` in the blue folder root and add your API keys:
```ini
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Using the Poetry Environment:
To check which Python executable your Poetry environment is using, run:
```sh
poetry env info
```
You should see output similar to:

```
Virtualenv
Python:         3.12.8
Implementation: CPython
Path:           {your path}
Executable:     {your executable path}
Valid:          True
```


## Optional: VS Code Dev Container
For an isolated development environment using Docker and VS Code, you can use the provided `.devcontainer` configuration.