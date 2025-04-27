# Drape Backend Startup Instructions

## 1. Clone the Repository
If you haven't already, clone the repository and navigate to the backend directory:
```sh
git clone <repo-url>
cd backend
```

## 2. Set Up the Python Virtual Environment
Create a new virtual environment (recommended):
```sh
python3 -m venv venv
```
Activate the virtual environment:
```sh
source venv/bin/activate
```

## 3. Install Dependencies
Install all required Python packages:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Set the GOOGLE_API_KEY (for Gemini)
You must set your Google Gemini API key as an environment variable. There are two recommended methods:

### Option A: Set for Current Session (recommended for local dev)
In your terminal, before starting the server, run:
```sh
export GOOGLE_API_KEY="your-actual-api-key"
```
Then start the server in the same terminal session.

### Option B: Set Globally (all future sessions)
Add this line to your `~/.zshrc` (for zsh) or `~/.bashrc` (for bash):
```sh
export GOOGLE_API_KEY="your-actual-api-key"
```
Then run:
```sh
source ~/.zshrc  # or source ~/.bashrc
```

### Option C: Use a .env File (project-specific)
1. Create a file named `.env` in the backend directory:
    ```sh
    echo "GOOGLE_API_KEY=your-actual-api-key" > .env
    ```
2. Ensure your FastAPI app loads the .env file by adding the following to the top of `main.py`:
    ```python
    from dotenv import load_dotenv
    load_dotenv()
    ```
3. Install python-dotenv if not already present:
    ```sh
    pip install python-dotenv
    ```

## 5. Start the FastAPI Server
Run the following command from the backend directory:
```sh
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- The server will be available at: http://localhost:8000
- API docs (Swagger UI) are available at: http://localhost:8000/docs

## 6. Troubleshooting
- If you see `RuntimeError: GOOGLE_API_KEY environment variable not set.`, make sure your key is set and the terminal session is refreshed. If you set the variable in ~/.zshrc or ~/.bashrc, open a new terminal or run `source ~/.zshrc` before starting the server. If you set it with `export`, be sure to start the server in the same terminal session.
- If `pip` or `python` are not found, ensure Python 3 is installed and on your PATH.
- If dependencies are missing, re-run `pip install -r requirements.txt`.

---

For further help, please contact the project maintainer.

---

## Notes on Environment Variables
- Environment variables set in ~/.zshrc or ~/.bashrc are only available in new terminal sessions after sourcing the file or restarting the terminal.
- If you set GOOGLE_API_KEY with export in a terminal, it is only available in that session. Always start your server from the same terminal where you exported the key.
- Using a .env file with python-dotenv is recommended for local development and avoids issues with variable visibility.
