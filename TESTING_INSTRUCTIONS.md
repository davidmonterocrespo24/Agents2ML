# Testing Instructions for Agents2ML Application

## 1. Setting Up the Environment

1. Ensure you have Python 3.8 or higher installed.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Make sure Docker Desktop is running on your system.
4. Ensure the model `gpt-oss:120b` is running in Ollama. You can start it with:
   ```powershell
   ollama run gpt-oss:120b
   ```
5. Build the Docker image for the application (from the included `Dockerfile`). In the project root, run:
   ```powershell
   docker build -t agents2ml .
   ```

## 2. Initializing the Database

If running for the first time, initialize the database:
```powershell
python database_init.py
```

## 3. Running the Application

To start the application:
```powershell
python start.py
```

## 4. Running Tests

If there are automated tests (e.g., using `pytest`):
```powershell
pytest
```
Or, to run a specific test file:
```powershell
pytest path/to/test_file.py
```

## 5. Sample Data

- Sample CSV files for testing can be found in the `files for testing/` directory.
- Example: `files for testing/all sales of 3 years.csv`
- To use sample data, place your test files in this directory and reference them in your tests or application as needed.

## 6. Additional Notes

- For agent-specific tests, refer to the `agents/` directory for individual agent modules.
- For database-related tests, ensure the database is initialized and accessible.
- For further documentation, see `README.md` and `MULTIAGENT_SYSTEM_DOCUMENTATION.md`.
