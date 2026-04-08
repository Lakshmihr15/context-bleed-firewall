## How To Run

The project has two parts: the FastAPI backend and the local dashboard UI.

1. Install the Python dependencies:
   pip install -r requirements.txt

2. Start the backend server:
   python -m uvicorn server:app --reload

3. Open a second terminal in the project folder and serve the UI:
   python -m http.server 8000

4. Open the dashboard in your browser:
   http://localhost:8000

5. Optional: load the Chrome extension to capture textbox input:
   - Open Chrome
   - Go to chrome://extensions/
   - Turn on Developer mode
   - Click Load unpacked
   - Select the chrome_extension folder

6. To verify the backend directly:
   - http://127.0.0.1:8000/health
   - http://127.0.0.1:8000/graph/summary
   - http://127.0.0.1:8000/graph/view
