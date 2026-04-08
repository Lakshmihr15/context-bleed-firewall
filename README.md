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

## Deploy On Render

This project can be deployed as a single Render Web Service because the FastAPI app serves the dashboard UI, CSS, and JS.

### Option 1: Deploy from GitHub

1. Push this project to a GitHub repository.
2. In Render, create a new **Web Service** and connect the repo.
3. Use these settings:
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Deploy the service.
5. Open the Render URL after deployment.

### Important For Chrome Capture

After deployment, update `chrome_extension/background.js` so the extension sends captured text to your Render URL instead of `http://127.0.0.1:8000`.

For example, change the backend base URL to:
`https://YOUR-RENDER-SERVICE.onrender.com`

Then the extension will POST to:
`https://YOUR-RENDER-SERVICE.onrender.com/chrome_input`

### Option 2: Use Render Blueprint

If you keep the included `render.yaml` file in the repo, you can create the service from the Render blueprint.

### Notes

- The first startup may take a little longer because `sentence-transformers` can download its model on first run.
- The app does not require an external API key.
- Render free services use ephemeral disk storage, so files like `interaction_graph.json` and `encryption.key` may reset on restart unless you add persistent storage.