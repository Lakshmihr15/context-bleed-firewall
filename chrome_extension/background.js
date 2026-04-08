// background.js
// Receives data from content scripts and posts it to the local FastAPI server.

const SERVER_URL = "http://127.0.0.1:8000/chrome_input";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "INPUT_CAPTURED") {
    const tabId = sender.tab ? sender.tab.id.toString() : "unknown_tab";
    const payload = message.payload;
    
    // Send to our backend
    fetch(SERVER_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        tab_id: tabId,
        element_id: payload.element_id,
        text: payload.text,
        page_title: payload.page_title,
        page_url: payload.page_url,
        heading: payload.heading,
        field_type: payload.field_type
      })
    })
    .then(response => response.json())
    .then(data => console.log("Successfully logged input to Pacific Context Engine", data))
    .catch(error => console.error("Error logging input:", error));
  }
});
