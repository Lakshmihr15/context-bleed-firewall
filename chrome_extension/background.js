// background.js
// Receives data from content scripts and posts it to the FastAPI server.

const LOCAL_SERVER_URL = "http://127.0.0.1:8000/chrome_input";
const DEFAULT_SERVER_URL = "https://pacific-context-bleed-firewall.onrender.com/chrome_input";
const SEARCH_HOSTS = new Set([
  "www.google.com",
  "google.com",
  "www.bing.com",
  "bing.com",
  "search.yahoo.com",
  "duckduckgo.com",
  "www.duckduckgo.com",
  "search.brave.com",
  "www.startpage.com"
]);

const lastSearchSignatureByTab = new Map();

function extractSearchQuery(urlString) {
  try {
    const url = new URL(urlString);
    if (!SEARCH_HOSTS.has(url.hostname)) {
      return "";
    }

    const candidateParams = ["q", "p", "query", "text"];
    for (const paramName of candidateParams) {
      const value = url.searchParams.get(paramName);
      if (value && value.trim()) {
        return value.trim();
      }
    }

    if (url.hostname.includes("duckduckgo.com")) {
      const duckQuery = url.searchParams.get("q") || url.searchParams.get("kp");
      if (duckQuery && duckQuery.trim()) {
        return duckQuery.trim();
      }
    }
  } catch (error) {
    return "";
  }

  return "";
}

function postCapturedInput(payload) {
  const requestOptions = {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  };

  return fetch(LOCAL_SERVER_URL, requestOptions)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Local server returned ${response.status}`);
      }
      return response.json();
    })
    .catch(() => {
      return fetch(DEFAULT_SERVER_URL, requestOptions).then((response) => {
        if (!response.ok) {
          throw new Error(`Remote server returned ${response.status}`);
        }
        return response.json();
      });
    });
}

function logSearchQuery(tabId, url, title) {
  const query = extractSearchQuery(url);
  if (!query) {
    return;
  }

  const signature = `${tabId}:${query}:${url}`;
  if (lastSearchSignatureByTab.get(tabId) === signature) {
    return;
  }
  lastSearchSignatureByTab.set(tabId, signature);

  postCapturedInput({
    tab_id: tabId.toString(),
    element_id: "omnibox_search",
    text: query,
    page_title: title || "Search Results",
    page_url: url,
    heading: "Chrome Search",
    field_type: "search"
  }).catch((error) => console.error("Error logging search query:", error));
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "INPUT_CAPTURED") {
    const tabId = sender.tab ? sender.tab.id.toString() : "unknown_tab";
    const payload = message.payload;
    // Prefer the local graph first so development runs update the local UI.
    postCapturedInput({
      tab_id: tabId,
      element_id: payload.element_id,
      text: payload.text,
      page_title: payload.page_title,
      page_url: payload.page_url,
      heading: payload.heading,
      field_type: payload.field_type
    })
      .then((data) => console.log("Successfully logged input to Pacific Context Engine", data))
      .catch((error) => console.error("Error logging input:", error));
  }
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete" || !tab || !tab.url) {
    return;
  }
  logSearchQuery(tabId, tab.url, tab.title || "Search Results");
});
