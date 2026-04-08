// content.js
// Injects listeners into text areas and input fields to capture user typing.

let typingTimer;
const DONE_TYPING_INTERVAL = 1500; // ms to wait after last keystroke

function getFieldHeading(el) {
  return (
    el.getAttribute('aria-label') ||
    el.getAttribute('placeholder') ||
    el.getAttribute('name') ||
    el.id ||
    el.tagName.toLowerCase()
  );
}

function sendDataToBackground(elementId, text, meta = {}) {
  if (!text || text.trim().length === 0) return;
  chrome.runtime.sendMessage({
    type: "INPUT_CAPTURED",
    payload: {
      element_id: elementId,
      text: text,
      page_title: meta.pageTitle,
      page_url: meta.pageUrl,
      heading: meta.heading,
      field_type: meta.fieldType
    }
  });
}

function handleInput(event) {
  const el = event.target;
  // Make sure it's a text input or textarea
  if (el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && (el.type === 'text' || el.type === 'search' || el.type === 'email'))) {
    clearTimeout(typingTimer);
    
    // Generate an ID if the element doesn't have one
    if (!el.id) {
       el.id = 'input_' + Math.random().toString(36).substr(2, 9);
    }
    
    typingTimer = setTimeout(() => {
      sendDataToBackground(el.id, el.value, {
        pageTitle: document.title,
        pageUrl: window.location.href,
        heading: getFieldHeading(el),
        fieldType: el.type || el.tagName.toLowerCase()
      });
    }, DONE_TYPING_INTERVAL);
  }
}

// Listen to input events on the document
document.addEventListener('input', handleInput, true);
