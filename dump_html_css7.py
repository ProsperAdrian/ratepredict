from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:8505")
    time.sleep(3)
    
    script = """
    () => {
        const elements = document.querySelectorAll('*');
        const results = [];
        for (let el of elements) {
            if (!el.closest('[data-testid="stTabs"]')) continue;
            const style = window.getComputedStyle(el);
            if (style.backgroundImage !== 'none' || style.borderBottomWidth !== '0px') {
                results.push({
                    tag: el.tagName,
                    testId: el.getAttribute('data-testid') || el.getAttribute('data-baseweb'),
                    bgImage: style.backgroundImage,
                    borderBtn: style.borderBottom
                });
            }
        }
        return results;
    }
    """
    res = page.evaluate(script)
    for r in res:
        print(r)
    
    browser.close()
