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
            
            for (let pseudo of ['::before', '::after']) {
                const style = window.getComputedStyle(el, pseudo);
                if (style.content && style.content !== 'none') {
                    if (style.display !== 'none') {
                        results.push({
                            tag: el.tagName,
                            testId: el.getAttribute('data-testid') || el.getAttribute('data-baseweb'),
                            pseudo: pseudo,
                            display: style.display,
                            bg: style.backgroundColor,
                            border: style.borderBottom,
                            width: style.width,
                            height: style.height
                        });
                    }
                }
            }
        }
        return results;
    }
    """
    res = page.evaluate(script)
    for r in res:
        print(r)
    
    browser.close()
