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
            const style = window.getComputedStyle(el);
            if (style.boxShadow !== 'none' || style.borderBottomWidth !== '0px' || style.borderBottomStyle !== 'none' || style.textDecorationLine !== 'none') {
                if (el.closest('[data-testid="stTabs"]')) {
                    results.push({
                        tag: el.tagName,
                        testId: el.getAttribute('data-testid') || el.getAttribute('data-baseweb'),
                        bg: style.backgroundColor,
                        border: style.borderBottom,
                        shadow: style.boxShadow,
                        textDec: style.textDecorationLine
                    });
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
