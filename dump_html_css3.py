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
            const rect = el.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0 && rect.height < 5) {
                if (style.backgroundColor !== 'rgba(0, 0, 0, 0)' || style.borderBottomWidth !== '0px') {
                    results.push({
                        tag: el.tagName,
                        class: el.className,
                        id: el.id,
                        width: rect.width,
                        height: rect.height,
                        bg: style.backgroundColor,
                        border: style.borderBottom,
                        display: style.display,
                        html: el.outerHTML.substring(0, 100)
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
