from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:8504")
    time.sleep(3)
    # Get computed style of the highlight
    highlight = page.locator("[data-baseweb='tab-highlight']").first
    print("display:", highlight.evaluate("el => window.getComputedStyle(el).display"))
    print("visibility:", highlight.evaluate("el => window.getComputedStyle(el).visibility"))
    print("opacity:", highlight.evaluate("el => window.getComputedStyle(el).opacity"))
    print("background:", highlight.evaluate("el => window.getComputedStyle(el).backgroundColor"))
    browser.close()
