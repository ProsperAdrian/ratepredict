from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:8505")
    time.sleep(3)
    page.screenshot(path="screenshot.png")
    print(page.locator("[data-testid='stTabs']").first.inner_html())
    browser.close()
