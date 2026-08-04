from playwright.sync_api import sync_playwright
import time
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:8505")
    time.sleep(3)
    highlight = page.locator("[data-baseweb='tab-highlight']").first
    print("tab-highlight display:", highlight.evaluate("el => window.getComputedStyle(el).display"))
    print("tab-highlight bg:", highlight.evaluate("el => window.getComputedStyle(el).backgroundColor"))
    
    border = page.locator("[data-baseweb='tab-border']").first
    print("tab-border display:", border.evaluate("el => window.getComputedStyle(el).display"))

    tab = page.locator("[data-baseweb='tab']").first
    print("tab border-bottom:", tab.evaluate("el => window.getComputedStyle(el).borderBottom"))
    print("tab box-shadow:", tab.evaluate("el => window.getComputedStyle(el).boxShadow"))
    
    print("tablist border-bottom:", page.locator("[data-baseweb='tab-list']").first.evaluate("el => window.getComputedStyle(el).borderBottom"))
    
    browser.close()
