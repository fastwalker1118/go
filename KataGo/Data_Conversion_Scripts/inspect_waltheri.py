#!/usr/bin/env python3
"""Quick script to inspect the rendered DOM of waltheri advanced page."""
from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Load the advanced page
    print("Loading https://ps.waltheri.net/database/advanced/ ...")
    page.goto(
        "https://ps.waltheri.net/database/advanced/", wait_until="networkidle", timeout=60000
    )
    time.sleep(5)

    # Count game links
    game_links = page.query_selector_all('a[href*="/database/game/"]')
    print(f"\nFound {len(game_links)} game links")

    # Show first 5 game link hrefs
    for i, a in enumerate(game_links[:5]):
        href = a.get_attribute("href") or ""
        text = a.inner_text() or ""
        print(f"  [{i}] href={href}  text={text!r}")

    # Look for pagination elements
    print("\n--- Pagination search ---")
    for sel_name, sel in [
        ("any button", "button"),
        ("any nav", "nav"),
        (".pagination", ".pagination"),
        ("[class*=pag]", "[class*=pag]"),
        ("[class*=page]", "[class*=page]"),
        ("a with >", 'a:has-text(">")'),
        ("a with Next", 'a:has-text("Next")'),
        ("a with »", 'a:has-text("»")'),
        ("select (dropdown)", "select"),
    ]:
        els = page.query_selector_all(sel)
        if els:
            print(f"  {sel_name} ({sel}): {len(els)} found")
            for j, el in enumerate(els[:3]):
                tag = el.evaluate("e => e.tagName")
                cls = el.get_attribute("class") or ""
                txt = (el.inner_text() or "")[:100]
                print(f"    [{j}] <{tag}> class={cls!r} text={txt!r}")

    # Dump outer HTML of the main content area
    print("\n--- Page structure (first 5000 chars of body) ---")
    body_html = page.evaluate("() => document.body.innerHTML.substring(0, 5000)")
    print(body_html)

    # Also check if there's a player page with pagination
    print("\n\n=== Now checking a player page ===")
    page.goto(
        "https://ps.waltheri.net/database/player/Lee%20Sedol/",
        wait_until="networkidle",
        timeout=60000,
    )
    time.sleep(5)

    game_links2 = page.query_selector_all('a[href*="/database/game/"]')
    print(f"Found {len(game_links2)} game links on Lee Sedol's page")

    # Look for pagination on player page
    print("\n--- Player page pagination search ---")
    for sel_name, sel in [
        ("any button", "button"),
        (".pagination", ".pagination"),
        ("[class*=pag]", "[class*=pag]"),
        ("a with >", 'a:has-text(">")'),
        ("a with Next", 'a:has-text("Next")'),
        ("select (dropdown)", "select"),
        ("all links", "a"),
    ]:
        els = page.query_selector_all(sel)
        if els:
            count = len(els)
            print(f"  {sel_name} ({sel}): {count} found")
            if count <= 20:
                for j, el in enumerate(els):
                    tag = el.evaluate("e => e.tagName")
                    cls = el.get_attribute("class") or ""
                    href = el.get_attribute("href") or ""
                    txt = (el.inner_text() or "")[:80]
                    print(f"    [{j}] <{tag}> class={cls!r} href={href!r} text={txt!r}")

    # Dump player page body
    print("\n--- Player page structure (first 5000 chars of body) ---")
    body_html2 = page.evaluate("() => document.body.innerHTML.substring(0, 5000)")
    print(body_html2)

    browser.close()
