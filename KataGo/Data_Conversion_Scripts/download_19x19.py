#!/usr/bin/env python3
"""
Download all 19x19 SGF files from 棋谱库 (Go game record library).

Workflow:
  1. Opens browser (headed) so you can log in
  2. After you press Enter, for each page:
     - Clicks each game box in the list
     - Waits for the detail panel (right side) to load
     - Clicks 下载 to download the .sgf
     - When page is done, clicks 上一页 to go to the previous page

Usage:
    python download_19x19.py --url "https://example.com/..." --output ./sgf_output
    python download_19x19.py --url "https://example.com/..." --first-page-only  # test run

Dependencies:
    pip install playwright
    playwright install chromium

Selectors (edit these if the site structure differs):
    GAME_BOX_SELECTOR  - each clickable game row
    DOWNLOAD_BTN_TEXT  - text of the download button (default: 下载)
    PREV_PAGE_TEXT     - text of previous-page link (default: 上一页)
"""

import argparse
import re
import time
from pathlib import Path
from typing import Optional

try:
    from playwright.sync_api import sync_playwright, Page, Download

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


# ── Selectors (adjust if your site uses different structure) ──
# Game boxes: direct children of list only. Tried in order; first with ≤MAX_BOXES wins.
# 19x19.com and similar: typically 5-20 games per page.
GAME_BOX_SELECTORS = [
    "ul.ant-list-items > li",
    ".ant-list-items > .ant-list-item",
    "[class*='SgfList'] > div > div",
    "[class*='sgf-list'] > div",
    "[class*='SgfList'] > div",
    "div[class*='List'] > div[class*='Item']",
    "[role='list'] > [role='listitem']",
]
MAX_BOXES_PER_PAGE = 50  # If selector returns more, it's wrong (e.g. nested items)
DOWNLOAD_BTN_TEXT = "下载"
PREV_PAGE_TEXT = "上一页"
# Delay between actions
CLICK_DELAY = 1.5
DETAIL_WAIT = 3.0  # Wait for detail panel to load
# ─────────────────────────────────────────────────────────────────────────────


def get_output_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    return (script_dir.parent / "Training_Dataset" / "19x19_sgfs").resolve()


def safe_filename(s: str) -> str:
    """Make a safe filename from game title."""
    return re.sub(r"[^\w\-_\s\u4e00-\u9fff]", "_", s).strip()[:100].strip()


def wait_for_login(page: Page) -> None:
    """Pause so user can log in manually."""
    print("\n" + "=" * 60)
    print("Browser is open. Please log in now.")
    print("When you're on the game list page (大赛棋谱 or similar), press Enter here to continue.")
    print("=" * 60)
    input()


def get_game_boxes(page: Page, selector_override: Optional[str] = None) -> list:
    """Get game box elements (only direct list rows, not nested items)."""
    if selector_override:
        for sel in [s.strip() for s in selector_override.split(",")]:
            if sel:
                els = page.locator(sel).all()
                if els:
                    return els
        return []
    for sel in GAME_BOX_SELECTORS:
        els = page.locator(sel).all()
        if els and len(els) <= MAX_BOXES_PER_PAGE:
            return els
        if els and len(els) > MAX_BOXES_PER_PAGE:
            continue  # Too many = wrong selector
    return []


def process_one_page(
    page: Page,
    output_dir: Path,
    first_page_only: bool,
    selector_override: Optional[str] = None,
) -> bool:
    """
    Process all games on the current page.
    Returns True if we should continue to the next (previous) page, False to stop.
    """
    boxes = get_game_boxes(page, selector_override)
    if not boxes:
        print(
            "  No game boxes found. Try --selector with a CSS selector for game rows "
            "(e.g. inspect the list in devtools)."
        )
        return False

    print(f"  Found {len(boxes)} game(s) on this page.")

    for i, box in enumerate(boxes):
        try:
            box.scroll_into_view_if_needed()
            time.sleep(0.3)
            box.click()
            time.sleep(DETAIL_WAIT)

            # Click 下载 (button or link)
            btn = (
                page.get_by_role("button", name=DOWNLOAD_BTN_TEXT)
                .or_(page.get_by_role("link", name=DOWNLOAD_BTN_TEXT))
                .or_(page.get_by_text(DOWNLOAD_BTN_TEXT, exact=False))
                .first
            )
            if btn.is_visible(timeout=5000):
                with page.expect_download(timeout=15000) as download_info:
                    btn.click()
                download: Download = download_info.value
                path = download.path()
                if path and Path(path).exists():
                    suggested = download.suggested_filename or f"game_{i}.sgf"
                    if not suggested.endswith(".sgf"):
                        suggested = suggested + ".sgf" if "." not in suggested else f"game_{i}.sgf"
                    out_path = output_dir / suggested
                    counter = 1
                    while out_path.exists():
                        stem = Path(suggested).stem
                        out_path = output_dir / f"{stem}_{counter}.sgf"
                        counter += 1
                    download.save_as(out_path)
                    print(f"    Saved: {out_path.name}")
            else:
                print(f"    [skip] 下载 button not found for game {i + 1}")
        except Exception as e:
            print(f"    [error] game {i + 1}: {e}")

        time.sleep(CLICK_DELAY)

    if first_page_only:
        return False

    # Go to previous page
    prev_link = page.get_by_text(PREV_PAGE_TEXT).first
    if prev_link.is_visible(timeout=2000):
        try:
            prev_link.click()
            time.sleep(2)
            return True
        except Exception as e:
            print(f"  Could not go to previous page: {e}")
            return False
    else:
        print("  No 上一页 link found (maybe on first page). Stopping.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 19x19 SGF files from 棋谱库 (login required)"
    )
    parser.add_argument(
        "--url",
        "-u",
        required=True,
        help="URL of the game list page (e.g. 大赛棋谱). You must navigate here after login.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=f"Output directory (default: {get_output_dir()})",
    )
    parser.add_argument(
        "--first-page-only",
        action="store_true",
        help="Only process the first page (for testing)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Stop after processing this many pages (default: no limit)",
    )
    parser.add_argument(
        "--selector",
        type=str,
        default=None,
        help="Override GAME_BOX_SELECTOR (CSS selector for game rows, comma-separated for fallbacks)",
    )
    args = parser.parse_args()

    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright is required:")
        print("  pip install playwright")
        print("  playwright install chromium")
        raise SystemExit(1)

    output_dir = args.output or get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    with sync_playwright() as p:
        # Headed so user can log in
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(30000)

        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)

        wait_for_login(page)

        page_num = 0
        while True:
            page_num += 1
            if args.max_pages and page_num > args.max_pages:
                print(f"Reached --max-pages={args.max_pages}. Stopping.")
                break
            print(f"\n--- Page {page_num} ---")
            if not process_one_page(page, output_dir, args.first_page_only, args.selector):
                break

        time.sleep(1)
        browser.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
