#!/usr/bin/env python3
"""
Download all SGF files from ps.waltheri.net/database/player for all players.

The site is JavaScript-rendered; this script uses Playwright to:
1. Load the player list
2. For each player, load their game list
3. For each game, obtain the SGF download URL and save the file

Dependencies:
    pip install playwright tqdm
    playwright install chromium

Output:
    KataGo/Training_Dataset/All_human_pros/
"""

import argparse
import gc
import re
import threading
import time
from queue import Empty, Queue
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote, unquote, urljoin

try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


BASE_URL = "https://ps.waltheri.net"
PLAYER_LIST_URL = f"{BASE_URL}/database/player/"


AJAX_GAME_URL = f"{BASE_URL}/include/ajax_game.php"
COLOR_MAP = {3: "B", 1: "W"}


def _json_to_sgf(data: dict) -> Optional[str]:
    """Convert waltheri JSON game data to SGF format."""
    size = data.get("size", 19)
    info = data.get("info", {})
    moves = data.get("game", [])
    if not moves:
        return None

    # Build SGF header
    parts = [f"(;FF[4]GM[1]SZ[{size}]"]
    black = info.get("black", {})
    white = info.get("white", {})
    if black.get("name"):
        parts.append(f"PB[{_sgf_escape(black['name'])}]")
    if black.get("rank"):
        parts.append(f"BR[{_sgf_escape(black['rank'])}]")
    if white.get("name"):
        parts.append(f"PW[{_sgf_escape(white['name'])}]")
    if white.get("rank"):
        parts.append(f"WR[{_sgf_escape(white['rank'])}]")
    for key, tag in [("RE", "RE"), ("DT", "DT"), ("EV", "EV"), ("KM", "KM"), ("HA", "HA")]:
        val = info.get(key)
        if val:
            parts.append(f"{tag}[{_sgf_escape(str(val))}]")
    sgf = "".join(parts) + "\n"

    # Parse moves as triplets: (color_code, x, y)
    i = 0
    while i + 2 < len(moves):
        color_code, x, y = moves[i], moves[i + 1], moves[i + 2]
        i += 3
        color = COLOR_MAP.get(color_code)
        if color is None:
            continue
        if 0 <= x < size and 0 <= y < size:
            col = chr(ord("a") + x)
            row = chr(ord("a") + y)
            sgf += f";{color}[{col}{row}]"
        else:
            sgf += f";{color}[]"

    sgf += ")\n"
    return sgf


def _sgf_escape(s: str) -> str:
    """Escape special characters for SGF property values."""
    return s.replace("\\", "\\\\").replace("]", "\\]")


def get_output_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    return (script_dir.parent / "Training_Dataset" / "All_human_pros").resolve()


def load_existing_game_ids(output_dir: Path) -> set:
    """Scan output dir for existing SGFs and extract game IDs (skips re-downloading)."""
    existing = set()
    if not output_dir.exists():
        return existing
    for p in output_dir.glob("*.sgf"):
        m = re.search(r"_(\d+)(?:_\d+)?\.sgf$", p.name)
        if m:
            existing.add(m.group(1))
    return existing


def extract_player_links(page, debug: bool = False) -> List[Tuple[str, str]]:
    """Extract (player_slug, player_name). Site uses name slugs like 'Abe Yoshiteru', not numeric IDs."""
    page.goto(PLAYER_LIST_URL, wait_until="networkidle", timeout=60000)
    time.sleep(5)

    if debug:
        (Path(__file__).parent / "debug_waltheri_player_page.html").write_text(
            page.content(), encoding="utf-8"
        )
        print("Debug: saved page HTML to debug_waltheri_player_page.html")

    links: List[Tuple[str, str]] = []
    seen = set()
    for a in page.query_selector_all('a[href*="/database/player/"]'):
        href = (a.get_attribute("href") or "").strip()
        text = (a.inner_text() or "").strip()
        if not text or text == "Browse by players":
            continue
        m = re.search(r"/player/([^/?#]+)/?$", href)
        if m:
            slug = unquote(m.group(1).strip())
            if len(slug) <= 1:
                continue  # Skip A, B, C... letter index links
            if slug and slug not in seen:
                seen.add(slug)
                links.append((slug, text))
    return links


def _scrape_game_ids(page) -> List[str]:
    """Scrape all game IDs currently in the DOM."""
    ids = []
    for a in page.query_selector_all('a[href*="/database/game/"]'):
        href = a.get_attribute("href") or ""
        m = re.search(r"/database/game/(\d+)", href)
        if m:
            ids.append(m.group(1))
    return list(dict.fromkeys(ids))


def _click_load_more_until_done(page, max_clicks: int = 500) -> None:
    """Click the 'Load more games' button until it disappears or stops producing new results."""
    LOAD_MORE_SEL = "button.btn-lg"
    GAME_LINK_SEL = 'a[href*="/database/game/"]'
    stale_rounds = 0
    for _ in range(max_clicks):
        btn = page.query_selector(LOAD_MORE_SEL)
        if not btn or not btn.is_visible():
            break
        count_before = len(page.query_selector_all(GAME_LINK_SEL))
        try:
            btn.scroll_into_view_if_needed()
            btn.click()
            time.sleep(2)
        except Exception:
            break
        count_after = len(page.query_selector_all(GAME_LINK_SEL))
        if count_after <= count_before:
            stale_rounds += 1
            if stale_rounds >= 3:
                break
        else:
            stale_rounds = 0


def extract_game_ids_from_player_page(page, player_url: str) -> List[str]:
    """Extract all game IDs from a player's page by clicking 'Load more games' repeatedly."""
    page.goto(player_url, wait_until="networkidle", timeout=60000)
    time.sleep(2)
    _click_load_more_until_done(page)
    return _scrape_game_ids(page)


def get_sgf_for_game(game_id: str) -> Optional[str]:
    """Fetch game JSON from ajax_game.php and convert to SGF."""
    import json
    import urllib.request

    url = f"{AJAX_GAME_URL}?id={game_id}"
    req = urllib.request.Request(url, headers={"Referer": f"{BASE_URL}/"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
            return _json_to_sgf(data)
    except Exception:
        return None


def _process_player(
    page,
    player_slug: str,
    player_name: str,
    output_dir: Path,
    limit_games: Optional[int],
    delay: float,
    seen_game_ids: set,
    lock: threading.Lock,
    total_saved_list: list,
    pbar,
) -> None:
    """Process one player: fetch game IDs, download SGFs. Uses shared seen_game_ids and lock."""
    safe_name = re.sub(r"[^\w\-]", "_", player_name)[:60]
    player_url = f"{BASE_URL}/database/player/{quote(player_slug)}/"

    game_ids = extract_game_ids_from_player_page(page, player_url)
    if limit_games:
        game_ids = game_ids[:limit_games]

    for gid in game_ids:
        with lock:
            if gid in seen_game_ids:
                continue
            seen_game_ids.add(gid)

        sgf = get_sgf_for_game(gid)
        if sgf:
            fname = f"{safe_name}_{gid}.sgf"
            out_path = output_dir / fname
            counter = 1
            with lock:
                while out_path.exists():
                    out_path = output_dir / f"{safe_name}_{gid}_{counter}.sgf"
                    counter += 1
            out_path.write_text(sgf, encoding="utf-8")
            with lock:
                total_saved_list[0] += 1
                if pbar:
                    pbar.set_postfix_str(f"saved: {total_saved_list[0]}")
        time.sleep(delay)


def _worker(
    player_queue: Queue,
    output_dir: Path,
    limit_games: Optional[int],
    delay: float,
    seen_game_ids: set,
    lock: threading.Lock,
    total_saved_list: list,
    pbar,
    restart_every: int,
) -> None:
    """Worker thread: own browser, pulls players from queue. Restarts browser every restart_every players to limit memory growth."""
    players_this_session = 0
    while True:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(30000)
            queue_empty = False
            while True:
                try:
                    item = player_queue.get_nowait()
                except Empty:
                    queue_empty = True
                    break
                if item is None:
                    queue_empty = True
                    break
                player_slug, player_name = item
                try:
                    _process_player(
                        page,
                        player_slug,
                        player_name,
                        output_dir,
                        limit_games,
                        delay,
                        seen_game_ids,
                        lock,
                        total_saved_list,
                        pbar,
                    )
                    if pbar:
                        pbar.update(1)
                    players_this_session += 1
                except Exception as e:
                    player_queue.put(item)
                    print(f"\n[retry] {player_name}: {e}")
                player_queue.task_done()
                if restart_every and players_this_session >= restart_every:
                    break
        if queue_empty:
            break
        if players_this_session > 0 and restart_every:
            gc.collect()
            time.sleep(1)
            players_this_session = 0


def download_all(
    output_dir: Path,
    only_player: Optional[str] = None,
    limit_players: Optional[int] = None,
    limit_games_per_player: Optional[int] = None,
    delay: float = 0.5,
    workers: int = 4,
    restart_every: int = 100,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_game_ids = load_existing_game_ids(output_dir)
    if seen_game_ids:
        print(f"Resuming: {len(seen_game_ids)} games already downloaded, skipping those")
    lock = threading.Lock()
    total_saved_list = [0]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(30000)
        players = extract_player_links(page)
        browser.close()

    print(f"Found {len(players)} players")

    if only_player:
        needle = only_player.lower()
        players = [(s, n) for s, n in players if needle in n.lower() or needle in s.lower()]
        print(
            f"Filtered to {len(players)} player(s) matching '{only_player}': {[n for _, n in players]}"
        )
        if not players:
            print("No matching players found. Exiting.")
            return

    if limit_players:
        players = players[:limit_players]
        print(f"Limiting to first {limit_players} players")

    player_queue: Queue = Queue()
    for pl in players:
        player_queue.put(pl)

    pbar = tqdm(total=len(players), desc="Players", unit="player") if TQDM_AVAILABLE else None
    threads = [
        threading.Thread(
            target=_worker,
            args=(
                player_queue,
                output_dir,
                limit_games_per_player,
                delay,
                seen_game_ids,
                lock,
                total_saved_list,
                pbar,
                restart_every,
            ),
        )
        for _ in range(min(workers, len(players)))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if pbar:
        pbar.close()

    print(f"\nDone. Saved {total_saved_list[0]} SGF files to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Waltheri database SGFs")
    parser.add_argument(
        "--output", "-o", help="Output directory (default: Training_Dataset/All_human_pros)"
    )
    parser.add_argument(
        "--only-player",
        type=str,
        help="Only download games for this player (case-insensitive substring match)",
    )
    parser.add_argument("--limit-players", type=int, help="Limit number of players (for testing)")
    parser.add_argument("--limit-games", type=int, help="Limit games per player (for testing)")
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Delay between requests (seconds)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4, help="Parallel browser workers (default: 4)"
    )
    parser.add_argument(
        "--restart-every",
        type=int,
        default=100,
        help="Restart each worker's browser after N players to limit memory growth (default: 100, use 0 to disable)",
    )
    args = parser.parse_args()

    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright is required. Install with:")
        print("  pip install playwright")
        print("  playwright install chromium")
        raise SystemExit(1)

    out = Path(args.output) if args.output else get_output_dir()
    download_all(
        output_dir=out,
        only_player=args.only_player,
        limit_players=args.limit_players,
        limit_games_per_player=args.limit_games,
        delay=args.delay,
        workers=args.workers,
        restart_every=args.restart_every,
    )


if __name__ == "__main__":
    main()
