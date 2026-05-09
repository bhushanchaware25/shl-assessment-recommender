"""
scraper.py — One-time SHL catalog scraper.

Scrapes Individual Test Solutions (type=1) from the SHL product catalog,
fetching all paginated pages and each assessment's detail page.
Saves structured data to catalog.json.

Usage: python scraper.py
"""

import json
import re
import time
import logging
from typing import Optional
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Test type code mappings
TEST_TYPE_CODES = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def get_page(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup object."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None


def scrape_listing_page(start: int) -> list[dict]:
    """Scrape a single listing page and return list of {name, url} dicts."""
    url = f"{CATALOG_URL}?start={start}&type=1"
    soup = get_page(url)
    if not soup:
        return []

    items = []
    # The catalog items are anchor tags in the main content area
    # They appear as: <a href="/products/product-catalog/view/slug/">Name</a>
    # within the product listing section
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/products/product-catalog/view/" in href:
            name = a.get_text(strip=True)
            if name and len(name) > 1:
                full_url = BASE_URL + href if href.startswith("/") else href
                items.append({"name": name, "url": full_url})

    logger.info(f"  start={start}: found {len(items)} items")
    return items


def scrape_detail_page(item: dict) -> dict:
    """Fetch and parse an assessment's detail page for full metadata."""
    soup = get_page(item["url"])
    if not soup:
        return {**item, "description": "", "test_types": [], "duration": None,
                "remote_testing": False, "adaptive_irt": False}

    # Extract description
    description = ""
    # Try meta description first
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()

    # Try og:description
    if not description:
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            description = og_desc["content"].strip()

    # Try main body text paragraphs
    if not description:
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:
                description = text
                break

    # Extract test type codes (A, B, C, D, E, K, P, S)
    test_types = []
    page_text = soup.get_text()

    # Look for test type indicators in specific patterns
    # SHL uses these letter codes prominently on detail pages
    type_patterns = [
        r'\b(Ability\s*&\s*Aptitude|Ability and Aptitude)\b',
        r'\b(Biodata\s*&\s*Situational|Biodata and Situational)\b',
        r'\b(Competencies)\b',
        r'\b(Development\s*&\s*360|Development and 360)\b',
        r'\b(Assessment Exercises)\b',
        r'\b(Knowledge\s*&\s*Skills|Knowledge and Skills)\b',
        r'\b(Personality\s*&\s*Behavior|Personality and Behavior|Personality & Behaviour)\b',
        r'\b(Simulations)\b',
    ]

    code_map = {
        "Ability": "A",
        "Biodata": "B",
        "Competencies": "C",
        "Development": "D",
        "Assessment Exercises": "E",
        "Knowledge": "K",
        "Personality": "P",
        "Simulations": "S",
    }

    # Also check for letter codes displayed on the page
    # SHL detail pages often list [A] [K] etc in tables
    for code in ["A", "B", "C", "D", "E", "K", "P", "S"]:
        # Look for patterns like " A " or "[A]" or class indicators
        if re.search(rf'\b{code}\b', page_text):
            pass  # too noisy

    # Check table rows for type information
    for td in soup.find_all("td"):
        td_text = td.get_text(strip=True)
        if td_text in TEST_TYPE_CODES:
            test_types.append(td_text)

    # Check for specific filter/tag elements that indicate type
    for span in soup.find_all(["span", "div", "li"], class_=True):
        cls = " ".join(span.get("class", []))
        text = span.get_text(strip=True)
        if text in TEST_TYPE_CODES and text not in test_types:
            test_types.append(text)

    # Infer from name/description if needed
    name_lower = item["name"].lower()
    desc_lower = description.lower()
    combined = name_lower + " " + desc_lower

    if not test_types:
        if any(kw in combined for kw in ["personality", "behaviour", "behavior", "opq", "motivat"]):
            test_types.append("P")
        if any(kw in combined for kw in ["verbal", "numerical", "inductive", "cognitive", "ability", "aptitude", "verify", "reasoning"]):
            test_types.append("A")
        if any(kw in combined for kw in ["situational", "sjt", "judgment", "judgement", "biodata"]):
            test_types.append("B")
        if any(kw in combined for kw in ["competenc", "360"]):
            test_types.append("C")
        if any(kw in combined for kw in ["simulation", "inbox", "in-basket", "e-tray"]):
            test_types.append("S")
        if any(kw in combined for kw in ["knowledge", "skill", "typing", "coding", "programming", "language", "microsoft", "excel", "word", "data entry"]):
            test_types.append("K")

    # Deduplicate
    test_types = list(dict.fromkeys(test_types))

    # Extract duration (look for minutes pattern)
    duration = None
    duration_match = re.search(r'(\d+)\s*(?:minutes?|mins?)', page_text, re.IGNORECASE)
    if duration_match:
        duration = int(duration_match.group(1))

    # Extract remote testing flag
    remote_testing = bool(
        re.search(r'remote\s*testing', page_text, re.IGNORECASE) or
        re.search(r'online\s*test', page_text, re.IGNORECASE)
    )

    # Extract adaptive/IRT flag
    adaptive_irt = bool(
        re.search(r'adaptive', page_text, re.IGNORECASE) or
        re.search(r'\bIRT\b', page_text)
    )

    return {
        "name": item["name"],
        "url": item["url"],
        "description": description,
        "test_types": test_types,
        "duration": duration,
        "remote_testing": remote_testing,
        "adaptive_irt": adaptive_irt,
    }


def get_total_pages() -> int:
    """Determine total number of pages for type=1 (Individual Test Solutions)."""
    soup = get_page(f"{CATALOG_URL}?start=0&type=1")
    if not soup:
        return 32  # fallback to known max

    # Look for last page link
    last_page_link = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "type=1" in href and "start=" in href:
            m = re.search(r"start=(\d+)", href)
            if m:
                start_val = int(m.group(1))
                if last_page_link is None or start_val > last_page_link:
                    last_page_link = start_val

    if last_page_link is not None:
        total_pages = (last_page_link // 12) + 1
        logger.info(f"Detected {total_pages} pages (last start={last_page_link})")
        return total_pages

    return 32  # fallback


def scrape_catalog(output_file: str = "catalog.json") -> None:
    """Main scraping function. Scrapes all Individual Test Solutions."""
    logger.info("=" * 60)
    logger.info("Starting SHL Catalog Scraper — Individual Test Solutions")
    logger.info("=" * 60)

    # Step 1: Get all listing pages
    total_pages = get_total_pages()
    logger.info(f"Total pages to scrape: {total_pages}")

    all_items = []
    seen_urls = set()

    for page_idx in range(total_pages):
        start = page_idx * 12
        logger.info(f"Scraping listing page {page_idx + 1}/{total_pages} (start={start})")
        items = scrape_listing_page(start)

        for item in items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)

        time.sleep(0.5)  # polite delay

    logger.info(f"Found {len(all_items)} unique Individual Test Solutions")

    # Step 2: Fetch detail pages
    catalog = []
    for i, item in enumerate(all_items):
        logger.info(f"Fetching detail {i + 1}/{len(all_items)}: {item['name']}")
        detail = scrape_detail_page(item)
        catalog.append(detail)
        time.sleep(0.3)  # polite delay

    # Step 3: Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Catalog saved to {output_file}")
    logger.info(f"Total assessments: {len(catalog)}")
    logger.info(f"With descriptions: {sum(1 for a in catalog if a['description'])}")
    logger.info(f"With test types: {sum(1 for a in catalog if a['test_types'])}")
    logger.info(f"With duration: {sum(1 for a in catalog if a['duration'])}")
    logger.info("=" * 60)


if __name__ == "__main__":
    scrape_catalog()
