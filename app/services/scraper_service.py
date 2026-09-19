import re
import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException, status

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_ecommerce_page(url: str, timeout: int = 15) -> str:
    """
    Fetches an e-commerce product page and extracts structured text
    from product specifications, description, and metadata.
    """
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to scrape e-commerce listing at '{url}': {str(e)}",
        )

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove irrelevant script, style, navigation, and footer elements
    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        element.extract()

    extracted_lines = []

    # Extract title
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        extracted_lines.append(f"Product: {title_tag.string.strip()}")

    # Look for common specification tables or definition lists on e-commerce sites
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = [c.get_text(separator=" ", strip=True) for c in row.find_all(["th", "td"])]
            if len(cols) >= 2:
                extracted_lines.append(f"{cols[0]}: {cols[1]}")

    # Look for lists of product specs
    for dl in soup.find_all("dl"):
        dt_list = dl.find_all("dt")
        dd_list = dl.find_all("dd")
        for dt, dd in zip(dt_list, dd_list):
            extracted_lines.append(f"{dt.get_text(strip=True)}: {dd.get_text(strip=True)}")

    # Extract remaining main text
    raw_text = soup.get_text(separator="\n")
    for line in raw_text.splitlines():
        cleaned = line.strip()
        # Keep lines that have substantial text or keywords
        if len(cleaned) > 3 and not cleaned.startswith("{") and not cleaned.endswith("}"):
            extracted_lines.append(cleaned)

    return "\n".join(extracted_lines)
