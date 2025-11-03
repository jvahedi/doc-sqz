# DOCSQZ/html/extract.py
# HTML extraction using readability and BeautifulSoup

from __future__ import annotations
import html
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from readability import Document


def extract_html_structure(html_path: str | Path) -> pd.DataFrame:
    """
    Extract structured text from HTML file using readability for content extraction.

    Args:
        html_path: Path to HTML file

    Returns:
        DataFrame with columns:
            - page: section number (increments at h1/h2 headings)
            - text: extracted text content
            - type: 'heading' for headings, or element tag ('p', 'li', 'td')
            - heading_level: 1-6 for headings, None otherwise

    Example:
        >>> df = extract_html_structure('policy.html')
        >>> df.head()
           page                          text      type  heading_level
        0     1        Student Records Policy   heading              1
        1     1  The educational records...         p           None
        2     2  Military Recruiter Access   heading              2
        3     2  Under FERPA section...            p           None
    """
    html_path = Path(html_path)

    with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Use readability to extract main content (removes nav, footer, ads, etc.)
    doc = Document(content)
    clean_html = doc.summary()

    # Decode HTML entities (&nbsp;, &mdash;, etc.)
    clean_html = html.unescape(clean_html)

    # Parse with BeautifulSoup
    soup = BeautifulSoup(clean_html, 'lxml')

    # Additional cleanup (readability is good but not perfect)
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
        tag.decompose()

    # Extract sections by heading hierarchy
    sections = []
    current_section = 0

    # Process headings, paragraphs, list items, and table cells
    for elem in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td']):
        text = elem.get_text(strip=True)
        if not text:  # Skip empty elements
            continue

        if elem.name.startswith('h'):
            level = int(elem.name[1])
            if level <= 2:  # h1/h2 start new section
                current_section += 1
            sections.append({
                'page': current_section,
                'text': text,
                'type': 'heading',
                'heading_level': level
            })
        else:
            sections.append({
                'page': current_section,
                'text': text,
                'type': elem.name,  # 'p', 'li', or 'td'
                'heading_level': None
            })

    # Return DataFrame or empty DataFrame if no content found
    if not sections:
        return pd.DataFrame(columns=['page', 'text', 'type', 'heading_level'])

    return pd.DataFrame(sections)
