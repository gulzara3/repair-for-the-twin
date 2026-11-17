"""Render Fig. 1 and the graphical abstract from their HTML/SVG sources.

Usage (from the repository root):

    pip install playwright && python -m playwright install chromium
    python scripts/render_diagrams.py

The Inter font gives the published look; any sans-serif fallback works.
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

# (source, output, device scale factor)
JOBS = [
    ("figures/source/fig1_architecture.html", "figures/paper/Fig1_architecture.png", 3),
    ("figures/source/graphical_abstract.html", "figures/paper/Graphical_abstract.png", 2),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    for src, out, scale in JOBS:
        page = browser.new_page(device_scale_factor=scale, viewport={"width": 1500, "height": 1200})
        page.goto(Path(src).resolve().as_uri())
        page.wait_for_timeout(400)
        page.locator("svg#fig").screenshot(path=out)
        page.close()
        print("wrote", out)
    browser.close()
