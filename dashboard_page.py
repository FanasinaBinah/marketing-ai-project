from pathlib import Path
import re
import streamlit as st
import streamlit.components.v1 as components

BASE = Path(__file__).parent
html = (BASE / "dashboard" / "dashboard_marketia.html").read_text(encoding="utf-8")

export = BASE / "dashboard" / "dashboard_data.json"
if export.exists():
    donnees = export.read_text(encoding="utf-8").replace("</script>", "<\\/script>")
    html = re.sub(
        r'(<script id="payload" type="application/json">).*?(</script>)',
        lambda m: m.group(1) + donnees + m.group(2),
        html, count=1, flags=re.S,
    )

st.set_page_config(page_title="MarketIA", layout="wide")
components.html(html, height=2400, scrolling=True)