import csv
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple, Set
import html

import streamlit as st

try:
    import pyperclip
    HAS_PYPERCLIP = True
except Exception:
    HAS_PYPERCLIP = False


# ------------------ CONFIG ------------------

DEFAULT_CSV = Path("organe.csv")

HEADER_KUERZEL = "kuerzel"
HEADER_NUMMER = "item_order"
HEADER_TEXT = "text"
HEADER_ACTIVE = "active"
HEADER_BILATERAL = "bilateral"
HEADER_ORGAN = "organ"

TRUTHY = {"1", "true", "yes", "y", "ja", "j"}


# ------------------ DATA MODEL ------------------

@dataclass(frozen=True)
class Item:
    kuerzel: str
    nummer: int
    text: str
    bilateral: bool
    active: bool = True
    organ: str = ""


@dataclass(frozen=True)
class Entry:
    side: Optional[str]  # "LINKS", "RECHTS", or None
    text: str


# ------------------ HELPERS ------------------

def parse_bool(value: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in TRUTHY


def iter_items_from_csv(csv_path: Path, *, include_active: bool = False) -> Iterable[Item]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kuerzel = (row.get(HEADER_KUERZEL) or "").strip().lower()
            if not kuerzel:
                continue

            nummer_raw = (row.get(HEADER_NUMMER) or "").strip()
            if not nummer_raw:
                continue
            nummer = int(nummer_raw)

            text = (row.get(HEADER_TEXT) or "").strip()
            organ = (row.get(HEADER_ORGAN) or "").strip()
            bilateral = parse_bool(row.get(HEADER_BILATERAL, ""), default=False)

            active = True
            if include_active:
                active = parse_bool(row.get(HEADER_ACTIVE, "1"), default=True)

            yield Item(
                kuerzel=kuerzel,
                nummer=nummer,
                text=text,
                bilateral=bilateral,
                active=active,
                organ=organ,
            )

def copy_button(text: str, label: str = "In Zwischenablage kopieren"):
    escaped = html.escape(text).replace("\n", "\\n")
    st.components.v1.html(
        f"""
        <button id="copybtn" style="padding:0.5rem 0.75rem; border-radius:0.5rem; border:1px solid #ccc; cursor:pointer;">
            {label}
        </button>
        <span id="copystatus" style="margin-left:0.75rem;"></span>
        <script>
        const text = "{escaped}";
        const btn = document.getElementById("copybtn");
        const status = document.getElementById("copystatus");

        btn.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText(text.replace(/\\n/g, "\\n"));
                status.textContent = "Kopiert ✓";
                setTimeout(() => status.textContent = "", 2000);
            }} catch (e) {{
                status.textContent = "Kopieren fehlgeschlagen (Browser blockiert).";
                setTimeout(() => status.textContent = "", 4000);
            }}
        }});
        </script>
        """,
        height=60,
    )

def load_organs_menu(csv_path: Path) -> List[Tuple[str, str]]:
    """Returns list of (organ, kuerzel) pairs."""
    pairs: Set[Tuple[str, str]] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            organ = (row.get(HEADER_ORGAN) or "").strip()
            kuerzel = (row.get(HEADER_KUERZEL) or "").strip().lower()
            if organ and kuerzel:
                pairs.add((organ, kuerzel))
    return sorted(pairs, key=lambda x: (x[0].lower(), x[1]))


def items_for_organs(
    selected_organs: List[str],
    *,
    csv_path: Path,
    use_active: bool,
) -> List[Item]:
    selected_set = {o.strip() for o in selected_organs if o.strip()}
    items = [
        it for it in iter_items_from_csv(csv_path, include_active=use_active)
        if (it.organ in selected_set) and (not use_active or it.active)
    ]
    # stable order: organ then nummer (and then kuerzel)
    return sorted(items, key=lambda it: (it.organ.lower(), it.nummer, it.kuerzel))


def build_summary_text(normal: List[Entry], pathological: List[Entry]) -> str:
    def bucket(entries: List[Entry]) -> Dict[Optional[str], List[str]]:
        d: Dict[Optional[str], List[str]] = defaultdict(list)
        for e in entries:
            d[e.side].append(e.text)
        return d

    pn = bucket(pathological)
    nn = bucket(normal)

    patho_parts: List[str] = []
    if pn.get("LINKS"):
        patho_parts.append(f"LINKS {', '.join(pn['LINKS'])}")
    if pn.get("RECHTS"):
        patho_parts.append(f"RECHTS {', '.join(pn['RECHTS'])}")
    if pn.get(None):
        patho_parts.append(", ".join(pn[None]))

    normal_parts: List[str] = []
    if nn.get("LINKS"):
        normal_parts.append(f"LINKS ({', '.join(nn['LINKS'])})")
    if nn.get("RECHTS"):
        normal_parts.append(f"RECHTS ({', '.join(nn['RECHTS'])})")
    if nn.get(None):
        normal_parts.append(", ".join(nn[None]))

    patho_line = " ".join(patho_parts) if patho_parts else "-"
    normal_line = "; ".join(normal_parts) if normal_parts else "-"

    return f"Pathologisch: {patho_line}\nNormal: {normal_line}"


# ------------------ STREAMLIT APP ------------------

st.set_page_config(page_title="Befund-Zusammenfassung", layout="wide")
st.title("Interaktive klinische Befund-Zusammenfassung")

with st.sidebar:
    st.header("Datenquelle")
    uploaded = st.file_uploader("CSV hochladen", type=["csv"])
    use_active = st.checkbox("Nur aktive Einträge", value=False)

    st.divider()
    st.header("Aktionen")
    if st.button("Alle Eingaben zurücksetzen"):
        # keep uploaded path, clear the rest
        keep = st.session_state.get("_tmp_csv_path")
        st.session_state.clear()
        if keep:
            st.session_state["_tmp_csv_path"] = keep
        st.rerun()

# Load CSV path
csv_path: Optional[Path] = None
if uploaded is not None:
    tmp = Path(st.session_state.get("_tmp_csv_path", "uploaded_organe.csv"))
    tmp.write_bytes(uploaded.getvalue())
    st.session_state["_tmp_csv_path"] = str(tmp)
    csv_path = tmp
else:
    if DEFAULT_CSV.exists():
        csv_path = DEFAULT_CSV

if csv_path is None:
    st.info("Bitte CSV hochladen (oder organe.csv neben app.py ablegen).")
    st.stop()

# Load organs menu
try:
    menu = load_organs_menu(csv_path)
except Exception as e:
    st.error(f"Konnte CSV nicht lesen: {e}")
    st.stop()

organs = sorted({o for o, _ in menu}, key=lambda s: s.lower())
if not organs:
    st.error("Keine Organe gefunden. Prüfe CSV-Spalten 'organ' und 'kuerzel'.")
    st.stop()


selected_organs = st.pills(
    "Welche Organe sind gewünscht?",
    options=organs,
    selection_mode="multi",
    width="stretch",
)
if not selected_organs:
    st.info("Wähle mindestens ein Organ.")
    st.stop()

items = items_for_organs(selected_organs, csv_path=csv_path, use_active=use_active)
if not items:
    st.warning("Keine Einträge für diese Organe gefunden.")
    st.stop()

# Group display by organ
st.subheader("Befund-Optionen (per Mausklick pathologisch markieren)")
items_by_organ: Dict[str, List[Item]] = defaultdict(list)
for it in items:
    items_by_organ[it.organ].append(it)

normal: List[Entry] = []
pathological: List[Entry] = []

for organ in sorted(items_by_organ.keys(), key=lambda s: s.lower()):
    st.markdown(f"### {organ}")
    for it in items_by_organ[organ]:
        # key base to keep stable
        base = f"{organ}__{it.kuerzel}__{it.nummer}"

        # show the item text
        st.markdown(f"**{it.nummer}** — {it.text}")

        if it.bilateral:
            c1, c2, c3 = st.columns([1, 1, 3])

            with c1:
                left_patho = st.checkbox("LINKS pathologisch", key=f"{base}__L_chk")
            with c2:
                right_patho = st.checkbox("RECHTS pathologisch", key=f"{base}__R_chk")

            left_note = ""
            right_note = ""

            with c3:
                if left_patho:
                    left_note = st.text_input("LINKS Text (leer = normal)", key=f"{base}__L_txt")
                if right_patho:
                    right_note = st.text_input("RECHTS Text (leer = normal)", key=f"{base}__R_txt")

            (pathological if left_patho else normal).append(
                Entry("LINKS", (left_note.strip() if left_patho else "") or it.text)
            )
            (pathological if right_patho else normal).append(
                Entry("RECHTS", (right_note.strip() if right_patho else "") or it.text)
            )

        else:
            cols = st.columns([1, 3])
            with cols[0]:
                is_patho = st.checkbox("Pathologisch", key=f"{base}__chk")
            note = ""
            with cols[1]:
                if is_patho:
                    note = st.text_input("Text (leer = normal)", key=f"{base}__txt")

            (pathological if is_patho else normal).append(
                Entry(None, (note.strip() if is_patho else "") or it.text)
            )

        st.divider()

out = build_summary_text(normal, pathological)

st.subheader("Ergebnis")
st.code(out, language="text")
copy_button(out)

colA, colB = st.columns([1, 2])
with colA:
    if st.button("In Zwischenablage kopieren", disabled=not HAS_PYPERCLIP):
        pyperclip.copy(out)
        st.success("Kopiert. (Strg+V zum Einfügen)")
with colB:
    if not HAS_PYPERCLIP:
        st.caption("Hinweis: Installiere pyperclip für Clipboard-Button: `pip install pyperclip`")
