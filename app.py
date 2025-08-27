import os
import re
import json
from typing import Dict, List, Tuple

import streamlit as st

# --- OpenAI SDK (>=1.0) ---
try:
    from openai import OpenAI
except ImportError:
    st.error("Brak pakietu 'openai'. Upewnij się, że jest w requirements.txt")
    st.stop()

# ========== Konfiguracja ==========
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # 🚀 szybszy model
MAX_CHARS_PER_CHUNK = 6500   # bezpieczny limit na chunk (przybliżenie)
CHUNK_OVERLAP = 200          # miękkie „zakładki” między chunkami

# Pobranie klucza z secrets Streamlita
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

if not OPENAI_API_KEY:
    st.warning("⚠️ Nie znaleziono OPENAI_API_KEY w secrets. "
               "Dodaj go do .streamlit/secrets.toml lub zmiennej środowiskowej.")
client = OpenAI(api_key=OPENAI_API_KEY)


# ========== Prompty ==========
SYSTEM_PROMPT_BASE = """Jesteś asystentem-edytorem tekstów naukowych w języku polskim.
Twoje zadanie: dodać linki Obsidiana w nawiasach [[ ]] do PODANEGO TEKSTU, bez redakcji treści.
Stosuj ścisłe zasady:

1) Linkujemy WSZYSTKIE kluczowe nazwy własne: osoby (historyczne, dynastie), budowle (zamki, pałace, kościoły, wieże), ulice/duże place/topografię,
   instytucje (sejm, urzędy, komisje, muzea), wydarzenia (wojny, unie, bitwy, konfederacje, elekcje).
2) Jeśli forma w tekście jest odmieniona (nie w mianowniku), użyj aliasu: [[LEMMA | forma_z_tekstu]].
   Jeśli forma jest równa mianownikowi, użyj [[LEMMA]].
3) LEMMA (mianownik) dobieraj następująco:
   - osoby: pełne „Imię Nazwisko” + numeracja/dynastia (np. „Zygmunt III Waza”, „Janusz I Starszy”).
   - rody/dynastie: liczba mnoga („Wazowie”, „Piastowie mazowieccy”).
   - budowle: oficjalna polska nazwa (np. „Zamek Królewski w Warszawie”, „Wieża Grodzka”, „Dwór Wielki”).
   - wydarzenia: forma słownikowa (np. „Unia lubelska”, „Potop szwedzki”, „Bitwa pod Kłuszynem”).
   - instytucje i topografia: forma słownikowa z kwalifikatorami miejsca (np. „Kolegiata św. Jana w Warszawie”, „Plac Zamkowy”).
4) NIE modyfikuj istniejących [[wikilinków]], linków Markdown, obrazków `![]()`, tytułów obrazów/cytatów w nawiasach, ani treści bloków kodu.
5) Zachowaj oryginalne odstępy, interpunkcję i diakrytykę. Nie skracaj, nie poprawiaj stylu.
6) Jeżeli dany byt był już zmapowany w KNOWN_ENTITIES (lista poniżej), używaj dokładnie tej lemmy (spójność całego dokumentu).
7) Nie linkuj zwykłych rzeczowników pospolitych ani zbyt ogólnych pojęć.

ZWROT:
- Najpierw zwróć WYŁĄCZNIE przetworzony tekst.
- Następnie w nowej linii daj separator:
-----ENTITY_MAP_JSON-----
i JSON z listą wykrytych/aktualizowanych bytów (pole "entities": [{surface, lemma, type}...]).
- Zakończ:
-----END-----

KNOWN_ENTITIES (priorytet spójności):
"""

# typowa heurystyka wykrywania bloków kodu/obrazków/linków: nie zmieniamy ich
FENCE_RE = re.compile(r"(```.*?```|`.*?`)", re.DOTALL)
MARKDOWN_LINK_OR_IMAGE_RE = re.compile(r"(!?\[[^\]]*\]\([^)]+\))")


# ========== Pomocnicze ==========

def count_total_chunks(full_text: str) -> int:
    segments = compound_protection(full_text)
    total = 0
    for seg, protected in segments:
        if protected or not seg.strip():
            continue
        total += len(chunk_text_by_paragraphs(seg, MAX_CHARS_PER_CHUNK, CHUNK_OVERLAP))
    return total
    
def split_keep_delimiters(text: str, pattern: re.Pattern) -> List[Tuple[str, bool]]:
    """
    Dzieli tekst na segmenty: [(segment, is_protected), ...]
    is_protected=True oznacza fragment, którego nie modyfikujemy (np. kod, link MD).
    """
    protected = []
    last = 0
    for m in pattern.finditer(text):
        # fragment zwykły
        if m.start() > last:
            protected.append((text[last:m.start()], False))
        # fragment chroniony
        protected.append((m.group(0), True))
        last = m.end()
    if last < len(text):
        protected.append((text[last:], False))
    return protected


def compound_protection(text: str) -> List[Tuple[str, bool]]:
    """Zabezpiecza jednocześnie bloki kodu i konstrukcje linków/obrazków Markdown."""
    segs = split_keep_delimiters(text, FENCE_RE)
    out = []
    for seg, prot in segs:
        if prot:
            out.append((seg, True))
        else:
            # wewnątrz zwykłego segmentu zabezpiecz dodatkowo ![]() i []()
            sub = split_keep_delimiters(seg, MARKDOWN_LINK_OR_IMAGE_RE)
            out.extend(sub)
    return out


def chunk_text_by_paragraphs(text: str, max_chars: int, overlap: int) -> List[str]:
    paras = text.split("\n\n")
    chunks = []
    cur = []
    cur_len = 0
    for p in paras:
        p_block = (p + "\n\n")
        if cur_len + len(p_block) > max_chars and cur:
            chunk_text = "".join(cur).rstrip() + "\n"
            chunks.append(chunk_text)
            # zakładka (weź końcówkę poprzedniego chunku)
            tail = chunk_text[-overlap:] if overlap > 0 else ""
            cur = [tail, p_block]
            cur_len = len(tail) + len(p_block)
        else:
            cur.append(p_block)
            cur_len += len(p_block)
    if cur:
        chunks.append("".join(cur))
    return chunks


def call_openai_linker(raw_text: str, known_entities: Dict[str, str], temperature: float = 0.1) -> Tuple[str, List[Dict]]:
    """
    Wywołuje model: zwraca (linked_text, entities_list)
    known_entities: dict lemma->list_of_known_surfaces (przekazujemy w prompt)
    """
    known_list = []
    for lemma, surfaces in known_entities.items():
        if isinstance(surfaces, list):
            known_list.append({"lemma": lemma, "surfaces": surfaces})
        else:
            known_list.append({"lemma": lemma, "surfaces": [surfaces]})

    sys = SYSTEM_PROMPT_BASE + json.dumps(known_list, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": raw_text}
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=messages
    )
    content = resp.choices[0].message.content

    # Parsowanie wyniku i mapy
    if "-----ENTITY_MAP_JSON-----" in content and "-----END-----" in content:
        text_part, json_part = content.split("-----ENTITY_MAP_JSON-----", 1)
        json_str = json_part.split("-----END-----", 1)[0].strip()
        try:
            ent_data = json.loads(json_str)
            entities = ent_data.get("entities", [])
        except Exception:
            entities = []
        return text_part.strip(), entities
    else:
        # fallback: brak JSON-a
        return content.strip(), []


def update_known_entities(known: Dict[str, List[str]], new_items: List[Dict]) -> Dict[str, List[str]]:
    for item in new_items:
        lemma = item.get("lemma")
        surface = item.get("surface")
        if not lemma or not surface:
            continue
        if lemma not in known:
            known[lemma] = []
        if surface not in known[lemma]:
            known[lemma].append(surface)
    return known


def process_text(full_text: str, temperature: float = 0.1) -> Tuple[str, Dict[str, List[str]]]:
    """
    Całościowy pipeline:
    - chronimy kod/linki MD
    - chunkujemy zwykłe segmenty
    - wywołujemy model per chunk z narastającą mapą encji
    - składamy wynik
    - AKTUALIZUJEMY pasek postępu Streamlit
    """
    segments = compound_protection(full_text)
    known_entities: Dict[str, List[str]] = {}
    out_segments = []

    # Pasek postępu (globalnie wszystkie chunki)
    total_chunks = count_total_chunks(full_text)
    done_chunks = 0
    progress = st.progress(0, text="Przygotowuję…")
    status_placeholder = st.empty()

    for seg, protected in segments:
        if protected or not seg.strip():
            out_segments.append(seg)
            continue

        chunks = chunk_text_by_paragraphs(seg, MAX_CHARS_PER_CHUNK, CHUNK_OVERLAP)
        processed_parts = []
        for i, ch in enumerate(chunks, start=1):
            with st.spinner(f"Przetwarzam fragment {done_chunks + 1}/{total_chunks}…"):
                linked, ents = call_openai_linker(ch, known_entities, temperature=temperature)
                processed_parts.append(linked)
                known_entities = update_known_entities(known_entities, ents)
            done_chunks += 1
            pct = int((done_chunks / max(total_chunks, 1)) * 100)
            progress.progress(pct, text=f"Postęp: {pct}% ({done_chunks}/{total_chunks})")

        out_segments.append("".join(processed_parts))

    progress.progress(100, text="Zakończono ✅")
    status_placeholder.success("Przetwarzanie ukończone.")
    return "".join(out_segments), known_entities



# ========== UI Streamlit ==========
st.set_page_config(page_title="Obsidian Linker (PL)", page_icon="🧭", layout="wide")

st.title("🧭 Obsidian Linker (PL)")
st.caption("Automatyczne dodawanie linków [[ ]] (osoby, miejsca, wydarzenia) z aliasami w mianowniku — zgodnie z Twoją logiką.")

with st.sidebar:
    st.subheader("Ustawienia")
    temp = st.slider("Temperatura (0 = bardzo zachowawczo)", 0.0, 1.0, 0.1, 0.05)
    st.divider()
    st.markdown("**Model**")
    st.code(MODEL)
    st.markdown("**Limit chunku**")
    st.code(f"{MAX_CHARS_PER_CHUNK} znaków")
    st.markdown("**Overlap**")
    st.code(f"{CHUNK_OVERLAP} znaków")
    st.divider()
    st.markdown("🔐 Klucz OpenAI pobierany z `st.secrets['OPENAI_API_KEY']`.")

st.markdown("### Wejście")
sample = st.toggle("Wstaw przykładowy fragment", value=False)
default_text = ""
if sample:
    default_text = (
        "Pierwszym królem polskim goszczącym w Zamku Warszawskim był Władysław Jagiełło. "
        "W 1526 roku Zygmunt I przejął Mazowsze po śmierci książąt Janusza i Stanisława. "
        "W 1569 roku Unia lubelska wyznaczyła Warszawę i Zamek na stałe miejsce obrad sejmu."
    )

input_text = st.text_area(
    "Wklej tekst (.md/.txt, bez limitu długości – aplikacja pociągnie w częściach):",
    value=default_text, height=260
)

uploaded = st.file_uploader("…lub wgraj plik .md / .txt", type=["md", "txt"])
if uploaded is not None:
    input_text = uploaded.read().decode("utf-8")

colA, colB = st.columns([1, 1])
with colA:
    run = st.button("🚀 Przetwórz", type="primary")
with colB:
    clear = st.button("🧹 Wyczyść pamięć encji tej sesji")

# Pamięć encji trzymamy w tle (dla spójności), ale jej nie wyświetlamy
if "known_entities_session" not in st.session_state or clear:
    st.session_state.known_entities_session = {}

if run:
    if not input_text.strip():
        st.warning("Wklej tekst lub wgraj plik.")
        st.stop()
    if not OPENAI_API_KEY:
        st.error("Brak OPENAI_API_KEY. Uzupełnij `.streamlit/secrets.toml`.")
        st.stop()

    # Przetwarzanie (z paskiem postępu w process_text, jeśli dodałeś)
    linked_text, new_map = process_text(input_text, temperature=temp)

    # Aktualizacja pamięci encji (bez wyświetlania)
    for lemma, surfaces in new_map.items():
        if lemma not in st.session_state.known_entities_session:
            st.session_state.known_entities_session[lemma] = []
        for s in surfaces:
            if s not in st.session_state.known_entities_session[lemma]:
                st.session_state.known_entities_session[lemma].append(s)

    # --- Ustal nazwę pliku wynikowego ---
    def slugify(name: str) -> str:
        s = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
        s = re.sub(r"[\s_-]+", "-", s)
        return s or "podlinkowany"

    if uploaded is not None and getattr(uploaded, "name", ""):
        base = uploaded.name.rsplit(".", 1)[0]
        suggested_name = slugify(base)
    else:
        head_line = input_text.strip().splitlines()[0] if input_text.strip() else "podlinkowany"
        head_line = head_line.lstrip("# ").strip()
        suggested_name = slugify(" ".join(head_line.split()[:6]))

    final_md = linked_text  # bez front matter

    st.success("Gotowe! Poniżej wynik.")
    st.markdown("### Wynik (`.md`)")
    st.text_area("Podlinkowany tekst", value=linked_text, height=320)

    st.download_button(
        "⬇️ Pobierz jako Markdown (.md)",
        data=final_md.encode("utf-8"),
        file_name=f"{suggested_name}.md",
        mime="text/markdown"
    )

else:
    st.info("Ustaw parametry, wklej tekst i kliknij **Przetwórz**. "
            "Aplikacja doda linki i zadba o aliasy w mianowniku.")
