import os
import re
import io
import json
import zipfile
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

# ➕ NOWE: Prompt do minimalnej korekty OCR/PDF (bez redakcji merytorycznej/stylowej)
SYSTEM_PROMPT_OCR = """Jesteś asystentem do minimalnej korekty tekstu po OCR/PDF w języku polskim.
Twoje zadanie: usunąć wyłącznie ARTEFAKTY formatowania, zachowując treść i styl bez redakcji merytorycznej.
Ściśle stosuj poniższe zasady:

A) Zachowaj oryginalną strukturę dokumentu: nagłówki, akapity, listy, cytaty. Nie zmieniaj kolejności treści.
B) NIE modyfikuj istniejących [[wikilinków]], linków Markdown, obrazków `![]()`, ani bloków kodu (```...``` lub `...`).
C) Usuń typowe artefakty OCR/PDF:
   - dzielenie wyrazów twardymi łącznikami na końcu linii (np. „techno-\nlogie” → „technologie”),
   - niezamierzone złamania w środku zdania/wyrazu,
   - zduplikowane spacje, przypadkowe tabulatory,
   - przypadkowe śmieciowe znaki (np. pojedyncze „.” na osobnych liniach, losowe znaki kontrolne),
   - „pływające” nagłówki i podpisy, jeśli ewidentnie należą do sąsiedniego akapitu.
D) Nie poprawiaj literówek ani interpunkcji, chyba że są skutkiem oczywistego artefaktu OCR (np. „wRzymie” → „w Rzymie”).
E) Nie zmieniaj sensu, nie skracaj, nie dopisuj.

ZWROT:
- Zwróć WYŁĄCZNIE oczyszczony tekst (bez dodatkowych komentarzy).
"""

# typowa heurystyka wykrywania bloków kodu/obrazków/linków: nie zmieniamy ich
FENCE_RE = re.compile(r"(```.*?```|`.*?`)", re.DOTALL)
MARKDOWN_LINK_OR_IMAGE_RE = re.compile(r"(!?\[[^\]]*\]\([^)]+\))")


# ========== Pomocnicze ==========

HDR_RE = re.compile(r"^(#{1,3})\s+(.+)$", flags=re.MULTILINE)

def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s or "podlinkowany"

def split_markdown_sections(text: str) -> List[Dict]:
    """
    Dzieli tekst po nagłówkach # / ## / ###.
    Zwraca listę: [{"title": "...", "content": "..."}].
    Jeżeli brak nagłówków -> 1 sekcja z tytułem z 1. linii albo z pierwszych słów.
    """
    positions = [(m.start(), m.group(1), m.group(2).strip()) for m in HDR_RE.finditer(text)]
    sections = []
    if not positions:
        head_line = text.strip().splitlines()[0] if text.strip() else "notatka"
        title = head_line.lstrip("# ").strip() or "notatka"
        sections.append({"title": title, "content": text})
        return sections

    positions.append((len(text), "", ""))  # sentinel
    for i in range(len(positions)-1):
        start, lvl, title = positions[i]
        end, _, _ = positions[i+1]
        block = text[start:end].rstrip()
        title = title or "sekcja"
        sections.append({"title": title, "content": block.strip() + "\n"})
    return sections

def count_total_chunks_multi(sections: List[Dict]) -> int:
    total = 0
    for s in sections:
        total += count_total_chunks(s["content"])
    return total

def split_keep_delimiters(text: str, pattern: re.Pattern) -> List[Tuple[str, bool]]:
    """
    Dzieli tekst na segmenty: [(segment, is_protected), ...]
    is_protected=True oznacza fragment, którego nie modyfikujemy (np. kod, link MD).
    """
    protected = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            protected.append((text[last:m.start()], False))
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
            tail = chunk_text[-overlap:] if overlap > 0 else ""
            cur = [tail, p_block]
            cur_len = len(tail) + len(p_block)
        else:
            cur.append(p_block)
            cur_len += len(p_block)
    if cur:
        chunks.append("".join(cur))
    return chunks


# ========== Wezwania do OpenAI ==========

def call_openai_linker(raw_text: str, known_entities: Dict[str, List[str]], temperature: float = 0.1) -> Tuple[str, List[Dict]]:
    """
    Zwraca (linked_text, entities_list)
    known_entities: dict lemma -> list_of_known_surfaces (dla spójności)
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
        return content.strip(), []

# ➕ NOWE: czyszczenie OCR/PDF (minimalne zmiany)
def call_openai_cleaner(raw_text: str, temperature: float = 0.0) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_OCR},
        {"role": "user", "content": raw_text}
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=messages
    )
    return resp.choices[0].message.content.strip()


# ========== Pipeline'y ==========

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

def count_total_chunks(full_text: str) -> int:
    segments = compound_protection(full_text)
    total = 0
    for seg, protected in segments:
        if protected or not seg.strip():
            continue
        total += len(chunk_text_by_paragraphs(seg, MAX_CHARS_PER_CHUNK, CHUNK_OVERLAP))
    return total

# ➕ NOWE: samodzielny pipeline do oczyszczania OCR/PDF (bez mapy encji)
def apply_ocr_cleanup(full_text: str, temperature: float = 0.0) -> str:
    segments = compound_protection(full_text)
    out_segments = []

    total_chunks = count_total_chunks(full_text)
    done_chunks = 0
    progress = st.progress(0, text="Czyszczę artefakty OCR/PDF…")

    for seg, protected in segments:
        if protected or not seg.strip():
            out_segments.append(seg)
            continue
        chunks = chunk_text_by_paragraphs(seg, MAX_CHARS_PER_CHUNK, CHUNK_OVERLAP)
        cleaned_parts = []
        for ch in chunks:
            with st.spinner(f"Oczyszczanie fragmentu {done_chunks + 1}/{total_chunks}…"):
                cleaned = call_openai_cleaner(ch, temperature=temperature)
                cleaned_parts.append(cleaned)
            done_chunks += 1
            pct = int((done_chunks / max(total_chunks, 1)) * 100)
            progress.progress(pct, text=f"Postęp: {pct}% ({done_chunks}/{total_chunks})")
        out_segments.append("".join(cleaned_parts))

    progress.progress(100, text="Czyszczenie zakończone ✅")
    return "".join(out_segments)

def process_text_with_map(full_text: str, temperature: float, initial_map: Dict[str, List[str]]) -> Tuple[str, Dict[str, List[str]]]:
    segments = compound_protection(full_text)
    known_entities: Dict[str, List[str]] = {k: list(v) for k, v in (initial_map or {}).items()}
    out_segments = []

    total_chunks = count_total_chunks(full_text)
    done_chunks = 0
    progress = st.progress(0, text="Przygotowuję…")

    for seg, protected in segments:
        if protected or not seg.strip():
            out_segments.append(seg)
            continue
        chunks = chunk_text_by_paragraphs(seg, MAX_CHARS_PER_CHUNK, CHUNK_OVERLAP)
        processed_parts = []
        for ch in chunks:
            with st.spinner(f"Przetwarzam fragment {done_chunks + 1}/{total_chunks}…"):
                linked, ents = call_openai_linker(ch, known_entities, temperature=temperature)
                processed_parts.append(linked)
                known_entities = update_known_entities(known_entities, ents)
            done_chunks += 1
            pct = int((done_chunks / max(total_chunks, 1)) * 100)
            progress.progress(pct, text=f"Postęp: {pct}% ({done_chunks}/{total_chunks})")
        out_segments.append("".join(processed_parts))

    progress.progress(100, text="Zakończono ✅")
    return "".join(out_segments), known_entities

def process_text(full_text: str, temperature: float = 0.1) -> Tuple[str, Dict[str, List[str]]]:
    segments = compound_protection(full_text)
    known_entities: Dict[str, List[str]] = {}
    out_segments = []

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
st.caption("Automatyczne dodawanie linków [[ ]] i (opcjonalnie) minimalne czyszczenie artefaktów OCR/PDF bez redakcji.")

with st.sidebar:
    st.subheader("Ustawienia")
    temp = st.slider("Temperatura (0 = bardzo zachowawczo)", 0.0, 1.0, 0.1, 0.05, key="k_temp")

    # ➕ NOWE: wybór trybu przetwarzania
    st.divider()
    mode = st.radio(
        "Tryb przetwarzania",
        options=[
            "Dodaj linki [[…]]",
            "Oczyść OCR/PDF (minimalnie)",
            "Oczyść OCR/PDF + dodaj linki"
        ],
        index=0,
        key="k_mode"
    )

    st.divider()
    st.markdown("**Model**"); st.code(MODEL)
    st.markdown("**Limit chunku**"); st.code(f"{MAX_CHARS_PER_CHUNK} znaków")
    st.markdown("**Overlap**"); st.code(f"{CHUNK_OVERLAP} znaków")
    st.divider()
    st.markdown("🔐 Klucz OpenAI pobierany z `st.secrets['OPENAI_API_KEY']`.")

st.markdown("### Wejście")
sample = st.toggle("Wstaw przykładowy fragment", value=False, key="k_sample_toggle")
default_text = ""
if sample:
    default_text = (
        "Pierwszym królem polskim goszczącym w Zamku Warszawskim był Władysław Jagiełło. "
        "W 1526 roku Zygmunt I przejął Mazowsze po śmierci książąt Janusza i Stanisława. "
        "W 1569 roku Unia lubelska wyznaczyła Warszawę i Zamek na stałe miejsce obrad sejmu."
    )

input_text = st.text_area(
    "Wklej tekst (.md/.txt, bez limitu długości – aplikacja pociągnie w częściach):",
    value=default_text, height=260, key="k_input_text"
)

uploaded = st.file_uploader("…lub wgraj plik .md / .txt", type=["md", "txt"], key="k_uploader")
if uploaded is not None:
    input_text = uploaded.read().decode("utf-8")

colA, colB = st.columns([1, 1])
with colA:
    run = st.button("🚀 Przetwórz", type="primary", key="k_run")
with colB:
    clear = st.button("🧹 Wyczyść pamięć encji tej sesji", key="k_clear")

if "known_entities_session" not in st.session_state or clear:
    st.session_state.known_entities_session = {}

if run:
    if not input_text.strip():
        st.warning("Wklej tekst lub wgraj plik.")
        st.stop()
    if not OPENAI_API_KEY:
        st.error("Brak OPENAI_API_KEY. Uzupełnij `.streamlit/secrets.toml`.")
        st.stop()

    # 1) wykryj sekcje
    sections = split_markdown_sections(input_text)

    # 2) przetwarzanie wg trybu
    if len(sections) >= 2:
        st.info(f"Znaleziono {len(sections)} sekcje – zastosuję podział notatek.")
        total_chunks = count_total_chunks_multi(sections)
        done_chunks = 0
        progress = st.progress(0, text="Start przetwarzania sekcji…")

        section_results = []
        global_map: Dict[str, List[str]] = {}

        for idx, sec in enumerate(sections, start=1):
            st.write(f"**Sekcja {idx}/{len(sections)}:** {sec['title']}")

            sec_content = sec["content"]

            # ➕ NOWE: tryby OCR
            if mode == "Oczyść OCR/PDF (minimalnie)":
                cleaned_text = apply_ocr_cleanup(sec_content, temperature=0.0)
                section_results.append({
                    "title": sec["title"],
                    "slug": slugify(sec["title"]),
                    "content": cleaned_text
                })
            elif mode == "Oczyść OCR/PDF + dodaj linki":
                cleaned_text = apply_ocr_cleanup(sec_content, temperature=0.0)
                linked_text, global_map = process_text_with_map(
                    cleaned_text, temperature=temp, initial_map=global_map
                )
                section_results.append({
                    "title": sec["title"],
                    "slug": slugify(sec["title"]),
                    "content": linked_text
                })
            else:  # "Dodaj linki [[…]]"
                linked_text, global_map = process_text_with_map(
                    sec_content, temperature=temp, initial_map=global_map
                )
                section_results.append({
                    "title": sec["title"],
                    "slug": slugify(sec["title"]),
                    "content": linked_text
                })

            done_chunks += count_total_chunks(sec["content"])
            pct = int((done_chunks / max(total_chunks, 1)) * 100)
            progress.progress(min(pct, 100), text=f"Postęp: {pct}% ({idx}/{len(sections)})")

        progress.progress(100, text="Wszystkie sekcje przetworzone ✅")

        # 3) „wszystko.md” = sklejone sekcje
        full_joined = "\n\n".join(s["content"] for s in section_results)

        # nazwa całości
        if uploaded is not None and getattr(uploaded, "name", ""):
            base = uploaded.name.rsplit(".", 1)[0]
            suggested_all = slugify(base)
        else:
            head_line = input_text.strip().splitlines()[0] if input_text.strip() else "wszystko"
            head_line = head_line.lstrip("# ").strip()
            suggested_all = slugify(" ".join(head_line.split()[:6]) or "wszystko")

        st.success("Gotowe! Poniżej pobieranie plików.")

        st.markdown("### Wynik: *wszystko* (`.md`)")
        st.download_button(
            "⬇️ Pobierz *wszystko* jako Markdown (.md)",
            data=full_joined.encode("utf-8"),
            file_name=f"{suggested_all}-wszystko.md",
            mime="text/markdown",
            key="k_download_all"
        )
        st.text_area("Tekst wynikowy (wszystko)", value=full_joined, height=300, key="k_output_text_all")

        st.markdown("### Pobierz sekcje osobno")
        for i, s in enumerate(section_results, start=1):
            st.write(f"**{i}. {s['title']}**")
            st.download_button(
                f"⬇️ Pobierz „{s['title']}”.md",
                data=s["content"].encode("utf-8"),
                file_name=f"{s['slug']}.md",
                mime="text/markdown",
                key=f"dl_{s['slug']}"
            )
            with st.expander("Podgląd sekcji", expanded=False):
                st.text_area("", value=s["content"], height=220, key=f"k_output_text_{i}")

        st.markdown("### Pobierz ZIP z wszystkimi sekcjami")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for s in section_results:
                zf.writestr(f"{s['slug']}.md", s["content"])
            zf.writestr(f"{suggested_all}-wszystko.md", full_joined)
        zip_buffer.seek(0)
        st.download_button(
            "📦 Pobierz wszystkie notatki jako ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{suggested_all}-notatki.zip",
            mime="application/zip",
            key="k_download_zip"
        )

    else:
        # tylko jedna sekcja – standardowo
        work_text = input_text

        if mode == "Oczyść OCR/PDF (minimalnie)":
            result_text = apply_ocr_cleanup(work_text, temperature=0.0)
            new_map = {}
        elif mode == "Oczyść OCR/PDF + dodaj linki":
            cleaned = apply_ocr_cleanup(work_text, temperature=0.0)
            result_text, new_map = process_text(cleaned, temperature=temp)
        else:  # "Dodaj linki [[…]]"
            result_text, new_map = process_text(work_text, temperature=temp)

        # aktualizacja pamięci encji (jeżeli była mapa)
        for lemma, surfaces in new_map.items():
            if lemma not in st.session_state.known_entities_session:
                st.session_state.known_entities_session[lemma] = []
            for s in surfaces:
                if s not in st.session_state.known_entities_session[lemma]:
                    st.session_state.known_entities_session[lemma].append(s)

        # nazwa pliku
        if uploaded is not None and getattr(uploaded, "name", ""):
            base = uploaded.name.rsplit(".", 1)[0]
            suggested_name = slugify(base)
        else:
            head_line = input_text.strip().splitlines()[0] if input_text.strip() else "podlinkowany"
            head_line = head_line.lstrip("# ").strip()
            suggested_name = slugify(" ".join(head_line.split()[:6]))

        st.success("Gotowe! Poniżej wynik.")
        st.markdown("### Wynik (`.md`)")

        st.download_button(
            "⬇️ Pobierz jako Markdown (.md)",
            data=result_text.encode("utf-8"),
            file_name=f"{suggested_name}.md",
            mime="text/markdown",
            key="k_download_single"
        )
        st.text_area("Tekst wynikowy", value=result_text, height=320, key="k_output_text_single")

else:
    st.info("Ustaw parametry, wklej tekst i kliknij **Przetwórz**. "
            "Aplikacja doda linki i/lub minimalnie oczyści artefakty OCR/PDF.")
