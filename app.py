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
    pro
