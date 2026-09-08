# video_llm_evaluation

Pipeline oceny błędów technicznych przysiadu przez model językowy: wysyła
nagranie do Gemini, konwertuje zwrócone przedziały czasowe na etykiety klatkowe
`6 x T` i porównuje je z ground truth z datasetu MLPSD.

**Instrukcja obsługi: [../docs/PORADNIK.md](../docs/PORADNIK.md)** — trzy kroki
(`run` → `evaluate` → notebook), flagi CLI, struktura wyników i typowe problemy.

## Struktura

- `cli.py` — komendy `run` (zapytania do Gemini) i `evaluate` (metryki)
- `constants.py` — 6 klas błędów, domyślne parametry, mapowanie wierszy MLPSD
- `schemas.py` — schematy pydantic dla manifestu i odpowiedzi modelu
- `discovery.py` — wykrywanie nagrań i odczyt metadanych przez OpenCV
- `evaluation/` — metryki frame/video/segment, konwersja segmentów na klatki, zapis artefaktów
- `notebook_support/` — warstwa prezentacji dla `squat_results_inspector.ipynb`
- `squat_results_inspector.ipynb` — przegląd nagrań, predykcji i metryk

Rozmowa z modelem jest w osobnym pakiecie `llm_api/gemini/` (config, budowa
promptu, klient, parser odpowiedzi).
