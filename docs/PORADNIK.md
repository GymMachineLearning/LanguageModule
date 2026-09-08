# Poradnik: ocena błędów przysiadu przez LLM

Praktyczna instrukcja obsługi pipeline'u. Opis projektowy feature'u jest osobno,
w [GeminiErrorEvaluationDocumentation.md](GeminiErrorEvaluationDocumentation.md).

---

## 1. Co ten pipeline robi

Wysyła nagranie przysiadu do Gemini, dostaje z powrotem listę wykrytych błędów
technicznych z przedziałami czasowymi, i porównuje to z ręcznymi adnotacjami
z datasetu MLPSD.

Rozpoznawanych jest **6 klas błędów** (`video_llm_evaluation/constants.py`):

| klasa               | opis                           |
| ------------------- | ------------------------------ |
| `Squat-depth`     | zbyt płytki przysiad          |
| `Back-round`      | zaokrąglenie pleców          |
| `Taking-off-foot` | odrywanie stóp od podłoża   |
| `Knee-collapse`   | zapadanie kolan do środka     |
| `Dominant-hip`    | ruch zdominowany przez biodra  |
| `No-knee-outlet`  | kolana nie wychodzą do przodu |

### Trzy kroki, trzy narzędzia

```
1. cli run        nagrania ──► Gemini ──► predictions_json/ + labels_npy/
                  płatne, wymaga klucza API, wolne (~15-30 s na nagranie)

2. cli evaluate   labels_npy/ + MLPSD .pkl ──► metrics/
                  lokalne, darmowe, szybkie — powtarzalne do woli

3. notebook       metrics/ ──► tabele i wykresy
                  tylko czyta, nic nie przelicza
```

Ten podział jest celowy. `evaluate` nie dotyka API, więc jeśli zmienisz metrykę
albo poprawisz mapowanie ground truth, przeliczasz **wszystkie** stare runy bez
ponownego płacenia za zapytania. Notebook nie liczy nic samodzielnie, więc nie
może pokazać innych liczb niż raport CLI.

### Moduły

| katalog                                    | odpowiedzialność                                                  |
| ------------------------------------------ | ------------------------------------------------------------------- |
| `llm_api/gemini/`                        | rozmowa z Gemini: config, budowa promptu, klient, parser odpowiedzi |
| `video_llm_evaluation/`                  | CLI, wykrywanie nagrań, schematy, konwersja segmentów na klatki   |
| `video_llm_evaluation/evaluation/`       | metryki i zapis artefaktów                                         |
| `video_llm_evaluation/notebook_support/` | warstwa prezentacji dla notebooka                                   |
| `tests/`                                 | testy;**żaden nie wywołuje API**                            |

---

## 2. Przygotowanie

```bash
pip install -r requirements.txt
```

Klucz API w pliku `.env` w katalogu głównym repo:

```
GEMINI_API_KEY=twoj_klucz
```

> **`google-genai` jest zapinowany na `2.22.0` i to nie jest ostrożność.**
> Sterowanie próbkowaniem klatek (`Part.media_processing` razem z
> `Part.video_metadata.fps`) nie istnieje w starszych wersjach, a starsze SDK
> **po cichu pominęłoby oba parametry** — dostałbyś runy, które wyglądają na
> skonfigurowane, a próbkują z domyślną częstotliwością 1 fps.

---

## 3. Krok 1 — zapytania do LLM (`run`)

```bash
python -m video_llm_evaluation.cli run \
  --results-root results/llm_evaluation/squat__gemini-3.8-flash__fps2 \
  --model-name gemini-3.8-flash \
  --video-fps 2.0 \
  --max-request 5
```

### Najważniejsze flagi

| flaga                  | domyślnie                       | do czego                                                 |
| ---------------------- | -------------------------------- | -------------------------------------------------------- |
| `--split`            | **`test`**               | z którego splitu MLPSD wysyłać nagrania               |
| `--results-root`     | `results/llm_evaluation/squat` | gdzie lądują wyniki                                    |
| `--model-name`       | `gemini-3.1-pro-preview`       | model                                                    |
| `--video-fps`        | `2.0`                          | ile klatek na sekundę widzi model                       |
| `--prompt-yaml`      | `squat_v1.yaml`                | który prompt wysłać                                   |
| `--min-certainty`    | brak progu                       | od jakiej deklarowanej pewności wskazanie się liczy    |
| `--max-request`      | wszystkie                        | limit liczby nagrań                                     |
| `--media-processing` | `STATIC`                       | `STATIC` albo `AGENTIC`                              |
| `--thinking-level`   | `MEDIUM`                       | `MINIMAL` / `LOW` / `MEDIUM` / `HIGH` / `NONE` |
| `--media-resolution` | domyślne API                    | tokeny na klatkę                                        |
| `--skip-existing`    | wyłączone                      | pomija nagrania z gotową predykcją                     |
| `--preferred-folder` | brak                             | podfolder brany w pierwszej kolejności                  |
| `--seed`             | `42`                           | losowanie przy`--max-request`                          |
| `--dataset-path`     | ścieżka do MLPSD`.pkl`       | źródło informacji o splicie                           |

### `--split` — domyślnie tylko nagrania testowe

Split pochodzi z kolumny `dataset_split` w `.pkl` MLPSD. Rozkład dla biblioteki
451 nagrań na dysku:

| `--split`           | nagrań wysłanych |
| --------------------- | ------------------ |
| `test` (domyślnie) | **42**       |
| `val`               | 41                 |
| `train`             | 179                |
| `all`               | 451                |

189 nagrań nie ma odpowiednika w MLPSD — nie mają przypisanego splitu, więc
wypadają przy każdej wartości poza `all`. Log przy każdym runie podaje pełny
skład, żeby ta strata nie była cicha:

```
Discovered 451 videos; MLPSD composition: train:179 val:41 test:42 unmatched:189
Split filter 'test' keeps 42 of 451 videos; 409 are excluded and will not be sent to the model.
```

**Domyślne `test` oznacza, że `run` bez flagi pominie 91% biblioteki.** Tak jest
celowo — ewaluacja modelu na zbiorze treningowym nie jest ewaluacją — ale
pamiętaj o tym, jeśli kiedyś chcesz przepuścić wszystko: `--split all`.

Jeden katalog wyników może zawierać nagrania z różnych splitów; filtrowanie
odbywa się dopiero przy `evaluate` i w notebooku.

### `--prompt-yaml` i `--min-certainty` — prompt jako hiperparametr

Prompt jest zmienną eksperymentu, nie stałą, i wersje różnią się tym, czego
wymagają od modelu:

| plik              | co robi                                                                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `squat_v1.yaml` | baseline: jednozdaniowe definicje klas,`confidence` jako liczba 0–1                                                                                              |
| `squat_v2.yaml` | kryterium zgłoszenia i kryterium wykluczenia osobno dla każdej klasy,`present: false` jako domyślna odpowiedź, `certainty` jako `low`/`medium`/`high` |

`--min-certainty` to **próg decyzyjny**, czyli poziom `certainty`, od którego
wskazanie modelu liczymy jako predykcję pozytywną. Segment poniżej progu jest
odrzucany; jeśli próg opróżni całą klasę, klasa staje się nieobecna, nawet gdy
model zadeklarował `present: true`.

Próg działa wyłącznie na prompt v2 i nowsze. Predykcje v1 nie mają pola
`certainty` — mają liczbowe `confidence`, którego **nie da się progować**, bo
wszystkie zwrócone wartości mieszczą się w przedziale 0.65–0.74, a wskazania
trafione i fałszywe są w nim nieodróżnialne. Dlatego dowolny `--min-certainty`
odrzuca segmenty v1 w całości, zamiast po cichu je przepuszczać. To zachowanie
celowe: jeśli zobaczysz pusty wynik na runie v1, sprawdź najpierw tę flagę.

### Nazwa katalogu wyników ma znaczenie

Konwencja: `squat__<model>__fps<N>__<prompt_version>`. Wersja promptu należy do
nazwy z tego samego powodu co model i fps: run v2 zapisany pod nazwą runu v1
wymiesza się z baseline'em i zniszczy porównanie. Runy **muszą być rodzeństwem**,
nie mogą być zagnieżdżone — `evaluate` wywodzi ścieżkę do MLPSD ze struktury katalogów
względem `--results-root`, więc dodatkowy poziom zagnieżdżenia zepsuje mapowanie.

Nigdy nie mieszaj w jednym katalogu runów o różnym fps, modelu albo prompcie.
`evaluate` zagreguje je wszystkie razem bez ostrzeżenia. Warunki są zapisywane w
`config.yaml` (w tym `prompt_version` i `min_certainty`) i w każdym
`predictions_json`, ale rozdzielić je musisz sam.

### `--video-fps` — dlaczego 2.0, a nie domyślne 1.0

Domyślna wartość API to 1 klatka na sekundę. Przysiad trwa 2–3 sekundy, więc
model widziałby **2–3 klatki na całe powtórzenie**. Efekt był mierzalny: przy
1 fps **87% granic czasowych zwracanych przez model lądowało równo na pełnych
sekundach** — nie szacował czasu, tylko przepisywał sekundowe znaczniki.

Pomiar na tym samym zestawie 5 nagrań:

| konfiguracja                  | granic na pełnych sekundach |
| ----------------------------- | ---------------------------- |
| gemini-3.1-pro-preview, fps=1 | 74.6%                        |
| gemini-3.1-pro-preview, fps=2 | 34.2%                        |
| gemini-3.8-flash, fps=2       | 15.2%                        |

Koszt: 65 tokenów na klatkę plus ~32 tokeny na sekundę ścieżki audio, czyli
przejście z 1 na 2 fps mniej więcej podwaja tokeny wejścia. Przy tych nagraniach
to grosze.

### `--media-processing STATIC` — nie zmieniaj bez powodu

`AGENTIC` pozwala modelowi samodzielnie nawigować po nagraniu i wybierać
momenty. Brzmi atrakcyjnie, ale **`--video-fps` przestaje wtedy działać** —
model dobiera próbkowanie sam. Domyślna wartość API to „model-specific
processing", więc bez jawnego `STATIC` nie masz gwarancji, co robi model z
rodziny 3.x Flash. Dlatego jest przypięte, a log przy każdym runie to odnotowuje.

### Co powstaje

```
<results-root>/
├── config.yaml                     parametry runu — model, fps, thinking, seed
├── manifest.csv                    wybrane nagrania: ścieżka, fps, liczba klatek
├── logs/
│   ├── run.log                     pełny log
│   ├── cases.csv                   status i czas każdego nagrania
│   └── failures.csv                tylko błędy
├── metrics/run_summary.json        podsumowanie runu (nie metryki jakości)
└── <kategoria>/<video_id>/
    ├── raw_responses/*.raw.json    surowa odpowiedź modelu
    ├── predictions_json/*.json     zwalidowana predykcja + warunki runu
    ├── labels_npy/*.npy            macierz 6 x T — wejście dla evaluate
    └── labels_csv/*.csv            to samo, czytelne
```

Surowa odpowiedź jest zapisywana **przed** parsowaniem, więc nagranie, które
padło na błędnym JSON-ie, i tak zostawia dowód w `raw_responses/`.

---

## 4. Krok 2 — ewaluacja (`evaluate`)

```bash
python -m video_llm_evaluation.cli evaluate \
  --results-root results/llm_evaluation/squat__gemini-3.8-flash__fps2
```

Bez API, bez kosztu. Bierze `labels_npy/*.npy`, dopasowuje do nagrań w pliku
`.pkl` MLPSD i liczy metryki na trzech poziomach.

| flaga                      | domyślnie                       | do czego                          |
| -------------------------- | -------------------------------- | --------------------------------- |
| `--results-root`         | `results/llm_evaluation/squat` | który run oceniać               |
| `--dataset-path`         | ścieżka do MLPSD`.pkl`       | ground truth i splity             |
| `--split`                | **`test`**               | który split oceniać             |
| `--max-frame-difference` | `1`                            | tolerancja rozjazdu liczby klatek |

### `--split` przy ewaluacji, i dlaczego metryki się nie nadpisują

`evaluate --split test` liczy metryki **wyłącznie** na nagraniach testowych,
niezależnie od tego, co jeszcze leży w katalogu. Wyniki lądują w osobnym
podkatalogu, więc różne splity współistnieją:

```
metrics/                  ← --split all
├── summary.json
├── frame_metrics.csv
└── split_test/           ← --split test
    ├── summary.json
    └── frame_metrics.csv
```

Dzięki temu `evaluate --split all` nie zniszczy wyników testowych policzonych
wcześniej. Nagrania spoza splitu trafiają do `evaluation_cases.csv` ze statusem
`skipped_split` i kolumną `dataset_split` — widać, co pominięto i dlaczego.

### Trzy poziomy metryk

- **frame-level** — porównanie klatka po klatce, macierz `6 x T`. Najsurowsze.
- **video-level** — czy klasa w ogóle wystąpiła w nagraniu. Ignoruje czas.
- **segment-level** — czy przedziały się pokrywają, przy tolerancji 0.1 / 0.5 / 1.0 s.

**Tolerancja to luz na obu końcach segmentu naraz.** Predykcja liczy się jako
trafienie tylko wtedy, gdy jej początek **i** koniec mieszczą się w ±tolerancji
od granic segmentu z ground truth. Dopasowanie jest 1:1, zachłannie po
malejącym IoU. Przy 60 fps `0.1 s` to około 6 klatek, `1.0 s` to 60 klatek.

### Pułapka: MLPSD ma 10 wierszy etykiet, nie 6

Tylko sześć z nich odpowiada klasom tego pipeline'u i **nie są to pierwsze
sześć** — właściwe indeksy to `(0, 1, 2, 3, 5, 7)`, stała
`MLPSD_ERROR_ROW_INDICES` w `constants.py`. Wzięcie wierszy 0–5 nie wywala się,
tylko po cichu podmienia dwie ostatnie klasy. Każdy nowy kod czytający MLPSD
`labels` musi używać tej stałej.

### Co powstaje w `metrics/`

| plik                                            | zawartość                                           |
| ----------------------------------------------- | ----------------------------------------------------- |
| `summary.json`                                | agregaty całego runu                                 |
| `frame_metrics.csv`                           | per klasa, agregat                                    |
| `frame_metrics_by_video.csv`                  | per klasa i nagranie                                  |
| `video_level_metrics.csv` / `_by_video.csv` | to samo, poziom nagrania                              |
| `segment_metrics.csv` / `_by_video.csv`     | to samo, poziom segmentu × tolerancja                |
| `evaluation_cases.csv`                        | które nagrania ocenione, które odrzucone i dlaczego |

**Zawsze zerknij na `evaluation_cases.csv`.** Nagrania bez dopasowania w MLPSD są
pomijane po cichu, a metryki liczą się na tym, co zostało.

---

## 5. Krok 3 — notebook

`video_llm_evaluation/squat_results_inspector.ipynb`

Uruchom **Restart Kernel → Run All**. Sekcja metryk jest na końcu.

### Wybór danych

Komórka wypisuje dostępne runy, potem wybierasz dwiema zmiennymi:

```python
SELECTED_MODEL = 'gemini-3.8-flash'
SELECTED_FPS = 2.0
SELECTED_SPLIT = 'test'
```

`SELECTED_MODEL` i `SELECTED_FPS` wybierają **run**; `SELECTED_SPLIT` wybiera
**widok** na niego: z którego katalogu metryk czytać i do czego ograniczyć listę
nagrań do przeglądania. Metryki nigdy nie są filtrowane w notebooku — robi to
`evaluate`, żeby notebook nie mógł pokazać liczby, której CLI nie policzyło.

Tabela runów ma kolumnę `content` z faktycznym składem katalogu
(`train:8 val:4 test:3 unmatched:10`) oraz `metrics_for` z listą splitów, dla
których metryki już istnieją. `content` liczony jest z zawartości, nie z configu
— to jedyny sposób, by zobaczyć skład katalogów zapełnionych przed dodaniem flagi.

Kolumna `video_fps_source`:

- `recorded` — wartość zapisana przez pipeline,
- `inferred` — run powstał przed flagą `--video-fps`, więc pokazane 1.0 to
  domyślna wartość API, a nie liczba faktycznie zanotowana.

Jeśli filtry pasują do zera albo więcej niż jednego runu, dostaniesz błąd z listą
kandydatów — notebook nigdy nie podepnie się po cichu pod „pierwszy lepszy" run.

### Tryb raportu

```python
PER_VIDEO = True   # rozbicie na poszczególne nagrania
PER_VIDEO = False  # czyste statystyki per klasa błędu
```

Rozkład przyczyn odrzuceń zostaje w obu trybach — to informacja o pokryciu
metryk, nie o pojedynczych nagraniach.

### Co pokazuje raport

Nagłówek runu → tabela per klasa → segment-level wobec tolerancji → wykresy →
karta każdego z 6 błędów → macierz nagranie × klasa → nagrania odrzucone →
agregaty runu. Wykresy zapisują się dodatkowo jako PNG w `metrics/figures/`.

Osią raportu jest **klasa błędu**, nie średnia z modelu. Agregaty są na końcu
jako kontekst zamykający.

Pojedyncza klasa z bliska:

```python
FOCUS_ERROR = 'Squat-depth'
show_class_card(report, FOCUS_ERROR, per_video=PER_VIDEO)
```

---

## 6. Typowe problemy

**`ImportError: cannot import name ... from notebook_support`**
Kernel trzyma starą wersję pakietu. Restart Kernel. Komórka konfiguracyjna ma
`%autoreload 2`, ale autoreload nie wykrywa **nowych plików** w pakiecie.

**`No metrics directory under ...` / `holds no evaluation output`**
Nie uruchomiłeś `evaluate` na tym runie. Komunikat zawiera gotową komendę.

**Ostrzeżenie „metryki są nieaktualne"**
Predykcje są nowsze niż `metrics/summary.json`. Uruchom `evaluate` ponownie.

**`Model 'x' is not available on this API key`**
Preflight odrzucił nazwę modelu **przed** uploadem czegokolwiek. Komunikat
podpowiada zbliżone nazwy.

**`JSONDecodeError` przy odpowiedzi modelu**
Zobacz `raw_responses/<video_id>.raw.json` — jest zapisany także przy błędzie.
Znany przypadek: prompt zawierał podwojone klamry `{{`, które `gemini-3.8-flash`
kopiował dosłownie. Naprawione; chroni przed tym test regresyjny.

**`no MLPSD recording matches ...` w `evaluation_cases.csv`**
Nagranie nie zostało dopasowane do ground truth i wypadło z metryk. Mapowanie
opiera się na ścieżce względem `--results-root`, więc winna bywa struktura
katalogów albo `RESULTS_TO_MLPSD_FOLDER` w `cli.py`.

---

## 7. Testy

```bash
python3 -m unittest discover -s tests
```

Żaden test nie wywołuje API — klient Gemini jest podmieniany atrapą. Testy
sprawdzają między innymi, że `fps`, `STATIC`, `media_resolution` i
`thinking_level` **faktycznie trafiają do requestu**: parametr, który nigdzie
nie dociera, wygląda identycznie jak parametr, który dociera i nie działa.

Przy realnych runach testowych trzymaj się `--max-request 5`.
