---
status: accepted
---

# Video-level precision per klasa jako główny cel optymalizacji

Pipeline mierzy trzy poziomy — video-level (czy nagranie zawiera błąd X), frame-level
(czy ta klatka zawiera błąd X) i segment-level z tolerancją czasową — a dokumentacja
feature'u zostawiała pytanie „która metryka jest najważniejsza" otwarte. Przyjmujemy
**video-level precision, raportowaną osobno dla każdej klasy błędu**, jako cel główny,
a frame-level IoU jako metrykę drugorzędną. Powód: przy próbkowaniu 2 FPS granice
czasowe zwracane przez model są zdegenerowane (87% ląduje na całych sekundach), więc
segment-level mierzy przede wszystkim szum czasowy, a nie to, czy model rozpoznał
błąd; agregaty (macro/micro F1) natomiast uśredniają klasy o skrajnie różnym
zachowaniu i ukrywają, gdzie naprawdę siedzi problem.

Docelowy punkt pracy: **precision ≥ 0.55 przy podłodze recall ≥ 0.40**, z F0.5 jako
jedną liczbą do porównywania wersji promptu. Podłoga recall jest częścią celu, nie
dodatkiem — bez niej „większa pewność" ma trywialne rozwiązanie w postaci modelu,
który nie zgłasza niczego.

## Consequences

Baseline `gemini-3.8-flash`, fps=2, prompt v1, split `test` (42 nagrania): video-level
micro precision 0.25, recall 0.61. Rachunek na tych danych pokazuje sufit dla zmian
samego promptu: nawet gdyby dwie najgorsze klasy stały się idealnie precyzyjne przy
zachowanym recall, micro precision wyszłaby **0.486** — poniżej celu. Osiągnięcie 0.55
prawdopodobnie wymaga instrumentu poza promptem (dwuprzebiegowa weryfikacja kandydatów
albo próg decyzyjny na certainty), i cel został ustawiony ze świadomością tego kosztu.
