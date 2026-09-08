---
status: accepted
---

# Dyskretna skala `certainty` zamiast liczbowego `confidence`, próg poza promptem

Prompt v1 prosi model o `confidence` jako liczbę 0–1. Zmierzone na 27 wskazaniach
wartości mieszczą się w całości w przedziale **0.65–0.74**, z medianą 0.72 dla
wskazań trafionych i 0.70 dla fałszywych; `global_confidence` zajmuje 0.68–0.70. Pole
jest zdegenerowane — każdy próg odcina trafienia i pomyłki w tej samej proporcji —
więc w v2 zastępujemy je **dyskretną skalą `low | medium | high` z jawną definicją
każdego poziomu**. Model nie potrafi produkować kalibrowanych liczb, ale potrafi
wybierać z etykietowanej skali, gdy każda etykieta ma opisany warunek.

Sam **próg decyzyjny należy do pipeline'u, nie do promptu**: prompt zwraca certainty,
a to, od którego poziomu wskazanie liczy się jako predykcja pozytywna, jest parametrem
uruchomienia. Inaczej każde przesunięcie progu kosztowałoby ponowny przebieg przez
API po 42 nagraniach.

## Consequences

`ErrorSegment` przyjmuje oba pola — `certainty` (v2 i dalej) oraz `confidence` (v1) —
bo predykcje v1 muszą pozostać czytelne i przeliczalne, żeby baseline dalej istniał.
Progowanie działa na etapie zapisu etykiet, więc przemiatanie progu po fakcie na
zapisanych `predictions_json` wymaga jeszcze ponownego wyprowadzenia macierzy etykiet
w `evaluate` — to nie jest jeszcze zrobione i dopóki nie będzie, zmiana progu oznacza
ponowne uruchomienie `run` (bez kosztu API tylko wtedy, gdy predykcje są w cache).
