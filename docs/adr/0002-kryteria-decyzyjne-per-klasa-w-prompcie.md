---
status: accepted
---

# Kryteria decyzyjne per klasa błędu, zamiast wspólnej instrukcji ostrożności

Prompt v1 opisuje sześć klas jednym zdaniem każdą i nakłada na wszystkie te same
zasady. Wyniki na splicie `test` (42 nagrania) pokazują, że problem precision **nie
jest jednolity**: `Dominant-hip` (video P=0.18, R=1.00) i `No-knee-outlet` (P=0.17,
R=1.00) odpowiadają za 65% fałszywych wskazań video-level i 73% fałszywych klatek,
podczas gdy `Taking-off-foot` ma precision **1.00**. Dlatego prompt v2 dostaje
**osobne kryteria wykrycia i wykluczenia dla każdej klasy**, a nie wspólną instrukcję
„bądź ostrożniejszy" — taka instrukcja nie miałaby czego poprawić w klasie, która już
jest idealnie precyzyjna, a miałaby co zepsuć w jej recallu (0.40).

## Considered Options

Rozważaliśmy prompt class-agnostic, żeby uniknąć zarzutu dopasowania do zbioru
testowego. Odrzucony, bo różnicowanie ma **uzasadnienie merytoryczne niezależne od
wyników**: `Taking-off-foot` jest jedyną klasą o wskazaniu binarnym (kontakt z
podłożem albo jest, albo nie), a dwie klasy o najgorszej precision mają w v1
definicje sformułowane jako niedookreślony stopień — „kolana nie wychodzą
**wystarczająco**", „biodra wychodzą **za bardzo** do tyłu". Model nie ma tam progu,
przy którym mógłby powiedzieć „nie", i recall równy dokładnie 1.00 na obu tych klasach
jest tego objawem. To defekt definicji, nie kwestia prevalencji w splicie.

Zarzut dopasowania do `test` pozostaje realny dla *kalibracji* progów. Dlatego
zaostrzanie kryteriów opiera się na definicjach klas z `CONTEXT.md` (materiał
źródłowy pracy, nie obserwacje ze splitu), a wszelki tuning liczbowy prowadzimy na
`val` (51 nagrań), zostawiając `test` nietknięty.

## Consequences

Każda z sześciu klas dostaje w v2 własne kryterium zgłoszenia i własne kryterium
wykluczenia, wyprowadzone z definicji w `CONTEXT.md`. Rozkład błędów zmienia jednak
to, **jak** te kryteria są napisane, a nie tylko to, że istnieją:

- `Dominant-hip` wymaga **koniunkcji dwóch obserwacji** (odchylony tor sztangi ORAZ
  kolana nieruchome, gdy biodro się przemieszcza). Spełnienie jednej nie wystarcza.
  Wykluczenie wskazuje wprost przypadek pomyłkowy: pochylony tułów przy
  skoordynowanym ruchu kolan i bioder.
- `No-knee-outlet` dostaje wykluczenie celujące w splątanie klas: płytki przysiad z
  proporcjonalnym wyjściem kolan **nie** jest tym błędem, a klasy nie wolno zgłaszać
  jako konsekwencji zgłoszenia `Squat-depth` czy `Back-round`. To reakcja na 26 z 29
  nagrań, w których obie klasy współwystępują w anotacji.
- `Taking-off-foot` zachowuje kryterium **binarne i celowo nieobwarowane** — jego
  wykluczenie odsiewa tylko przeniesienie ciężaru na przodostopie przy zachowanym
  kontakcie pięty. Precision 1.00 nie ma tam czego zyskać, a recall 0.40 ma co
  stracić.
- `Knee-collapse` i `Back-round` dostają wykluczenia odsiewające najbliższy stan
  pomyłkowy: odpowiednio wąską pozycję startową bez zmiany odległości kolan oraz
  pochylenie tułowia przy neutralnym kręgosłupie.

**Wynik v2 na splicie `test` (42 nagrania), video-level micro:** precision 0.25 → 0.50,
recall 0.607 → 0.321, F0.5 0.283 → 0.450. Najmocniejszy dowód dla koniunkcji dwóch
obserwacji: `Dominant-hip` przeszedł z precision 0.18 na 0.667 **przy nietkniętym
recallu 1.00**, a segment-level F1@1s wyniósł 0.867. `Taking-off-foot` nie drgnął
(P=1.00, R=0.40), zgodnie z zamiarem.

Dwa kryteria wykluczenia przestrzeliły. `No-knee-outlet` spadł z recall 1.00 na **0.00**
— zakaz zgłaszania klasy jako konsekwencji `Squat-depth` wyciął razem z fałszywymi
wskazaniami także trzy prawdziwe, bo w GT te klasy współwystępują w 26 z 29 nagrań.
`Squat-depth` spadł z recall 0.50 na 0.125: sformułowanie „kryterium ma być tolerancyjne"
w połączeniu z domyślnym `present: false` dało klasę, która prawie nic nie zgłasza.
Obie sytuacje wymagają poluzowania, nie zacieśnienia.

**v3 zmienia wyłącznie `Knee-collapse`** — jedyną klasę, która po v2 pogorszyła się na
obu osiach (precision 0.40 → 0.286, recall 0.80 → 0.40) i która trzyma 5 z 9 pozostałych
fałszywych wskazań video-level oraz 14 z segment-level. Dostaje ten sam instrument, który
zadziałał na `Dominant-hip`: koniunkcję dwóch obserwacji — malejąca odległość między
kolanami **oraz** przejście kolana przyśrodkowo względem linii palców własnej stopy.
`Dominant-hip` zostaje nietknięty celowo: przy 2 pozostałych FP i recallu 1.00 zacieśnianie
go stawia na szali jedyną działającą klasę dla zysku dwóch nagrań.

Poza zakresem v2 i v3 zostaje świadomie **oprawa promptu**: wiodące
`expected_error_classes` w `user_prompt` naprawiamy w v2 tylko przez zmianę nazwy pola
na neutralne (bo pozostawienie go jest wprost sprzeczne z regułą abstynencji), ale
przykład odpowiedzi, który kotwiczy model na dwóch obecnych klasach z sześciu, zostaje
w v2 nietknięty i trafia do v3. Dzięki temu efekt zmiany definicji jest przypisywalny
osobno od efektu zmiany kotwiczenia — dokumentacja feature'u traktuje `prompt_version`
jak hiperparametr eksperymentu.

`Back-round` ma precision 0.00 i frame recall 0.00 przy 312 pozytywnych klatkach:
kryterium wykluczenia jej nie pomoże, bo ta klasa nie cierpi na nadwykrywanie, a na
niewykrywanie. Wymaga osobnej diagnozy poza tym ADR-em.
