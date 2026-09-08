# Kontekst domenowy

Glosariusz projektu. Wyłącznie znaczenia terminów — bez decyzji implementacyjnych
(te trafiają do `docs/adr/`) i bez opisu struktury kodu.

## Klasa błędu

Jeden rodzaj błędu technicznego w przysiadzie ze sztangą na plecach. Zbiór klas
został wybrany pod trzy wymagania:

- **popularność u początkujących** — grupą docelową są osoby uczące się ruchu, nie
  zawodnicy;
- **atomowość** — klasa nie może być rozkładalna na złożenie innych klas. Dlatego
  „przysiad na dzień dobry" (*good morning squat*) **nie jest** klasą: jest tożsamy
  ze złożeniem [No-knee-outlet](#no-knee-outlet) i [Squat-depth](#squat-depth);
- **rozpoznawalność z estymowanej sylwetki** — błąd musi dać się wskazać gołym okiem
  na samej sylwetce, nie na obrazie. To wymaganie odrzuca np. *Butt-wink*, który
  widać na nagraniu, ale nie na sylwetce.

Pierwotnie zebrano **10 klas**; do modeli weszło **6**. Cztery odpadły albo z braku
nagrań (*Back-warped*, *Bad-head-movement*, *Butt-wink*), albo z niespełnienia
powyższych wymagań (*Butt-wink*, *Lack-of-abdominal-tension*). Dwie z odrzuconych
klas nie mają w zbiorze ani jednej pozytywnej klatki.

Kolejność sześciu klas jest stała w całym projekcie i nie wolno jej zmieniać: pozycja
klasy jest jej tożsamością w macierzy etykiet.

### Squat-depth

Zbyt płytki przysiad — osoba nie schodzi wystarczająco nisko, by uzyskać pełny zakres
ruchu.

Wskazanie: **biodra nie schodzą poniżej linii kolan**. Definicji poprawnej głębokości
jest kilka i różnią się między federacjami; ta jest przyjmowana najczęściej i służyła
jako drogowskaz przy anotacji, ale ostateczna ocena należała do anotatora. Kryterium
jest **celowo nieostre** — przy nauce początkujących nie jest potrzebna dokładność
sędziowska.

### Back-round

Zaokrąglenie kręgosłupa w trakcie ruchu, o krzywiźnie **wklęsłej**. Obejmuje zarówno
odcinek lędźwiowy (poważniejszy), jak i piersiowy — klasa odnosi się do obu, więc nie
istnieje jedno proste kryterium położenia. Może wystąpić w fazie schodzenia i w fazie
wstawania.

Nie należy mylić z **Back-warped** — plecami wypukłymi, czyli przeprostem. To osobny
błąd o innych przyczynach; był zbierany do zbioru danych, ale nie wszedł do modeli z
powodu małej liczby nagrań.

Przyczyny (brak ciśnienia w przeponie, brak mobilności bioder lub stawu skokowego i
inne) są z nagrania trudne do rozstrzygnięcia nawet dla trenera. Rozróżnienie
**wykrycia** błędu od wskazania jego **przyczyny** jest istotne dla modułu językowego.

### Dominant-hip

Ruch zdominowany przez biodra zamiast równomiernego zaangażowania całego ciała.
Osoba wypycha biodra do tyłu zbyt mocno, zamiast schodzić w dół z kontrolą, i unika
pełnego zgięcia kolan — ruch zaczyna przypominać martwy ciąg.

Wskazania: **tor sztangi wykrzywia się** (w poprawnym wykonaniu jest linią prostą i
pionową) oraz **kolana pozostają mniej więcej w tej samej pozycji** między dołem
przysiadu a wstawaniem, mimo że biodro się przemieszcza. Prostego zestawu kryteriów
nie ma — opisy są drogowskazami, ocena należy do anotatora.

### No-knee-outlet

Kolana nie przesuwają się wystarczająco do przodu w trakcie ruchu: pozostają zbyt
blisko ciała lub nawet nieznacznie schodzą za stopy. Ruch zostaje ograniczony do
minimalnego zgięcia kolan, a pracę przejmują biodra i pośladki.

Wskazanie: **zakres ruchu kolan do przodu**; w wykonaniu błędnym pozycja kolan jest
w kolejnych momentach powtórzenia stała, w poprawnym kolana wychodzą do przodu i
dostosowują się do ruchu biodra i pleców.

Ta klasa **łatwo generuje inne błędy** — współwystępuje ze [Squat-depth](#squat-depth)
albo z [Back-round](#back-round). W zbiorze 26 z 29 nagrań z tą klasą ma również
Squat-depth. Współwystępowanie nie znosi jednak atomowości: to trzy niezależnie
wskazywane klasy, a nie jedna.

### Knee-collapse

Kolana zbliżają się do siebie, czyli zapadają do środka, w trakcie ruchu.

Heurystyka: **kolana przybliżają się do siebie wraz z ruchem**. U profesjonalistów
zapadanie kolan w fazie wstawania nie jest uznawane za błąd — ale ze względu na grupę
docelową w tym projekcie **zapadanie się kolan zawsze jest błędem**.

### Taking-off-foot

Obie stopy tracą stabilność lub kontakt z podłożem: unoszą się pięty albo całe stopy.
Występuje szczególnie w fazie wstawania, gdy ciężar ciała przesuwa się do przodu.

Jedyna klasa o wskazaniu **binarnym i wprost obserwowalnym** — kontakt z podłożem albo
jest, albo go nie ma. Pozostałe pięć klas są kwestią stopnia.

## Nagranie

Jedno wideo z jednym lub wieloma powtórzeniami przysiadu, wykonanymi przez jedną
osobę. Podstawowa jednostka zbioru danych, podziału i oceny.

## Anotacja

Ludzki opis błędów w nagraniu. Dla żadnej klasy poza [Taking-off-foot](#taking-off-foot)
nie istnieje mechaniczne kryterium; anotacja jest **oceną anotatora**, opartą na
wskazaniach opisanych wyżej. To jest własność domeny, nie luka w danych: „prawda" tu
jest ludzkim sądem i granica między klasą obecną a nieobecną jest nieostra.

## Podział

Przydział nagrania do `train`, `val` albo `test`. Należy do zbioru danych, nie do
pipeline'u — nagranie ma swój podział niezależnie od tego, który eksperyment je
dotyka.

## Pewność

Termin przeciążony; w tym projekcie rozdzielamy trzy znaczenia i **nie używamy słowa
„pewność" bez wskazania, o które chodzi**:

- **certainty** — jak przekonany o swoim wskazaniu jest model; jego własna deklaracja,
  część odpowiedzi;
- **próg decyzyjny** — wartość certainty, od której wskazanie modelu traktujemy jako
  predykcję pozytywną; nasza decyzja, nie modelu;
- **precision** — jaki odsetek predykcji pozytywnych okazał się zgodny z anotacją;
  wynik pomiaru, dostępny dopiero po ewaluacji.

Zdanie „wymagam większej pewności" znaczy: podnieś **precision**, przy czym dostępne
narzędzia to certainty i próg decyzyjny.
