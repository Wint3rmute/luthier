# Master's thesis

## Building

In the `latex/` folder:
```bash
bibtex Dyplom && pdflatex -interaction=nonstopmode Dyplom.tex
```


## Finalne poprawki

- usunąć obramowania przy łączach
- odwołując się do rysunków w tekście podać rys. numer, albo rysunek numer

<!-- 3 zbędne: "W przeciwieństwie do technik wykorzystujących sieci neuronowe jako narzędzia syntezy" -->
4 mamy dwa podrozdziały "problem optymalizacji" i "problem optymalizacyjny", to trzeba jakoś zmienić
4 dobór funkcji celu to rozdział, który sprawia największe problemy, ale o tym będzie dalej
5 badania i analiza wyników w jednym rozdziale "Badania symulacyjne"

9 jeśli nie ma tabel, to spis tabel należy usunąć; to rodzi pytanie, dlaczego nie ma tabel? obok wykresów można dodać tabele, nawet jeśli reprezentują te same dane (wykresy są wtedy poglądowe, tabele bardziej szczegółowe)

Zastanawiam się, czy warto wprowadzać punkt "Skróty" (czy takie są teraz wymagania w szablonie?). Czytając pracę brakowało mi jednak wyjaśnień bezpośrednio w tekście, przy pojawianiu się oznaczeń. Może warto w tekście podawać chociaż nazwy angielskie (np. DAW w tekście pojawia się najpierw bez angielskiej wersji, dopiero dalej jest po angielsku - to nie ma sensu).

<!-- DAW jest wiele, skróty sugerują, że jedno. -->

# Rozdział 1

<!-- 12 zbędne "za pomocą sieci neuronowych" przy punkcie 3, chyba, że to faktycznie jedyne możliwe podejście i Pan to na pewno wie -->
<!-- 12 proszę unikać kolokwialnych określeń, nawet jeśli są w ""; "nawigują" należy zmienić -->

<!-- 13 czy podejście typu stable diffusion jest jedynym podejściem do generacji plików audio? -->
- Doprecyzowałem opis kategorii, chodziło tutaj o algorytmy generujące cały plik audio na podstawie opisu, tutaj pierwsze prace pojawiły się dopiero w tym roku i one rzeczywiście wykorzystują Stable Diffusion wytrenowane na spektrogramach

<!-- 13 "promptem" -->
<!-- 13 po co uwypuklać "dźwięku"? pozostałe uwypuklenia też wydają się być zbędne -->
- Uwypuklenia usunięte 

<!-- 13 opisane zostały "pierwsza grupa algorytmów", "druga grupa algorytmów", logicznie, powinien się pojawić akapit "trzecia grupa algorytmów" -->
- Akapit dopisany

<!-- 13 "Wynik pracy algorytmu implementowanego w ramach pracy" -->
- Zmienione na "Graf przetwarzania sygnałów wytworzony przez algorytm implementowany w ramach pracy magisterskiej jest"

<!-- 14 cel pracy mi się nie podoba; celem miało być opracowanie algorytmu generacji grafów przetwarzania sygnałów (tak jak w tytule) i proponuję tego się trzymać; faktycznie to właśnie cel pracy można uwypuklić boldem -->
- Poprawiłem, już się zgadza z tytułem pracy

14 cały ten fragment "Problem poruszany w pracy można zakwalifikować do grupy zagadnień związanych z pojęciami
computer-aided design oraz generative artificial intelligence, zastosowanymi w dziedzienie inży-
nierii dźwięku. Docelowo zaimplementowany algorytm będzie automatyzował pracę inżyniera
dźwięku, tworząc i konfigurując grafy przetwarzania sygnałów dźwiękowych, dostępne w pro-
gramach typu digital audio workstation (1.4)."
jest tutaj zupełnie zbędny; ma być prosto: cel pracy to XXX, a zadania szczegółowe to YYY; zadań szczegółowych jest za mało, np. robione są też badania
- Rozpisałem zadania szczegółowe. Nie do końca zgadzam się ze stwierdzeniem że ten akapit jest zbędny, chciałbym podać potencjalne zastosowania takiego algorytmu, żeby to nie było "badanie żeby odbębnić magisterkę"

15 należy napisać, że dobór funkcji celu nie jest problemem trywialnym, żeby nie niepokoić czytelnika zaznajomionego z optymalizacją (ale o tym więcej dalej)

<!-- 15 Pana postulat jest taki, że wygenerowany graf można wykorzystać jako instrument muzyczny; pewnie jakoś i można, ale co my o tym instrumencie wiemy, poza tym, że poprawnie generuje dokładnie jeden dźwięk (próbkę); może nie warto o tym wspominać, bo jednak cały instrument nie jest w jakiś sposób optymalizowany; nie ma wielu próbek -->
- Zmieniłem stwierdzenie "gotowy instrument" na "przepis na instrument", bo nie zdążyłem w ramach pracy dopisać modułu, który
pozwalałby uruchomić graf tak, żeby działał w czasie rzeczywistym jak instrument. Niemniej jednak parametry syntezy i struktura
grafu są przepisem na instrument i można je wykorzystać dalej.

15 "Struktura grafu powinna być możliwie jak najbardziej przejrzysta dla użytkownika" to nic nie znaczy; przejrzysta, czyli jaka? czy to będzie w funkcji celu? nie będzie, więc po co o tym

16 "algorytm maksymalizuje wykorzystanie poszczególnych bloków przetwarzania w grafie" co to znaczy?
16 czym jest krzyżowanie dwóch grafów przetwarzania sygnału? to nie zostało wcześniej nigdzie opisane i w dalszej części pracy też nie, czyli nie zostało zrobione

<!-- 16 w 1.3 proszę sprowadzić to do ciągłego tekstu, bez odwołań w nawiasach -->
- Poprawione

<!-- 17 "Wyniki padań" -->
- Poprawione

Ogólnie, rozdział trzeba przerobić.

Jeśli na początku ma być ogólnie o problematyce syntezy dźwięku, to ten fragment tekstu należy rozbudować. Napierw skrótowo należy przedstawić PROBLEMY a dopiero potem wymienić kilka metod/algorytmów rozwiązania. Proponuję skorzystać z pozycji
"A Comprehensive Survey on Deep Music Generation: Multi-level Representations, Algorithms, Evaluations, and Future Directions" SHULEI JI, JING LUO, and XINYU YANG
Konkretniej, w rozdziale 2 Related Work, autorzy powołują się na kilka prac przeglądowych. Może warto je odszukać i wykorzystać.

Jest tutaj kluczowe, żeby jasno napisać czym jest sam problem syntezy dźwięku. Nie jakie są do tego stosowane algorytmy, ale czym jest sam problem. O algorytmach można kawałek dalej.

Ostatni akapit to łagodne wprowadzenie w wykorzystanie grafów DSP, czyli do podpunktu 1.1.

Potem należy zrobić przegląd literaturowy dotyczący właściwego problemu, czyli generacji/zastosowania grafów DSP. Najpierw należy napisać czym jest graf DSP - to będzie kolejny podrozdział. Jeszcze niekoniecznie formalnie. Tutaj trzeba podać i opisać jakąś literaturę na temat grafów DSP. Dlaczego się z nich korzysta. Gdzie zostały skutecznie wykorzystane. A co najważniejsze: czego jeszcze nie zrobiono. Wszystko wspomagane odwołaniami do literatury.

Później podajemy cel pracy, to jest podpunkt 1.2. (Cel i zakres pracy) Celem pracy jest opracowanie algorytmu generacji grafu DSP do rozwiązywania problemu syntezy dźwięku (tak jak w tytule). Potem podajemy zadania szczegółowe. Wybór/opracowanie funkcji celu najpierw (bo taka jest kolejność). Opracowanie algorytmu później. A na końcu przeprowadzenie badań.

Potem akapit albo dwa (albo i trzy) o strukturze pracy. Tak, żeby względnie łatwo było powiązać zadania szczegółowe i rozdziały pracy.

W pracy są też Dodatki (może lepiej Załączniki) - o nich też należy wspomnieć.

# Rozdział 2

<!-- 18 zbędne: "Jak opisano w zakresie pracy (1.2), w pracy rozwiązywany jest problem budowy grafu przetwa- -->
<!-- rzania sygnałów oraz opracowywana jest funkcja celu, porównująca dwa sygnały pod względem -->
<!-- ich barwy. Następnie algorytm generujący graf przetwarzania sygnałów oraz funkcja porównu- -->
<!-- jąca barwy sygnałów są wykorzystane do rozwiązania problemu optymalizacyjnego." -->
<!-- nie ma co się powtarzać -->
- powtórzenia usunięte

<!-- 18 Rysunek 2.1 jest nieczytelny; powiększyć i dać na środek -->
- powiększony

18 czym jest "synteza subtraktywna"; wykorzystane są tu pojęcia, które nie zostały do końca opisane - np. czym są częstotliwość odcięcia i rezonans, jaka jest postać sygnału wejściowego albo wygenerowanego? Proponuję poświęcić trochę miejsca na opis pojedynczego wierzchołka grafu; napisać jaką postać będą miały wejścia i wyjścia; jakie mogą być zastosowane operacje matematyczne. W końcu w pracy dalej wykorzystywane są konkretne.

19 Rysunek 2.2 jest nieczytelny; w zasadzie wszystkie rysunki z grafami są nieczytelne i wymagają przeróbki (a może umieścić je w pracy poziomo?)

19 N to w końcu zbiór czy ciąg (tak jak jest zapisane, to ciąg)? Bo na pewno nie "liczba węzłów"
19 po co są dwa oznaczenia na pojedyncze wejście: p_1 i i_{l, m} ? ujednolicić
19 oznaczenia p_1, p_2, itd. są zduplikowane; nie mam tu na myśli tylko tego, że są w wejściach i wyjściach, ale też w tym, że duplikują się między węzłami
19 f_i(x) czym jest x? jak się ma to do wejść i wyjść
19 postać C nie jest do końca jasna, proszę ją sformalizować

19 "W pracy wykorzystano algorytm genetyczny differential evolution [40] do wygenerowania struk-
tury i parametrów grafu. Genotyp opisujący dany graf przetwarzania sygnałów składa się z
dwóch części:
1. fragment decydujący o strukturze grafu, S = [s1 , s2 , . . .],
2. fragment decydujący o wartości parametrów w wolnych wejściach, P = [p2 , p2 , . . .]."
to nie tutaj; najlepiej w algorytmie rozwiązania; zresztą ten opis genotypu jest zupełnie niejasny o czym powiem jeszcze w rodziale a algorytmem rozwiązania

19 "2.1.1. Struktura grafu" strukturę grafu już opisaliśmy więc nie ma sensu takie tytułowanie tego podpunktu
19 wzór (2.1) zupełnie nie ma sensu
19 dla każdego (l, m) z jakiego zakresu wartości; dodatkowo, (l, m) nie może należeć do C, bo C jest listą zbiorów; ponownie, opis genotypu to nie tutaj; algorytm genetyczny będzie dopiero częścią rozwiązania i w części z rozwiązaniem powinien zostać opisany

20 czy coś wiemy o postać x i \hat{x} ? dalej są wykorzystywane we wzorach (2.4) i dalej; trzeba to formalnie wprowadzić
20 czym jest 'a', czym jest 'N' (bo chyba nie liczbą węzłów w grafie DSP - konflikt oznaczeń)
20 jak obliczać X_r(k), czym jest i jak obliczać H_m(k) ?

21 te ograniczenia (2.8) też powinny wywędrować do algorytmu rozwiązania; nie może być mowy o jakichś ograniczeniach wynikających z implementacji na etapie formułowania problemu; żadnego float; natomiast powinny się tu pojawić ograniczenia na graf DSP - w końcu w rozdziale "cel pracy" pojawiły się wymagania co do tego grafu DSP, to teraz powinny pojawić się tutaj stosowne ograniczenia inaczej nie możemy dalej mówić o optymalizacji

21 należy odczepić zmienne decyzyjne od postaci algorytmu genetycznego; tutaj definiujemy zmienne decyzyjne w ogólności, dopiero potem skorzystamy z algorytmu genetycznego, który te zmienne decyzyjne będzie jakoś wewnętrznie reprezentować; w konsekwencji, nie może być mowy o S, P będących fragmentami jakiegoś genotypu.

Same oznaczenia S, P mogą oczywiście zostać, ale trzeba je jasno przedstawić, odnieść do struktury grafu, bo to strukturę grafu optymalizujemy.

# Rozdział 3

22 "Aby stopniowo dostosować graf przetwarzania sygnałów zaimplementowany w rozdziale" grafu nie będzie nikt implementować, implementowany będzie algorytm generacji grafu; odwołanie do rodziału nr 5 jest tu zbędne
<!-- 22 "Autoregressive, Waveform != Perception" trzymajmy się polskiego -->
- Usunięty angielski podpis

22 brak odwołania w tekście do rys 3.1 (jest do 3)
22 może warto te prace 17, 19, 12 (kolejność??) itd. opisać

<!-- 23 "dwóch cech:" -- kropka -->

23 czy RMS i DTW można jakoś wyjaśnić?

<!-- 27 "TODO: obrazek" -->
- usunięte, obrazek już dodany

Przyznam się, po dokładnym przeczytaniu tego rozdziału, że proces doboru funkcji celu jest niezrozumiały. Dlaczego? Ano dlatego, że spodziewałem się analizy własności funkcji celu pod kątem percepcji słuchacza; w końcu tak zaczyna się rozdział. Ostatecznie dostajemy coś zupełnie innego.

Dobór różnych funkcji celu prowadzi do różnych problemów optymalizacji. Trudno o jakieś podstawy do ich porównania.

Popatrzmy na punkt 3.3.1. Jest mowa o "właściwych parametrach" albo o "niedokładnym odwzorowaniu dynamiki" albo o "poprawnej wartości parametru". Z czego mamy te parametry odniesienia względem których można porównywać funkcje celu? Jak są wyznaczane? Mowa jest o karach? Jakie kary, z czego wynikają?

Inaczej: jakie kryterium zostało wykorzystane do porównania funkcji celu wyjściowego problemu generacji grafu (czy faktycznie - problemów, bo mamy różne funkcje celu).

Popatrzmy teraz na punkt 3.4. Jest mowa o "zweryfikowanie skuteczności każdej z funkcji w uproszczonym problemie optymalizacyjnym". W jakim problemie optymalizacyjnym, skoro mamy różne funkcje celu a w konsekwencji - różne problemy optymalizacyjne?

To co jest tutaj badane, to nie funkcje celu, tylko grupa algorytmów (i to heurystyk) generacji dźwięku. Tylko po co, skoro i tak całość jest porównywana przez sprowadzenie do spektrogramu?

Czemu nie wybrać spektrogramu jako bazy do stworzenia kryterium?

Moze jest jakiś powód. Za długo się liczy? Za trudno?

# Rozdział 4

29 Nie zaczynajmy od rysunku. Zacznijmy od opisu.

Najpierw trzeba opisać rysunek 4.1. Czy w zasadzie zacząć od opisu, a potem pokazać ten rysunek. Rysunek jest tylko wsparciem. Sam niewiele wyjaśnia.

Dalej trzeba opisać algorytm genetyczny. To co pojawiło się w sformułowaniu problemu, a dotyczyło algorytmu genetycznego należy przenieść tutaj.

Ale nawet gdyby to wszystko tu przenieść, to nadal jest zdecydowanie za mało. Proszę zamieścić schemat/pseudokod stosowanego algorytmu genetycznego, opisać wszystkie parametry - jak była robiona selekcja, krzyżowanie i mutacja. Dokładnie pokazać kodowanie (i dekodowanie).

Patrząc się na rysunek 4.1, w samym algorytmie generującym graf mamy trzy bloki, z czego - jeśli rozumiem - budowa/modyfikacja grafu, oraz optymalizacja parametrów bloków DSP w grafie to części składowe. Gdzie tu wykorzystujemy algorytm genetyczny, a gdzie nie? Jeśli wszędzie, to czemu jest rozbicie na dwa podproblemy?

Dalej jest mowa o wyborze źródeł sygnału. Czy korzystamy tu z algorytmu genetycznego? Bo moglibyśmy. Czy np. jest tak, że algorytm genetyczny tylko wybiera z jednego z predefiniowanych źródeł sygnału, czy też może źródła sygnału (które są przecież fragmentem grafu DSP) tworzyć od podstaw?

Punkt 4.1 - czy można coś o tych generatorach powiedzieć dokładniej? Dlaczego te zostały wybrane? Czym może tu manipulować algorytm genetyczny.

<!-- Punkt 4.2 - tu na pewno trzeba napisać zdecydowanie więcej; co to za filtr, jakie ma parametry, czym może manipulować algorytm genetyczny. -->
- dopisałem trochę więcej, ale tutaj opis z perspektywy "co filtr potrafi" jest wystarczający

Punkt 4.3 - analogicznie, jakie parametry mają te efekty, czym manipuluje algorytm genetyczny, co wchodzi w skład osobnika? Pojawia się też pytanie, dlaczego mamy strukturę chorus -> delay -> reverb. W końcu czy nie było zamiarem pracy strukturę generować dynamicznie?

Wracając do rysunku 4.1 - jest cała część algorytmu opisująca ocenę wygenerowanego dźwięku. Spodziewałem, się, że tu będzie stosowane wybrane wcześniej kryterium jakości MFCC. A nagle pojawia się też analiza spektralna i jeszcze zupełnie niewyjaśnione algorytmy symulujące odczucia psychoakustyczne (z których warto było skorzystać przy wyborze funkcji celu, a niekoniecznie tutaj).

Podsumowując, nie jest jasne jak ten algorytm działa. Nie jest jasne, jakie są jego elementy składowe. Na podstawie tego, co zostało przedstawione nikt nie będzie w stanie odtworzyć tego algorytmu (a powinien móc).

# Rozdział 5

- ten rozdział faktycznie jest o implementacji; tak go należy nazwać
- punkt 5.1 "Podstawy syntezy dźwięku w syntezatorach modułowych" to zupełnie nie tutaj. Tutaj już musimy mieć wszystkie podstawy i wstępy teoretyczne daleko za sobą. Tutaj jest już tylko na temat przygotowanego rozwiązania. Tego rodzaju informacje można przenieść do wstępu (do opisu grafu DSP), albo do sformułowania problemu.
35 "Pojedynczy węzeł DSP ..." to już było w sformułowaniu.
- nie wiem czy warto tutaj wspominać o "wymaganiach" - one wynikają z rozwiązywanego problemu; jeśli faktycznie jest to istotne, to może warto je zebrać na początku rozdziału
- natomiast jest za mało o samej implementacji; jakie moduły programu utworzono, jakie funkcje, itd. jak przekazywane są dane (może warto podać format)

# Rozdział 6
- praca wykorzystuje próbki z literatury; jakie próbki, co o nich wiemy? czemu te?
- czym jest "Pure Data"; skoro jest tak kluczowy, to dlaczego nie ma o nim słowa w przeglądzie literaturowym
- proszę wyjaśnić proces ustalania częstotliwości podstawowej, czemu ma on jakiekolwiek znaczenie? Może coś o samym algorytmie YIN.

Badania z ustalonymi wartościami parametrów: rozmiar populacji 50, liczba iteracji 200 są niewystarczające. To faktycznie nie są badania, a uruchomienia.

Wypadałoby przebadać jaki wpływ mają te parametry na przygotowany algorytm. Zresztą 200 iteracji to jest stanowczo za mało jak na algorytm genetyczny (chyba, że akurat tutaj wyszłoby inaczej, ale to musiałyby potwierdzić badania).

I robimy badania tylko na 3 próbkach (bo co jest w rozdziale 6.4 trudno powiedzieć)? To żadne testy. Zdecydowanie za mało.

Spodziewałbym się przynajmniej wykresów wartości funkcji celu w zależności od a. rozmiaru populacji, b. liczby iteracji. A inne parametry algorytmu genetycznego (np. elitaryzm)?

Co z czasem działania? To też kluczowe kryterium, które możnaby przebadać.

Czy można wynik z czymś porównać? Np. z innymi algorytmami? Czemu nie porównać bezpośrednio z pracą [32]?

# Rozdział 7
- tak jak wspominałem, to co dotyczy badań idzie do rozdziału z badaniami.
- "algorytm wytwarza interesujące barwy" nieścisłe, zbędne
- "Testy na małej próbie słuchaczy pozwalają stwierdzić, że nie są oni w stanie odróżnić sygnałów wygenerowanych dla problemów z rozdziału 3 od dźwięków docelowych." o żadnych takich testach nie było mowy - to z czego nagle wyciągać wnioski?
- "brzmią w sposób „muzyczny” – nawet gdy wygenerowany dźwięk znacząco różni się od docelowej barwy" nieścisłe, zbędne

# Zakończenie
- brakuje zakończenia - podsumowania tego, co w pracy zostało zrobione; to powinien być osobny rozdział, który może (wyjątkowo) być krótki; być może wystarczy rozbić rozdział nr 7, część do badań, część do podsumowania
- i teraz mała uwaga, dalszych prac jest zdecydowanie za dużo; to sugeruje, że praca jest bardzo niekompletna; proszę ogólnie wyznaczyć kierunki dalszych badań w 2-3 akapitach
- w skrócie: rodział ma mieć podsumowanie wykonanych prac, w odniesieniu do postawionego celu pracy i zadań szczegółowych oraz kierunki dalszych badań (ale bardzo syntetycznie)

# Literatura
- czas dostępu do źródeł internetowych
- ujednolicić cytowania -- np. w [25] brakuje wydawnictwa i roku wydania, w [24] i [32] i [33] (i w innych) różnie wyróżniono strony, czasami stron brakuje
- tutaj też proszę usunąć ramki przy odwołaniach; docelowo praca jest papierowa
- w [40] proszę skorzystać z "et al" (i tam gdzie jest więcej niż... powiedzmy 3 autorów)

