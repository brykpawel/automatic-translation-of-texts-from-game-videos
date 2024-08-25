# Wprowadzenie

Celem pracy jest stworzenie prototypowego systemu pozwalającego na automatyczne wykrywanie i tłumaczenie tekstów występujących na zapisach video z gier (tzw. gameplays).

Prace będą składały się z następujących zadań głównych:

• zapoznanie się z tematyką i przeprowadzenie badań literaturowych w zakresie aktualnych sposobów wykrywania i tłumaczenia tekstów, oraz rozpoznawania mowy i jej tłumaczenia,

• zebranie odpowiednich danych do analizy (bazy danych dostępne publicznie oraz sekwencje video),

• opracowanie koncepcji systemu, wybór algorytmów i metod przydatnych w postawionym zadaniu,

• zestawienie stanowiska oraz weryfikacja koncepcji systemu poprzez implementacje jego kluczowych elementów,

• ewaluacja możliwości prototypu systemu, zaproponowanie dalszych kierunków prac oraz potencjalnych zastosowań w edukacji.


# Opis plików:

Pliki aplikacja.py, badanie_czasu.py, bleu_from_csv.py, t5_large_spell_test dotyczą ostatecznej wersji aplikacji, testów końcowych, a także testu modelu t5_large_spell. W celu uruchomienia kodu należy zainstalować niezbędne biblioteki (plik requirements_aplikacja.txt)

Plik easyocr_test_na_zdjeciach.py dotyczy testu biblioteki EasyOCR na pojedynczych zdjęciach. Testy wykonane na bazie danych MLT19-TestImages. Do uruchomienia kodu konieczna jest instalacja bibliotek (plik requirements_easyocr_test.txt)

Plik kerasocr_test_na_zdjeciach.py dotyczy testu biblioteki KerasOCR na pojedynczych zdjęciach. Testy wykonane również na bazie danych MLT19-TestImages. Do uruchomienia kodu konieczna jest instalacja bibliotek (plik requirements_kerasocr_test.txt)

Pliki MT5.ipynb i convolutional_model_by_fairseq.ipynb służą testom bibliotek i modeli do tłumaczenia tekstu. Testy wykonywane na bazie danych pobranej ze strony tatoeba.org. Niezbędne biblioteki są po uruchomieniu kolejnych części pliku Jupyter Notebook. Kod uruchomiono z użyciem platformy Google Colaboratory.

# Wymagania sprzętowe
W przypadku budowy aplikacji wykorzystano komputer z kartą graficzną (GPU) Nvidia
Geforce RTX 3070 Ti i pamięcią RAM 32 GB. System operacyjny zainstalowany
na komputerze to Windows 10. W celu uruchomienia programu z użyciem karty graficznej
zainstalowano kompatybilną wersję CUDA 11.8. Na dysku głównym zainstalowano również
interpreter języka Pythona w wersji 3.9.1.
Zalecane minimalne wymagania sprzętowe i programowe do uruchomienia programu:
• Karta graficzna NVIDIA wraz z zainstalowaną i kompatybilną do niej wersją
oprogramowania CUDA.
• Pamięc RAM: 16 GB lub więcej
• System operacyjny Windows 10/11, Linux
• Interpreter języka Python w wersji 3.6 lub wyższej
