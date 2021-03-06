# Alberi di decisione con dati mancanti
Nella seguente repository viene implementato l'algoritmo di apprendimento per alberi di decisione con l'aggiunta di una strategia per la gestione dei valori mancanti degli attributi basata sull'assegnamento di una probabilità per ogni valore di un attributo in base ai valori noti negli esempi.

## Come usare il codice
Per poter utilizzare il codice è necessario scaricare i tre datasets utilizzati dai link specificati qui di seguito ed inserirli nella directory "**datasets**" dove sono già presenti i files *.name* in cui sono riportati i nomi degli attributi di ciascun dataset.

In tutti e tre i casi si dovrà scaricare il file con estensione *.data* per poter ottenere il dataset d'interesse.

I tre datasets usati sono:
1. **Tic-Tac-Toe Endgame**: reperibile a questo [link](http://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame).
2. **MONK's Problems**: reperibile a questo [link](http://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems). Scaricare tutti i file aventi estensione .train ed unirli in unico file chiamato monks.data.
3. **Balance Scale**: reperibile a questo [link](http://archive.ics.uci.edu/ml/datasets/Balance+Scale).

## Descrizione del codice
I file che compongono il progetto sono:
1. _**main.py**_: gestisce l'esecuzione dell'intero codice. Si occupa anche della creazione dei grafici, della divisione del dataset in training set, validation set e test set e della validazione.
2. _**performances.py**_: contiene i metodi per il calcolo dell'accuratezza:
   * _test_: per ogni elemento del test set verifica se il valore del target ricavato tramite l'albero di decisone coincide o meno con il suo valore effettivo
   * _accuracy_: calcola l'accuratezza dell'albero di decisione in funzione del numero di valori del target correttamente predetti
3. _**dataset.py**_: contiene i metodi per la gestione del dataset e per la rimozione dei valori:
   * _remove_data_: rimuove in modo casuale e uniforme con una data probabilità alcuni valori dagli esempi
   * _get_dts_name_: restituisce il nome del dataset in esame
   * _get_attributes_: restituisce il nome degli attributi che compongono il dataset e la posizione dell'attributo target
   * _get_examples_: permette di estrapolare dal file *.data* tutti gli esempi che compongono il dataset con la relativa classificazione memorizzata a parte
   * _get_attributes_values_: restituisce i valori di ciascun attributo
   * _get_target_values_: resistuisce i valori assunti dal target
4. _**dt_learning.py**_: classe che implementa l'algoritmo di apprendimento per alberi di decisione con la gestione dei valori mancanti:
   * _height_update_: aggiorna il valore dell'attributo height che tiene traccia della profondità dell'albero di decisione
   * _same_classification_: verifica se gli esempi forniti in ingresso al metodo hanno tutti la stessa classificazione (i.e. stesso valore del target)
   * _check_missing_: verifica se nell'insieme di esempi ricevuto in ingresso mancano dei valori
   * _get_values_: restituisce i valori di un certo attributo
   * _plurality_value_: fornisce il valore più frequente del target entro un certo insieme di esempi
   * _get_weighted_occur_: restituisce le occorrenze pesate per ogni singolo valore di un determinato attributo per un certo insieme di esempi
   * _get_prob_: calcola la probabilità di un certo attributo in relazione ai suoi valori osservati in un certo insieme di esempi
   * _get_entropy_: calcola l'entropia
   * _gain_: calcola il gain per ciascun attributo disponibile su un dato insieme di esempi fornendo poi l'attributo con il gain più grande
   * _dt_learning_: implementazione dell'algoritmo di apprendimento per alberi di decisione

## Linguaggio e librerie
Il progetto è stato realizzato utilizzando *Python 3.8* come linguaggio di programmazione. Sono inoltre state utilizzate alcune librerie esterne:
* **matplotlib**: libreria esterna per la creazione dei grafici, utilizzata per mostare i risultati. [[download](https://matplotlib.org/users/installing.html)]
* **termcolor**: per la stampa a colori sulla console. [[download](https://pypi.org/project/termcolor/)]

Sono state utilizzate anche le seguenti librerie di Python:
* **random**: per generare numeri casuali, rimuovere casualmente e uniformemente con una data probabilità alcuni valori e per scegliere casualmente degli elementi da una lista
* **math**: per eseguire calcoli matematici
* **copy**: per eseguire la deepcopy di alcune liste nel main
* **statistics**: per calcolare la moda delle altezze per cui si ha la maggior accuratezza
