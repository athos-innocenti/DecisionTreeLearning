# Alberi di decisione con dati mancanti
Nella seguente repository viene implementato l'algoritmo di apprendimento per alberi di decisione con l'aggiunta di una strategia per la gestione dei valori mancanti degli attributi nel trainingset basata sull'assegnamento di una probabilità per ogni valore di un attributo in base ai valori noti negli esempi.

## Come usare il codice
Per poter utilizzare il codice è necessario scaricare i tre datasets utilizzati dai link specificati qui di seguito ed inserirli nella directory [*datasets*](https://github.com/athos-innocenti/DecisionTree/tree/master/datasets) dove sono già presenti tre files *.name* in cui sono riportati i nomi degli attributi di ognuno dei tre datasets.

In tutti e tre i casi si dovrà scaricare il file con estensione *.data* per poter ottenere il dataset d'interesse.

I tre datasets usati sono:
1. **Car Evaluation**: reperibile a questo [link](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation).
2. **Tic-Tac-Toe Endgame**: reperibile a questo [link](http://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame).
3. **Chess(King-Rook vs. King-Pawn)**: reperibile a questo [link](http://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29).

## Descrizione del codice
I file che compongono il progetto sono:
1. _**main.py**_: gestisce l'esecuzione dell'intero codice. Si occupa anche della creazione dei grafici e della divisione del dataset in trainingset, validationset e testset.
2. _**performances.py**_: contiene i metodi per il calcolo dell'accuratezza:
   * _average_: calcola la media sulle accuratezze ricavate da ciascuna interazione per ogni dataset
   * _test_: per ogni elemento del testset verifica se il valore del target ricavato tramite l'albero di decisone coincide o meno con il suo valore effettivo
   * _accuracy_: calcola l'accuratezza dell'albero di decisione in funzione del numero di valori del target correttamente predetto
3. _**dataset.py**_: contiene i metodi per la gestione del dataset e per la rimozione di alcuni valori dal training set:
   * _remove_data_: rimuove in modo casuale e uniforme con una data probabilità alcuni valori dagli esempi
   * _get_dts_name_: restituisce il nome del dataset in esame per poterlo stampare sulla console
   * _get_attributes_: restituisce il nome degli attributi e del target che compongono il dataset
   * _get_examples_: permette di estrapolare dal file *.data* tutti gli esempi che compongono il dataset
   * _get_attributes_values_: restituisce i valori di ciascun attributo
   * get_target_values_: resistuisce i valori assunti dal target
4. _**dt_learning.py**_: classe che implementa l'algoritmo di apprendimento per alberi di decisione con la gestione dei valori mancanti:
   * _same_classification_: verifica se gli esempi forniti in ingresso al metodo hanno tutti la stessa classificazione (stesso valore del target)
   * _get_classification_: restituisce il valore del target nell'esempio ricevuto in ingresso
   * _plurality_value_: fornisce il valore più frequente del target entro un certo insieme di esempi
   * _get_values_: restituisce i valori di un certo attributo
   * _get_occurrence_: restituisce le occorrenze per ogni singolo valore di un determinato attributo per un certo insieme di esempi
   * _check_missing_: verifica se nell'insieme di esempi ricevuto in ingresso mancano dei valori
   * _get_entropy_: calcola l'entropia
   * _information_gain_: calcola l'information gain per ciascun attributo disponibile su un dato insieme di esempi
   * _get_max_gain_: fornisce il massimo valore di information gain per poter identificare l'attributo con information più grande
   * _dt_learning_: implementazione dell'algoritmo di apprendimento per alberi di decisione
5. _**tree_pruning.py**_: classe composta dai metodi necessari per poter implementare il _pruning_ dell'albero di decisione appena creato così da poter risolvere il problema dell'_overfitting_.
   * _pruning_: richiama ricorsive il metodo _chi_squared_pruning_ fino al completamento del pruning
   * _chi_squared_pruning_ e _delta_: implementano il chi squared pruning dell'albero di decisione
   * _leaf_descendants_: verifica se un nodo dell'albero ha come figli delle foglie

## Linguaggio e librerie
Il progetto è stato relizzato utilizzando *Python 3.8* come linguaggio di programmazione. Sono inoltre state utilizzate alcune librerie esterne:
* **matplotlib**: libreria esterna per la creazione dei grafici, utilizzata per mostare i risultati. [[download](https://matplotlib.org/users/installing.html)]
* **termcolor**: per la stampa a colori sulla console. [[download](https://pypi.org/project/termcolor/)]

Sono state utilizzate anche le seguenti librerie di Python:
* **random**: per generare numeri casuali, riordinare casualmente una lista e sceglierne casualmente degli elementi
* **math**: per eseguire calcoli matematici
