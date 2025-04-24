
# Compilazione ed Esecuzione Principale

Questo documento descrive come compilare il progetto, preparare le matrici necessarie e lanciare i test di performance. Il sistema di build si basa su `CMake` e `make`.

## Compilazione

Per compilare il programma principale, che include tutte le varianti implementate:

1.  Assicurati di essere nella directory principale (root) del progetto, dove si trova il file `Makefile`.
2.  Esegui il comando:
    ```bash
    make build-test-stats
    ```
3.  Al termine della compilazione, verrà creata una directory `build/`. All'interno di questa directory troverai l'eseguibile principale  necessario per lanciare i test.

## Preparazione delle Matrici

I test richiedono la presenza di matrici in formato Matrix Market (`.mtx`) o altro formato supportato.

* **Posizionamento:** Le matrici devono trovarsi (o essere linkate) nella directory `mat/` presente nella root del progetto.
* **Metodi:**
    * Puoi copiare i file `.mtx` direttamente dentro `mat/`.
    * Puoi creare dei link simbolici dentro `mat/` che puntano alla posizione reale dei file delle matrici.
* **Script Ausiliario (`AddMatrici.sh`):**
    * Se stai lavorando sul server del dipartimento (o hai configurato lo script localmente), puoi usare `AddMatrici.sh`.
    * Questo script contiene un array che elenca le matrici da includere nei test quando si esegue la modalità "tutte le matrici". Puoi modificare questo array per selezionare le matrici desiderate.
    * **Importante:** L'esecuzione dello script `AddMatrici.sh` creerà automaticamente i link simbolici nella cartella `mat/` per tutte le matrici specificate nel suo array, prelevandole dalla directory standard `/data/matrici/` del server di dipartimento.

## Esecuzione dei Test

L'esecuzione dei benchmark avviene tramite il target `make run-test-stats`. La configurazione specifica dei test (numero di thread, iterazioni, matrici) viene passata tramite la variabile `ARGS`.

Ci sono due modalità operative:

### 1. Esecuzione su una Matrice Specifica

Per testare una singola matrice, specificane l'identificativo (es. `cant.mtx`) come terzo argomento in `ARGS`.

**Sintassi:**
```bash
make run-test-stats ARGS='"<threads>" <iterations> <matrix_id>'
```


### 2. Esecuzione su Tutte le Matrici Selezionate

Per eseguire i test su tutte le matrici definite nell'array dello script `AddMatrici.sh`, ometti il terzo argomento.

**Sintassi:**
```bash
make run-test-stats ARGS='"<threads>" <iterations> <matrix_id>'
```
É importante specificare le matrici nel file `AddMatrici.sh` poiche l'eseguibile copiera l'array al suo interno per prendere le matrici dalla cartella mat, quindi anche se non lo si utilizza esplicitamente per aggiungere le matrici bisogna comunque aggiungere le matrici nel array.