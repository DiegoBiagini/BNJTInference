# BNJTInference
Project assigned as a final examination method for the course B003725 Intelligenza Artiﬁciale (2019/20), taught by professor Frasconi Paolo at Università degli studi di Firenze.  
More details about this project can be found in the file **Biagini_assignment.pdf**

## Installazione
Tutto il codice necessario ad eseguire questo progetto è nella cartella **code**.  
E' necessario installare le seguenti librerie se non già presenti sul sistema:
* numpy
* pickle
* console-menu
* networkx
E' possibile installarle direttamente attraverso il file requirements.txt  
```shell
pip install -r requirements.txt
```

## Uso Guidato

E' possibile eseguire il programma in modalità guidata, eseguendo il file main.py.  
Ciò non consente la creazione di nuovi modelli, ma rende possibile il caricamento dei modelli già presenti nella cartella models.  
Una volta caricato uno di essi è possibile inserire evidenza in nodi inseriti dall'utente, propagare l'evidenza e infine consultare le probabilità di qualsiasi variabile nel modello.  
Inoltre è possibile visualizzare il grafo della rete bayesiana e del junction tree.

## Uso libero
### Creazione di un nuovo modello
Per poter creare un nuovo modello è necessario importare i file bayes_nets.py e tables.py.  
E' possibile creare una variabile nel seguente modo:  
```python
var = Variable("Nome", "Descrizione", ["valore1", "valore2", ...])
```
Specificando un nome identificativo, una descrizione e i valori che tale variabile può assumere(alfanumerici o numerici, ma non entrambi per una stessa variabile).  
  
A partire da una o più variabili è possibile creare una tabella di probabilità:
```python
table = BeliefTable([varA, varB, ...])
```
Ed impostare le probabilità delle variabili nella tabella:
```python
table.set_probability_dict({'Nome': 'valore1'}, 0.01)
```
Per creare un modello di rete Bayesiana i passi sono:
* Crea l'oggetto BayesianNet  
```python
net = BayesianNet()
```
* Aggiungi delle variabili alla rete  
```python
net.add_variable(var)
```
* Inserisci un arco tra variabili  
```python
net.add_dependence(padre, figlio)
```
* Inserisci la tabella BeliefTable relativa a una certa variabile(consistente con la struttura della rete)
```python
net.add_prob_table(var, table)
```
  
Successivamente è necessario costruire manualmente il junction tree della rete bayesiana, i passi sono:
* Costruisci il JunctionTree con le variabili volute
```python
jtree = JunctionTree([var1, var2, ...])
```
* Aggiungi delle cricche al JunctionTree
```python
jtree.add_clique([var1, var2, ...])
```
* Aggiungi collegamenti tra cricche
```python
jtree.connect_cliques([var1, var2, ...], [var3, var4, ...])
```
Fatto questo il modello è pronto per essere usato.  
E' possibile salvarlo su file con la funzione `serialize_model(net, jtree, filename)`, disponibile importando il file util.py.  

### Uso del modello
Prima di tutto è necessario inizializzare il JunctionTree ai valori inseriti nella BayesianNet
```python
jtree.initialize_tables(net)
```

A questo punto è possibile inserire evidenza su variabili:
```python
jtree.add_evidence('Nome', 'valore1')
```

Propagare tale evidenza nel JunctionTree:
```python
jtree.sum_propagate()
```
E infine consultare le nuove probabilità delle variabili:
```python
jtree.calculate_variable_probability('Nome')
```