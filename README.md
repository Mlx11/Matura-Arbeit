
# Support Vector Machines zur Bildklassifikation

Dies ist der praktische Teil meiner Maturaarbeit *Support Vektoren zur Bildklassifikation - Theoretische Grundlagen und Implementation am Beispiel von Baumblättern*. 

## Erste Schritte

Dieser Schritte ermöglichen die Benützung des Programms. 

### Benötigte Software

Zum Ausführung des Programms empfiehlt sich die wissenschaftliche Python-Distribution Anaconda (https://www.anaconda.com/). Zur Entwicklung wurde Anaconda 4.4.10 mit Python 3.6.4 verwendet. 

Folgende Bibliotheken müssen zusätzlich über die Anaconda-Prompt installiert werden:

**cvxopt:**
```conda install -c anaconda cvxopt```

**mahotas:**
```conda install -c conda-forge mahotas```

**opencv3:**
`conda install -c menpo opencv3`



### Überprüfung der Bibliotheken

Zur Überprüfung der richtigen Installation aller Bibliotheken kann die Datei *Test.py* ausgeführt werden.

## Ordner Data
Im Ordner Data sind die für das Training der SVM benötigten Daten gespeichert. Der Unterordner *Leaves_BestParams* für jede Klasse einen Ordner mit einem Bild, einem X- und einem Y-Vektor pro Blatt. Zudem ist eine SVM zur Klassifikation dieser Daten und deren Test- und Trainingsdaten enthalten. 


## User Interface
Dieser Abschnitt erläutert die Funktionsweise des Interfaces, welches durch die Ausführung der Datei UI.py geöffnet wird.

![alt text][InterfaceCreate.png]

Unter *"SVM & Testdaten laden"* können gespeicherte SVMs und Daten geladen werden. Um ein oder mehrere Bild/er zu klassifiziern, muss mindestens eine SVM oder MultiSVM und X-Daten geladen sein. Werden zusätzlich noch y-Daten angegeben, so wird nicht die Klasse der Bilder sondern die Korrektklassifizierungsrate ausgegeben. 

![alt text][InterfaceLoad.png]

*"SVM erstellen"* erlaubt die Erstellung einer SVM mit gegebenen Parametern C und Kernel. Für eine hard-margin SVM muss die Checkbox markiert werden. Anschliessend müssen die Trainingsdaten geladen werden. Die erstellte SVM ist danach unter  *"SVM & Testdaten laden"*  geladen.
Je nach Anwendung empfiehlt sich anstelle des Interfaces direkt Pythoncode zu verwenden.


## Autor

* **Michael Linder** - *michael.linder@students.gymneufeld.ch* - [Mlx11](https://github.com/Mlx11)


## Lizenz

Das Programm steht unter der MIT-Lizenz (siehe licence.txt).
