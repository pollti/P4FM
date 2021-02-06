# P4FM

## Ansatz \& Implementierung

Unser Ansatz basiert auf [P1]………

Zunächst lesen wir eine Audiodatei als Signalvektor ein. Unsere Audiodatein sind zunächst mit einem Gerät und ohne Rauschunterdrückung aufgenommene WAV-Dateien mit einer Samplerate von 44,1kHz.

### Rauschreduzierung

Wir analysieren zunächst auf kleinen Fenstern das Vorhandensein von Sprache. Dazu nutzen wir auf den Fenstern eine Implementierung von Googles WebRTCVAD. Diese ist grundlegend nur für wenige Samplerates implementiert. Deshalb approximieren wir das Resultat durch Nutzung der logarithmisch nächsten Samplerate, was in unseren Tests zu sehr guten Ergebnissen führte. Aus der boolischen Klassifizierung der Einzelfenster nach Enthalten von Sprach-/Tonanteilen detektieren wir darauf basierend das größte Intervall mit reinem Rauschen in der gegebenen Audiodatei. Dieses wird benötigt, um eine Rauschreduzierung durchführen zu können. Die rauschreduzierte Version ist dann Grundlage für die Extraktion reinen Rauschens.

Mithilfe des zuvor bestimmten größten Rauschintervalls führen wir nun eine Rauschreduzierung auf üerlappenden Fenstern durch. Diese ist ähnlich zur Implementierung in Audacity. Dennoch funktionierte sie vermutlich durch flexiblere Parametrisierung und die automatische Erkennung eines geeigneten Rauschintervalls besser\footnote{Auch wenn es hier keine klare Metrik gibt basiert diese Aussage auf einem akustischen Anhören der resultierenden Audiodateien und der Betrachtung von Zeit-Frequenz-Plots indem die jeweiligen Energien durch Farben dargestellt werden. Jeweils kann das Verbleiben von Sprachfrequenzen subjektiv beurteilt werden.}. Eine erste Rauschapproximation erhalten wir nun durch Subktraktion des rauschreduzierten Signals vom Ursprungssignal.

Um verbleibende Sprachanteile „Voice Leakage“ weiter zu reduzieren, folgen wir dem Ansatz von [P2]. Dort wird Multiband Spectral Substraction vorgestellt. Basierend auf einer Matlab-Implementierung haben wir eine funktionell an unsere Bedürfnisse angepasste Python-Implementierung erstellt. Diese nutzt bei uns ebenfalls die erkannten Intervalle reinen Rauschens als Grundlage. Hierbei mussten wir zwar einen deutlichen Abfall der Gesamtenergie und damit einen potenziellen Präzisionsverlust verzeichnen, erhielten aber nach o. g. Kriterien auch sprachfreiere Ergebnisse.

### Umgebungsdetektion

Um das Rauschen verschiedener Aufnahmen verschiedenen Umgebungen zuzuordnen bzw. Zuordnungsplausibilität zu prüfen, vergleichen wir die Energieabweichungen aller Frequenzen zwischen Umgebungsrauschen und Rauschen der Aufnahme.

Zunächst generieren wir also pro Umgebung eine erwartete Energie pro Frequenz. Dazu berechnen wir den Median aus allen Fenstern (Länge unterschiedlich wählbar) aller an diesem Ort aufgenommenen Aufnahmen.

Zur Zuordnung bzw. Plausibilitätsprüfung berechnen wir den Median der Frequenzenergien nun also auch über die Fenster einer Aufnahme. Damit können wir nun durch verschiedene Abweichungsberechnungen verschiedene Metriken erhalten. Grundlegend mitteln wir die quadrierten Abweichungen aller Frequenzen und sehen Umgebungen mit kleineren Ergebnissen als plausibler an. Implementierte Metrikänderungen sind die folgenden:
* Lineare (oder andere wie Wurzel) Abweichungsberechnung statt quadrierter Fehler
    * funktionierte in Tests schlechter
* Logarithmische y-Skalierung vor Differenzberechnung und anschließende Anwendung der Exponentialfunktion
    * betrachtet prozentuale statt statt absoluten Abweichungen
    * beides ähnlich gut, abhängig von anderen Parametern
* Mean oder Median für Differenzen
    * derzeit: Mean über Differenzen eines Fensters
    * derzeit: Median über die Fenster einer Aufnahme
* Gewichtungen der Frequenzen (x-Achse)
    * Gleichverteilt oder niedrige Frequenzen bevorzugt → funktioniert gut
    * Bevorzugung niedriger Frequenzen, da dort nach FFT geringe Frequenzdichte pro Oktave → funktioniert gut
    * geringere Gewichtung des Sprachspektrums (implementiert durch zwei Gauß-Kurven) → funktioniert weniger gut

### Future Work

## Ergebnisse



### Plots

Designalisierung

Umgebungsübersicht

Frequenzplots der Umgebungen

### Erkennung

Wir haben in sechs Umgebungen je mindestens fünf Aufnahmen mit mindestens einem Gerät gemacht. Die Umgebungen sind die folgenden:
* Raum 1
* Raum 2 (ähnliche Größe, gleiches Haus)
* Treppenhaus (gleiches Haus, Fenster geöffnet)
* Bundesstraße
* Wald
* Windiger Platz


unter Decke/Fenster auf

### Bewertung

## Bedienung

Aufrufen, Paramter setzen, …








