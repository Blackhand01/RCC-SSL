Ecco una panoramica dell’albero della cartella **“RCC_WSIs”** su Google Drive e dei formati presenti, con particolare attenzione ai file Excel.

### Struttura della cartella principale

| Sottocartella/file     | Tipo di elemento | Contenuto/Osservazioni                                                                                                                                                                                                                                             |
| ---------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Annotations_chromo** | Cartella         | Contiene immagini delle sezioni (estensione **.svs** e **.tif**) numerate da 1.svs a 13.tif.  L’unico file non‑immagine è l’Excel **CHROMO_patients_correspondence.xlsx** che, presumibilmente, mappa i codici delle diapositive ai pazienti del gruppo “CHROMO”.  |
| **Annotations_onco**   | Cartella         | Contiene serie di immagini (molti file .svs e .tif).  Dalla pagina sorgente emerge la presenza di **ONCO_patients_correspondence.xlsx** (formato *application/vnd.openxmlformats‑officedocument.spreadsheetml.sheet*), l’analogo per i pazienti del gruppo “ONCO”. |
| **ccRCC**              | Cartella         | Raccolta di immagini di istologia; il sorgente non riporta file .xlsx ma molti file **.svs** o **.tif**, quindi è una collezione di diapositive digitali.                                                                                                          |
| **CHROMO**             | Cartella         | Contiene numerose diapositive **.svs** (es. *HP20.2506_1338.svs* e analoghi).  Non sono presenti file Excel.                                                                                                                                                       |
| **ONCOCYTOMA**         | Cartella         | Non accessibile tramite il browser testuale; in base alla struttura del dataset è presumibilmente simile a ccRCC/CHROMO (immagini .svs).                                                                                                                           |
| **pRCC**               | Cartella         | Comprende una sottocartella **pRCC_xml** e numerosi file **.scn** (diapositive digitali, es. *HP17.7980.A2.pRCC.scn*, *HP18.11474.A6.pRCC.scn* ecc.).  È presente anche l’Excel **pRCC.xlsx** che probabilmente contiene la corrispondenza tra slide e pazienti.   |
| **pre**                | Cartella         | Contiene due cartelle denominate **ccRCC** e **pRCC**; sembra raccogliere versioni pre‑elaborate delle diapositive (non sono emersi file Excel).                                                                                                                   |
| **README**             | Cartella         | Comprende vari tipi di file: un archivio Python pickled (**H19‑754‑IHC‑ccRCC_crop_obj.pickle**), un’immagine **.tif**, un file **Readme.html**, un notebook **Readme.ipynb** e uno script **wsi_manager.py**.  Serve da materiale di supporto al dataset.          |
| **AAA‑README**         | Google Docs      | È un documento Google (file **.gdoc**) utilizzabile come introduzione generale; il sorgente indica il tipo *application/vnd.google‑apps.document* ma non ne permette l’esportazione diretta.                                                                       |

### Approfondimento sui file Excel

1. **CHROMO_patients_correspondence.xlsx** – localizzato in **Annotations_chromo**; è un file Excel (tipo *Microsoft Excel*) e dovrebbe contenere la tabella di corrispondenza tra le diapositive “CHROMO” e i rispettivi pazienti/campioni.
2. **ONCO_patients_correspondence.xlsx** – presente in **Annotations_onco**; analogo del precedente per il gruppo “ONCO”.
3. **pRCC.xlsx** – trovato nella cartella **pRCC**; presumibilmente contiene informazioni tabellari (paziente, codici slide, eventuali annotazioni) relative alle diapositive pRCC.

### Altri formati degni di nota

* **.svs**, **.tif**, **.scn** – file di immagini di microscopia a tutta vetrata (Whole Slide Images) in diversi formati.  Sono la parte predominante del drive.
* **.xml** – nella sottocartella **pRCC_xml**, il codice sorgente mostra file come *HP17.7980.A2.pRCC.xml* e *HP17.7980.B.pRCC.xml*; probabilmente sono annotazioni in formato XML per ciascuna diapositiva.
* **.pickle**, **.html**, **.ipynb**, **.py** – file presenti nella cartella **README** per documentazione e codice di supporto.

### Conclusione

Il drive *RCC_WSIs* è una raccolta di Whole‑Slide Images di vari tumori renali (ccRCC, pRCC, oncocitoma, cromofobo).  I file Excel (tre in totale) sono collocati nelle cartelle **Annotations_chromo**, **Annotations_onco** e **pRCC** e servono come tabelle di riferimento tra i pazienti e i nomi delle diapositive.  Non è possibile scaricarli direttamente dal browser testuale, ma la loro presenza e tipo sono confermati dal sorgente HTML.
