Classificazione RCC con modelli SSL
A.1 Titolo e claim: “RCC subtype classification da WSI con SSL + XAI”
Il Renal Cell Carcinoma (RCC) comprende un insieme eterogeneo di neoplasie renali che originano prevalentemente dai tubuli renali; la classificazione istologica è clinicamente rilevante perché influenza prognosi, stratificazione del rischio e decisioni terapeutiche. In ambito anatomo-patologico, il riferimento (“gold standard”) per la categorizzazione è la valutazione microscopica di vetrini H&E digitalizzati come Whole-Slide Images (WSI), eseguita da patologi esperti. [2]
Nel progetto presentato, l’obiettivo è costruire un sistema di classificazione dei sottotipi RCC a partire da patch estratte da WSI, riducendo la dipendenza da annotazioni massive grazie a Self-Supervised Representation Learning (SSRL/SSL), e introducendo una componente di Explainable AI (XAI) per rendere ispezionabili le basi morfologiche delle decisioni del modello. La componente XAI è pensata come strumento di verifica qualitativa: non solo “quanto” il modello è accurato, ma “perché” prende una decisione e se il razionale è compatibile con pattern istopatologici attesi. [3]

A.2 Problema operativo: scala WSI e costo delle etichette
Le WSI sono immagini gigapixel: per essere utilizzate in deep learning vengono tipicamente suddivise in migliaia di crop/patch, con una predizione finale ottenuta aggregando le predizioni patch-level (es. majority voting o somma di probabilità). Questa dinamica di “esplosione” del numero di campioni rende evidente il collo di bottiglia: l’annotazione patch-level richiede tempo esperto e diventa rapidamente impraticabile su larga scala. [2]
In altre parole, anche quando esistono label a livello di paziente o vetrino, ottenere label robuste e granulari su regioni specifiche è costoso. Da qui nasce la motivazione per SSL: imparare rappresentazioni utili senza etichette, e usare un numero limitato di label solo per la fase supervisionata finale (es. linear probe / head di classificazione).

A.3 Perché è difficile: eterogeneità biologica, somiglianze morfologiche e variabilità tra patologi
Dal punto di vista clinico-patologico, il RCC è suddiviso in cinque sottotipi principali (clear cell, papillary, chromophobe, collecting duct, unclassified). Tra questi, i tre più frequenti sono clear cell (70–80%), papillary (14–17%) e chromophobe (4–8%). [1]
Queste percentuali introducono già un primo problema “ingegneristico-clinico”: le classi sono sbilanciate e alcune sono intrinsecamente rare.
La difficoltà non è solo statistica: esistono condizioni reali in cui la diagnosi differenziale è complessa perché:
Overlapping features tra neoplasie renali sono frequenti. [1]
In alcune casistiche, la concordanza inter-osservatore nella sotto-tipizzazione RCC è solo “poor to fair”, con valori medi riportati nell’intervallo 0.32–0.55. [2]
In diversi casi, l’immunoistochimica diventa un supporto indispensabile proprio perché la sola morfologia H&E può non essere sufficiente per separare entità con fenotipi sovrapposti. [1]
Questi punti guidano la scelta di metriche e obiettivi: non basta “accuracy media”, ma serve robustezza sulle classi clinicamente rilevanti e difficili, e strumenti per analizzare gli errori.

A.4 Focus CHROMO vs ONCO: differenziale critico, confusione diagnostica e parallelismo con imaging radiologico
Nel progetto il differenziale CHROMO vs ONCO è centrale perché rappresenta un caso reale in cui:
ONCO è una delle principali entità benigne (circa 10% dei tumori renali è in categorie benigne, e oncocytoma è tra le più comuni). [1][2]
La letteratura riconosce che la diagnosi differenziale tra chromophobe RCC e oncocytoma è difficile e soggetta a errori per sovrapposizione immunoistochimica e morfologica. [2]
Anche con pannelli immunoistochimici, il problema non è banale: sono riportate difficoltà e limiti nel trovare marcatori “definitivi” per separare oncocytoma e chromophobe RCC, specialmente in varianti eosinofiliche. [1]
Questo messaggio si rafforza ulteriormente se guardiamo l’ambito radiologico: in uno studio MRI su oncocytoma e chromophobe RCC, gli autori riportano che le due entità mostrano risultati di imaging simili e che nessuna feature MRI è risultata affidabile per distinguerle, con necessità di conferma istologica. [4]
Questa “doppia difficoltà” (anche su imaging non istologico) rende CHROMO↔ONCO un ottimo stress-test per un sistema ML: se il modello funziona qui, è più plausibile che stia imparando pattern realmente discriminanti.

A.5 Domanda guida: “SSL aiuta con poche label? e posso fidarmi del perché?”
Il progetto si articola attorno a due domande, una quantitativa e una qualitativa:
Efficacia con poche etichette (SSL → downstream supervisionato)
La letteratura evidenzia che, in domini come medicina/biologia computazionale, la carenza di dati etichettati è strutturale; l’obiettivo della SSRL è mitigare la necessità di annotazioni, imparando una rappresentazione riusabile e robusta. [2]
Affidabilità e verificabilità (XAI come controllo clinico)
Per controllare che le decisioni non dipendano da artefatti (bordi tessuto, pen-marks, bolle, bias di acquisizione) e che invece riflettano strutture istologiche plausibili, nel progetto applichiamo XAI post-hoc e organizziamo gli overlay per errore/classe/modello, con attenzione specifica alla separazione CHROMO vs ONCO. [3]
KPI promessi (quantitativi + qualitativi)
Macro-F1: misura globale robusta allo sbilanciamento tra classi. (Metriche attese nei risultati del progetto.) [3]
Min recall sulle classi tumorali / classi critiche: per evitare che il modello “vada bene in media” ma fallisca sulle classi più importanti o più difficili (es. CHROMO/ONCO). [3]
XAI sanity check clinico: verifica qualitativa che le regioni salienti corrispondano a pattern morfologici attesi e non ad artefatti; usata anche per identificare failure mode sistematici. [3]

B.1 Che dati avevamo: pazienti, WSI, classi, struttura delle annotazioni
Il dataset FP03 consiste di WSI H&E di biopsie renali con etichetta clinico-patologica a livello paziente in cinque classi: ccRCC, pRCC, chRCC (CHROMO), ONCO e NOT_TUMOR; ogni paziente può avere più WSI e, quando disponibili, annotazioni patologiche delimitano regioni tumorali e aree non neoplastiche (parenchima, necrosi, fibrosi).
Nel set inventariato risultano 197 WSI annotate, distribuite in: 125 WSI ccRCC, 48 WSI pRCC, 11 WSI chRCC, 13 WSI ONCO.
La classe NOT_TUMOR non corrisponde a pazienti dedicati: viene ottenuta a livello patch campionando regioni non neoplastiche all’interno delle stesse WSI tumorali (o ROI dedicate o tessuto apparentemente sano).
Origine delle regioni “omogenee” (fondamentale per capire il preprocessing):
Per ccRCC e pRCC sono presenti file .xml che identificano regioni omogenee nella WSI (tumor o non tumor).
Per CHROMO e ONCO sono disponibili ROI dedicate come immagini multirisoluzione (.svs/.tif) già ritagliate dall’originale e contenenti (approssimativamente) un singolo tipo di tessuto.
Questa dualità (“wsi+xml” vs “roi dedicate”) viene poi tracciata esplicitamente nel dataset patch-level tramite il campo origin con valori "wsi" e "roi".

B.2 Cos’è una WSI (Whole-Slide Image): piramide, MPP e gigapixel
Una WSI è un’immagine gigapixel acquisita a ingrandimenti elevati (tipicamente 20× o 40×) e memorizzata come piramide multi-risoluzione: livello 0 = massima risoluzione, livelli successivi = downsample progressivi. Questo è essenziale perché:
non è possibile caricare l’intera WSI a piena risoluzione in memoria;
l’estrazione patch avviene scegliendo un livello piramidale o ricampionando.
Nel nostro dataset, le WSI sono acquisite prevalentemente con obiettivo 40× e risoluzione nominale di circa 0.25 µm/pixel, con formati eterogenei (.scn, .svs, .tif) e backend diversi (Leica, Aperio, generic-TIFF).
Per gestire questa eterogeneità, raccogliamo metadati armonizzati (dimensioni, MPP, numero livelli piramidali, vendor, presenza XML) in CSV/Parquet per supportare preprocessing automatico.

B.3 Preprocessing WSI: normalizzazione risoluzione, tissue mask, stain handling
Il preprocessing è organizzato per ridurre variabilità non informativa e impedire che il modello impari scorciatoie (artefatti o sfondo):
Lettura e normalizzazione della risoluzione (MPP)
Le WSI multi-vendor sono caricate via librerie come OpenSlide, leggendo i metadati di risoluzione (micrometri per pixel, MPP). Selezioniamo il livello piramidale più vicino a una risoluzione target (es. equivalente a 20×) oppure ricampioniamo bilinearmente se necessario; se MPP non è affidabile, usiamo tag nativi o metadati specifici quando disponibili.
Rilevazione tessuto e mascheramento dello sfondo (tissue mask)
Per evitare patch di puro background costruiamo una maschera di tessuto a bassa risoluzione: conversione HSV, sogliatura automatica del canale di saturazione via Otsu, morfologia per rimuovere rumore/fori, filtro per area minima (soglia espressa in mm² alla MPP target), e upscaling alla risoluzione di estrazione patch.
Normalizzazione della colorazione H&E (stain normalization)
Per ridurre variabilità cromatica multi-centro adottiamo normalizzazioni configurabili: Macenko, Vahadane e, in alternativa, Reinhard (Lab) per scenari “light”. La scelta è configurabile per ablation su stain-handling.
Questo si integra con il razionale generale: in istopatologia normalizzazione e stain-augmentation sono strumenti consolidati per robustezza, e pipeline QC aiutano a mitigare artefatti (sfocatura, pen-marks, saturazione).

B.4 WSI → patch: concetto operativo (ROI mask + tissue mask)
Dopo preprocessing, la conversione WSI→patch avviene combinando:
tissue mask (per scartare sfondo),
ROI tumor/not_tumor (per guidare sampling e label patch-level).
Per ccRCC/pRCC generiamo maschere binarie tumor_Lk.npz e not_tumor_Lk.npz (a un livello piramidale k), e indicizziamo tutto in masks_index.jsonl insieme a record_id, patient_id, class_label e path WSI.
Per CHROMO/ONCO campioniamo direttamente dentro ROI raster dedicate, con cap per ROI/paziente, perché l’origine è “roi”.

B.5 Formato del patch dataset: 224×224, ~0.50 µm/px, chiave e metadati
A valle del preprocessing, costruiamo un dataset patch-level come tabella Parquet/JSONL, dove ogni riga è una patch 224×224 a circa 0.50 µm/px con metadati completi.
Ogni patch è identificata da una chiave stringa key del tipo:
<patient_id>/<record_id>/<x>_<y>

ad esempio HP13.7465/01109a1bfd65.../1784_9380, che codifica paziente, sorgente logica e coordinate di estrazione su WSI/ROI.
Campi principali (per tracciabilità e analisi errori/leakage):
patient_id (pseudonimizzato), class_label in {ccRCC, pRCC, CHROMO, ONCO, NOT_TUMOR}.
subset (train/val/test), definito rigorosamente a livello paziente.
origin = "wsi" (patch da WSI+XML) vs "roi" (patch da ROI dedicate CHROMO/ONCO).
coords include (x, y), level, patch_size e downsample_at_level0 (fattore scala).
roi_coverage: frazione tumorale/non tumorale nella patch stimata dalla maschera, utile per filtrare patch ambigue o analizzare casi di confine.
Questo schema rende possibile:
audit completo dall’output patch fino alla slide originale,
analisi per paziente (essenziale quando alcune classi hanno pochissimi casi),
slicing diagnostico mirato (es. NOT_TUMOR in contesto ccRCC vs pRCC tramite parent_tumor_subtype).

B.6 Split corretto (patient-first): 70/15/15 e motivazione anti-leakage
Per evitare leakage, lo split è definito a livello paziente: tutte le WSI di uno stesso paziente ricadono nello stesso split.
Lo schema riportato è 70/15/15, implementato via folds.json/logica patient-first.
Nel setup esplicitato, il numero pazienti per split e classe è:
train: ccRCC 38, pRCC 16, CHROMO 3, ONCO 3
val: ccRCC 9, pRCC 3, CHROMO 1, ONCO 1
test: ccRCC 9, pRCC 3, CHROMO 1, ONCO 1
Questi numeri implicano un totale di 88 pazienti (ccRCC 56, pRCC 22, CHROMO 5, ONCO 5) coerente con l’idea di dataset clinico sbilanciato e con il focus su classi rare.
Nota importante di progettazione: il task finale è patient-level; la specifica di progetto prevede aggregazione per paziente (majority voting sui crop) escludendo crop predetti come not-tumour.
Questo rende lo split patient-first non negoziabile: uno split patch-level avrebbe reso il voto “inquinato” da patch dello stesso paziente viste in train.

B.7 Generazione patch candidate: budget e cap per WSI/paziente/ROI
Prima del bilanciamento produciamo un universo di patch candidate per split (train/val/test) in file JSONL:
per ccRCC/pRCC: sampling guidato da maschere ROI (tumor / not_tumor) con budget globale 300k/60k/60k(train/val/test) e cap per WSI e per paziente; include anche downscaling dinamico delle maschere se troppo grandi.
per CHROMO/ONCO: sampling dentro ROI raster con cap per ROI e per paziente.
Questa fase è deliberatamente “abbondante”: generiamo un set ampio e poi applichiamo bilanciamento e filtri, mantenendo la possibilità di audit.

B.8 Problema imbalance reale: CHROMO/ONCO pochi casi → rischio overfit/leakage “mascherato”
Già a livello candidate, il dataset è fortemente sbilanciato: ccRCC/pRCC derivano da WSI con molte ROI e molti pazienti, mentre CHROMO/ONCO provengono da poche ROI/pazienti, con squilibri sia tra classi sia tra pazienti.
L’effetto pratico è che, senza controlli per-paziente, un modello potrebbe:
“memorizzare” pattern idiosincratici di un singolo paziente rare-class,
ottenere metriche patch-level buone ma generalizzare male a pazienti nuovi.
Nel paper evidenziamo esplicitamente il rischio: per CHROMO e ONCO, pochi casi contribuiscono con migliaia di patch ciascuno (ordine di grandezza ~6.5k a paziente).
Questo è il motivo per cui il bilanciamento non può essere solo “per classe”, ma deve essere anche intra-classe per paziente.

B.9 Bilanciamento train: target per classe + allocazione per paziente (con cap)
Il bilanciamento viene applicato solo al train.
Definiamo un dizionario targets[c] che fissa quante patch desideriamo per ciascuna classe; nel setup riportato i target sono nell’ordine di:
NOT_TUMOR 38,250; ccRCC 40,504; pRCC 27,505; CHROMO 30,642; ONCO 30,642.
Idea chiave: la classe più rara determina il budget; le altre classi vengono tagliate a uno stesso ordine di grandezza (T = min caps[c]) per ridurre dominance delle classi grandi.
Poi, per ciascuna classe tumorale (ccRCC/pRCC/CHROMO/ONCO), facciamo allocazione per paziente in due passaggi:
quota base qbase = floor(targets[c] / P) dove P = #pazienti con candidati in classe c
cap per paziente qcap = ceil(per_patient_cap_factor · qbase) con per_patient_cap_factor = 1.5
Se mancano patch per raggiungere il target di classe, il residuo viene assegnato solo ai pazienti che hanno ancora patch disponibili fino a qcap.
Le patch selezionate così sono marcate con selected_reason="balanced".
Bilanciamento NOT_TUMOR condizionato al tumore
NOT_TUMOR viene bilanciata per:
non superare il target globale targets["NOT_TUMOR"]
rispettare un rapporto tumorale/non tumorale ≈ 1:1 per paziente, quando possibile.
In caso di eccesso si applica un ulteriore sottocampionamento casuale senza rimpiazzo.

B.10 Numeri finali train/val/test e politica “val/test naturali”
Nel setup descritto, il train risulta quasi bilanciato con circa 18k–19.5k patch per classe, con conteggi espliciti:
NOT_TUMOR 19,402
ccRCC 18,332
pRCC 16,576
CHROMO 19,500
ONCO 19,500
Per validation e test non applichiamo bilanciamento: sono pass-through dei candidati, mantenendo distribuzioni naturali osservate nel dataset clinico; selected_reason non viene impostato.
In pratica, questo crea una separazione netta:
train = ottimizzazione controllata (anti-dominance + anti-memorization per paziente)
val/test = stima di generalizzazione su distribuzione reale (più vicina allo scenario clinico).

B.11 Serializzazione finale: da record JSONL/Parquet a WebDataset per SSL
A partire dai file selected_patches_{train,val,test}.jsonl costruiamo un WebDataset compatibile con training SSL:
per ogni split generiamo rcc_webdataset_final/<split> con shard shard-XXXXXX.tar da massimo 5,000 samples per shard.
per ogni record:
recuperiamo la patch 224×224 da WSI/ROI via OpenSlide quando possibile, altrimenti fallback PIL;
scartiamo patch che non rispettano esattamente la dimensione target;
serializziamo immagine come img.jpg/img.png e metadati come meta.json.
Ogni sample WebDataset è quindi {immagine, metadati strutturati} e conserva tracciabilità (patient_id/record_id/origin/source_rel_path), utile per analisi post-hoc di leakage e variabilità intra-paziente.

Sezione C. Perché SSL e tassonomia
C.1 Perché non basta il supervised “classico” nel nostro setting RCC da WSI
Nel nostro progetto, la classificazione dei sottotipi RCC parte da WSI (Whole Slide Images), cioè immagini istopatologiche gigapixel da cui estraiamo patch. Questo setting crea un paradosso operativo:
Dati grezzi abbondanti, label scarse.
Le WSI generano moltissime patch, ma le label affidabili (sottotipo RCC a livello paziente e, ancor più, annotazioni ROI patch-level) sono costose. In generale, nei progetti ML una quota significativa del tempo può essere assorbita dall’annotazione (ordine di grandezza ~25%) ; in patologia questo costo aumenta perché le immagini sono enormi, la colorazione varia e le aree patologiche possono occupare una frazione dell’intera slide .
Pochi casi per classi rare → rischio di overfitting e bassa generalizzazione.
Nel nostro dataset alcune classi sono meno rappresentate (es. CHROMO/ONCO e altri sottotipi rari) e il supervised puro tende a “memorizzare” scorciatoie quando i segnali discriminativi sono sottili o confondibili (tema che riemergerà nel focus ONCO vs CHROMO). In più, la patologia digitale soffre di fattori non semantici ma dominanti: assenza di orientazione canonica e natura fortemente “testurale” di molti pattern .
Serve un paradigma che sfrutti l’unlabeled e induca feature robuste.
La SSL nasce esattamente per ridurre la dipendenza da label manuali e sfruttare la struttura interna dei dati . Per noi, significa: usare milioni di patch potenzialmente non annotate per imparare rappresentazioni utili, e poi usare le poche label affidabili per specializzare il modello sul task RCC.

C.2 Tassonomia SSL (e dove si posizionano i modelli che abbiamo testato)
Per evitare confusione terminologica, adottiamo una tassonomia standard “per obiettivo” che riassume la SSL in tre macro-categorie: generative, contrastive, generative-contrastive (adversarial) . Questa tassonomia è utile perché collega direttamente:
Cosa il modello deve predire/ottimizzare (ricostruzione vs discriminazione vs gioco adversarial),
Che architettura tipicamente usa (encoder+decoder vs encoder-only + loss contrastiva),
Che tipo di bias/inductive prior introduce (es. invariance indotta via augmentations).
Nel nostro progetto ci siamo concentrati soprattutto sulle famiglie oggi più forti per vision con ViT:
Contrastive / instance discrimination (encoder-only + InfoNCE-like).
Qui l’obiettivo è rendere simili due viste della stessa immagine e dissimili quelle di immagini diverse; metodi come MoCo formalizzano questa idea con un grande set di negative e un encoder “key” aggiornato con momentum . Questa famiglia è storicamente molto efficace in downstream discriminativi.
Joint-embedding / distillation / non-contrastive invariance-based (senza negative esplicite o con teacher-student).
Rientrano nei metodi “invariance-based” che costruiscono più viste e forzano allineamento tra embedding; survey e letteratura mostrano che la qualità dipende molto da strategie di sampling e augmentations multi-view .
Predictive (JEPA / I-JEPA): predire rappresentazioni mancanti invece di ricostruire pixel.
I-JEPA formalizza una terza via: non generativa (non ricostruisce pixel), ma predittiva nello spazio delle rappresentazioni, e punta a rappresentazioni “semantiche” senza affidarsi a view-augmentations hand-crafted .
La chiave concettuale è distinguere tra:
Joint-Embedding Architecture (JEA): allineare embedding di viste compatibili,
Generative Architecture: ricostruire il segnale,
Joint-Embedding Predictive Architecture (JEPA): predire embedding di una parte dall’altra .
Nel progetto, questa tassonomia ci è servita per un motivo preciso: non volevamo “un solo SSL”, ma capire quale meccanismo produce feature più trasferibili e più affidabili nel dominio istopatologico, dove le scorciatoie sono frequenti.

C.3 Augmentations: non sono un dettaglio, sono parte dell’algoritmo
In pratica, molte differenze tra metodi SSL non sono solo nella loss, ma nella costruzione del segnale di training (le viste).
Nei metodi invariance-based, le viste positive sono costruite con trasformazioni “hand-crafted” (crop/scale, jitter colore, ecc.) .
Questo porta due implicazioni chiave:
Le augmentations inducono invariance: il modello impara a ignorare ciò che le trasformazioni cambiano.
Le augmentations introducono bias: queste invariance possono essere dannose se rimuovono segnali diagnostici o se non sono coerenti col dominio (ad esempio alterazioni di colore che in istologia possono essere informative) .
In survey SSL, è esplicitato che le strategie multi-view di data augmentation sono un fattore determinante per le prestazioni dei metodi contrastivi e che la teoria del “perché” aiutino non è sempre chiara, limitando la portabilità ad altri domini .
I-JEPA nasce anche come risposta a questo limite: mostra che si possono apprendere rappresentazioni forti senza appoggiarsi a view augmentations hand-crafted, usando una strategia di masking e predizione in representation space . Inoltre, evidenzia che i metodi basati su viste multiple richiedono processare più view e questo impatta scalabilità/efficienza .
Cosa abbiamo fatto noi (scelta pratica per comparabilità):
Per confrontare in modo “fair” più famiglie SSL su patologia renale, abbiamo adottato un set di augmentations coerente con la letteratura istopatologica e con la nostra pipeline, includendo sia componenti fotometriche (stain/colore) sia geometriche. Nel nostro progetto, ad esempio, abbiamo esplicitamente incluso una fase di stain normalization e l’uso di una tissue mask per limitare l’apprendimento da background . Inoltre, il riferimento FP03 descrive che le augmentations sono condivise tra modelli, con varianti per analisi/ablazioni .
Questa scelta serve a isolare l’effetto della famiglia SSL dalla variabilità introdotta da pipeline di augmentations completamente diverse.

C.4 Rischi della SSL in pathology (e perché li abbiamo considerati “di progetto”, non un dettaglio)
La SSL in patologia non fallisce “solo” per hyperparameter sbagliati: fallisce spesso per ragioni strutturali del dominio.
Shortcut learning su texture/stain e assenza di orientazione canonica.
In istopatologia molte decisioni sono texture-driven e non c’è orientazione canonica delle strutture; questo rompe assunzioni implicite dei modelli e rende facile apprendere correlazioni spurie .
Mitigazione nel nostro approccio: normalizzazione del colore e filtering del background (tissue mask) per ridurre scorciatoie non semantiche .
Domain shift: scanner, artefatti, variabilità di stain.
Le review recenti sui foundation model in medical imaging sottolineano che nelle WSI sono frequenti artefatti e variazioni di stain, e questo produce shift rilevante .
Mitigazione: pipeline di preprocessing e augmentations controllate (stain-aware), oltre a split patient-first (già trattato nella sezione B) per evitare leakage.
Problema dei “falsi negativi” nel contrastive su patch istologiche.
Nel contrastive, due campioni diversi sono trattati come negative; ma in WSI è comune che patch diverse siano semanticamente molto simili. Questo rende probabili “false negative pairs” che degradano l’apprendimento .
Questa è una delle ragioni per cui nel progetto ha senso includere e confrontare anche famiglie non puramente contrastive (distillation/joint-embedding, predictive), per capire se riducono questa fragilità.
Trade-off locale vs globale (patch-level ≠ caso clinico).
La SSL su patch può imparare feature locali eccellenti, ma il task clinico è spesso multi-scala e multi-focale. Questo impatta direttamente l’interpretabilità: se il modello “vince” ma basandosi su micro-pattern non correlati alla diagnosi, l’XAI diventa essenziale (anticipazione della sezione XAI).
Conclusione della sezione C:
La SSL non è stata scelta come “moda”, ma come risposta strutturale a tre vincoli: costo label, dominio istopatologico non canonico (texture/stain/orientazione), e necessità di feature trasferibili. La tassonomia (contrastive vs distillation/joint-embedding vs predictive) ci ha permesso di trasformare una domanda vaga (“SSL funziona?”) in una domanda misurabile: quale meccanismo di SSL genera rappresentazioni più robuste e meno soggette a scorciatoie nel nostro dataset RCC?

D.1 Setup comune (valido per tutti i modelli)
Il nostro approccio è tile/patch-based: ogni WSI viene convertita in patch (descritto nella sezione B), e i modelli SSL vengono pre-addestrati a livello di patch su dati unlabeled; successivamente valutiamo le rappresentazioni tramite downstream supervision (linear probe / transfer learning) (vedi [1]).
Per garantire una comparazione corretta, abbiamo fissato una pipeline comune:
Backbone e varianti supervisionate
Per i modelli SSL abbiamo usato un backbone ViT (nel progetto è citata esplicitamente l’implementazione su ViT-B/16 per la comparazione tra strategie SSRL) (vedi [1]).
Come controllo supervisionato abbiamo incluso una baseline ResNet-50 addestrata end-to-end, più una variante di transfer learning (vedi [1]).
Downstream: come trasformiamo le feature SSL in predizioni cliniche
Dopo il pretraining SSL, valutiamo le rappresentazioni con due modalità standard:
Linear probing: backbone congelato e classificatore lineare (setup tipico per confrontare la “qualità” delle feature) (vedi [1]).
Transfer learning: fine-tuning supervisionato (parziale o totale, a seconda della configurazione sperimentale) per misurare il guadagno pratico sulle classi RCC (vedi [1]).
Aggregazione patient-level
Poiché il dato è patch-based ma l’obiettivo è patient-level, per l’inferenza abbiamo usato due regole di aggregazione:
probability-sum: somma delle probabilità di classe sulle patch di un paziente;
vote: voto di maggioranza sulle etichette patch-level.
Inoltre, per ridurre rumore, le patch predette come NOT_TUMOR possono essere escluse dall’aggregazione (vedi [1]).

D.2 MoCo v3 (contrastive / momentum)
Idea e obiettivo. MoCo v3 è un metodo contrastive: prende due viste augmentate della stessa immagine, produce una coppia (query, key) e ottimizza una InfoNCE in cui i negativi sono gli altri campioni del batch (vedi [2]).
Componenti chiave (paper → nostra implementazione).
Due encoder: f_q (online) e f_k (momentum encoder), con update EMA (vedi [2]).
Proiezione + predictor sull’online encoder: l’algoritmo standard usa backbone + proj MLP + pred MLP per f_q e backbone + proj MLP per f_k (vedi [2]).
Loss simmetrizzata: ctr(q1,k2)+ctr(q2,k1) (vedi [2]).
Cosa abbiamo fatto nel codice (moco_v3.py).
La nostra classe MoCoV3 replica esattamente questo schema:
due viste globali (o “stack” multicrop), forward su backbone_q/backbone_k, projector MLP, predictor MLP per la branch q;
negativi in-batch via prodotto matriciale q @ k.T e cross-entropy con label sulla diagonale (stessa idea del pseudocodice ufficiale) (vedi [2]).
aggiornamento EMA del teacher con schedule a coseno verso 1.0 (opzionale in config), in linea con la prassi MoCo/BYOL-like (vedi [2]).
Perché è interessante in WSI/patch (motivazione pratica).
MoCo v3 tende a funzionare bene quando:
le augmentations producono viste “hard” ma semanticamente consistenti;
il batch effettivo è abbastanza grande da rendere utili i negativi in-batch (vedi [2]).

D.3 DINO v3-style (self-distillation + patch loss)
Idea generale. DINO appartiene alla famiglia teacher-student distillation “negative-free”: lo student impara a predire le assegnazioni/prototipi del teacher su viste diverse della stessa immagine. La variante DINOv3 combina più termini (CLS + patch + regolarizzazioni) in un’unica ricetta (vedi [3]).
Cosa abbiamo implementato (dino_v3.py).
Nel nostro codice, il modello DINOv3 include:
Teacher-student con EMA
teacher backbone = copia dello student + congelato; update EMA a ogni step (ema_update) (coerente con distillation frameworks) (vedi [3]).
Loss sul CLS (DINO loss)
head dedicata al CLS (dino_head) e loss DINO con normalizzazione tipo clustering/Sinkhorn (nel codice DINOLoss(...).sinkhorn_knopp_teacher(...)), cioè target del teacher come “soft assignment” su prototipi (vedi [3]).
Loss sui patch token (iBOT patch loss)
oltre al token globale, aggiungiamo un termine sui patch tokens mascherati: la ricetta DINOv3 esplicita l’uso di una componente iBOT con mascheramento (mask ratio) (vedi [3]).
Regolarizzazioni (KoLeo + Gram)
KoLeo per favorire una distribuzione più “uniforme” delle feature;
Gram objective per aggiungere un vincolo strutturale sui token (nel nostro codice parte dopo una frazione dei passi, gram_start_frac) (vedi [3]).
Nota implementativa importante (adattamento ingegneristico).
Poiché i backbone timm non sempre supportano “token drop” nativo, nel nostro DINOv3 applichiamo il mascheramento iBOT in modo compatibile: zeroing a livello pixel delle regioni corrispondenti ai patch mascherati, mantenendo invariata la shape input per il backbone.

D.4 iBOT (masked image modeling + distillation)
Idea e obiettivo. iBOT unisce:
una loss tipo DINO sul token globale (CLS),
e una loss di distillazione sui patch tokens mascherati (masked image modeling in token space), sempre con teacher EMA e centratura per stabilizzare (vedi [4]).
Cosa abbiamo fatto nel codice (ibot.py).
La nostra classe IBOT segue l’Algoritmo 1 concettuale:
due viste globali u,v;
masking solo nello student in modo block-wise, con mask ratio campionato (nel codice: 50% r=0, 50% r~U[min,max]);
teacher su immagini non mascherate;
loss totale = L_cls + L_MIM dove L_MIM è media della distillazione sui soli token mascherati (vedi [4]).
Abbiamo inoltre implementato:
centri separati per CLS e patch (center_cls, center_patch) con update momentum, coerente con la pratica “centering” per evitare collassi nelle distillation-based methods (vedi [4]).
Perché iBOT è rilevante nel nostro caso.
Su patch istologiche, l’informazione discriminativa è spesso locale: introdurre un termine esplicito sui token di patch(non solo sul globale) rende naturale testare iBOT nel nostro confronto (vedi [4]).

D.5 I-JEPA (predictive in embedding space)
Idea generale. JEPA/I-JEPA è un paradigma predictive: invece di imporre invariance via augmentations o via contrastive negatives, lo student usa un context encoder e un predictor per predire le rappresentazioni (embedding) di regioni target viste dal teacher/target encoder (vedi [5]).
Il punto chiave è che il training è formulato come predizione in spazio embedding tra contesto e target (vedi [5]).
Cosa abbiamo fatto nel codice (i_jepa.py).
La nostra implementazione IJEPA è adattata al dominio WSI/patch nel modo seguente:
Teacher/Target encoder e student/context encoder
teacher = copia EMA dello student (update_teacher), congelato;
teacher vede tutti i token dell’immagine (full tokens), student vede solo token di contesto (masking prima dei blocchi ViT).
Maschere context/target a blocchi
generiamo per ogni sample:
un set di target blocks (num_target_masks),
un context block grande, evitando (per quanto possibile) overlap con l’unione dei target.
Predictor transformer
un piccolo Transformer (IJEPA_Predictor) prende i token di contesto e le positional embedding dei target e produce predizioni dei token target in spazio embedding.
Filtro tessuto/sfondo (adattamento pathology)
introduciamo una tissue mask patch-level: stimiamo lo sfondo via deviazione standard del patch e pesiamo la loss MSE solo sui patch con tessuto, per evitare che lo sfondo domini l’ottimizzazione.

D.6 Baseline supervision / transfer (ResNet)
Abbiamo incluso una baseline supervisionata perché serve come controllo sperimentale: ci dice “quanto” della performance finale deriva davvero dal pretraining SSL e quanto invece è raggiungibile con supervised training standard a parità di dataset (vedi [1]).
Nel progetto è esplicitato:
ResNet-50 supervised end-to-end;
transfer learning come variante downstream standard (vedi [1]).

D.7 Confronto sintetico (tabella “modello → objective → input → failure mode atteso”)
Metodo
Famiglia
Obiettivo ottimizzato
Views / masking usati
Teacher/EMA
Rischio/failure mode “tipico” (da monitorare)
MoCo v3
Contrastive
InfoNCE con negativi in-batch (vedi [2])
2 global crops (aug) (vedi [2])
fk EMA (vedi [2])
instabilità su ViT se ricetta/BN/batch non adeguati (vedi [2])
DINO v3-style
Distillation
DINO CLS + iBOT patch + regolarizzazioni (vedi [3])
global + local, masking per iBOT (vedi [3])
teacher EMA
collasso/shortcut se centering/temperature non ok (monitoring necessario)
iBOT
Masked + distillation
CLS distill + patch distill su mascherati (vedi [4])
2 global views + blockwise masking
teacher EMA + centers (vedi [4])
ricostruzione di texture non diagnostica se masking/aug non calibrati
I-JEPA
Predictive
predizione embedding target da contesto (vedi [5])
blocchi context/target, no “hard” invariance obbligatoria
teacher EMA
“under-constraint” se context/target troppo facili → rappresentazioni poco semantiche
ResNet supervised
Supervised
cross-entropy su label
n/a
n/a
overfit su classi rare; dipendenza forte da etichette


D.8 Scelte implementative comuni (optimizer, schedule, batch, ecc.)
Per mantenere il confronto comparabile, abbiamo adottato un recipe di training comune, riportato nel progetto:
AdamW, linear warm-up e cosine schedule;
early stopping;
gradient accumulation per stabilizzare l’ottimizzazione;
batch size fissato (in quella configurazione: 128) (vedi [1]).
Infine, abbiamo organizzato lo studio come ablation sulle componenti che tipicamente influenzano SSL su patch (augmentations, batch size, sampling strategy, masking ratio, ecc.), esplicitate come assi di sperimentazione nel progetto (vedi [1]).

E.1 Training protocol: SSL pretraining → linear probe / fine-tune → test
Il training è stato progettato come pipeline multi-stadio, in cui ogni stadio produce artefatti (checkpoint, metriche, predizioni) riutilizzati nello stadio successivo.
Stadio 1 — SSL pretraining (patch-level, label-agnostico)
Per tutti i metodi SSL abbiamo adottato un backbone comune (default ViT-B/16, salvo ablation). L’addestramento SSL è stato eseguito per un numero limitato di epoche (parametrizzato in configurazione), con AdamW, warm-up e cosine decay; il batch effettivo è stato gestito tramite gradient accumulation, che abbiamo considerato anche come asse di ablation quando necessario.
Questo stadio ottimizza una rappresentazione patch-level coerente con la strategia SSL scelta (contrastive, distillation/masked, predictive). L’output principale è un insieme di checkpoint SSL, più i log delle loss/componenti.
Stadio 2 — Linear probe (valutazione “pura” della rappresentazione)
Per ogni checkpoint SSL abbiamo eseguito un linear probe: backbone congelato e addestramento di un singolo strato lineare (logit) con cross-entropy e pesi di classe (per compensare lo sbilanciamento).
Questo step è cruciale perché isola l’effetto della rappresentazione SSL: se il probe migliora, significa che lo spazio delle feature è più separabile per i sottotipi RCC.
Stadio 3 — Fine-tuning / Transfer learning (valutazione “realistica” del modello)
In parallelo al probe abbiamo considerato:
Baseline supervisionata: training end-to-end di un backbone (es. ResNet-50) direttamente sulle patch annotate.
Transfer learning: fine-tuning di backbone SSL o di modelli pre-addestrati (es. ImageNet), per misurare quanto la pre-inizializzazione influisce sulla generalizzazione.
Stadio 4 — Test + aggregazione patient-level
La valutazione è stata eseguita a due livelli:
Patch-level: accuracy, macro-F1, precision/recall per classe.
Patient-level: aggregazione delle predizioni delle patch appartenenti allo stesso paziente, escludendo NOT_TUMOR in fase di decisione paziente. Le due strategie implementate sono:
prob_sum: somma delle probabilità per classe sulle patch del paziente.
vote: voto di maggioranza sulle predizioni hard.
Le metriche patient-level principali che guidano la selezione sono macro-F1, macro-AUC e min recall sulle classi tumorali.
La scelta del checkpoint migliore avviene con early stopping su validation.

E.2 Augmentations effettive: “cosa vede davvero il modello”
Le augmentations non sono un dettaglio estetico: in SSL definiscono quali invarianti il modello può imparare e quali shortcut rischia di sfruttare. Per questo, la “ricetta” di viste/crop è stata coerente con la famiglia del metodo.
DINO v3-style (multi-crop)
Abbiamo usato due crop globali e 4–6 crop locali, con scale global/local nello stesso range di DINO/DINOv2. Le viste globali includono color jitter marcato, blur e solarization su una vista; le viste locali usano scale ridotte e blur moderato. Inoltre, HED-jitter è applicato a tutte le viste per gestire variabilità di stain.
iBOT (masked distillation + DINO-like views)
Per iBOT abbiamo mantenuto viste tipo DINO (quindi compatibili con multi-crop) e aggiunto masking blockwise dei token patch nello student. La mask ratio è stata un asse centrale di ablation, includendo sia valori fissi sia schedule (es. dal 70% verso il 40%); abbiamo inoltre considerato la scelta teste CLS/patch condivise vs separate.
I-JEPA (predictive, masking geometrico in embedding space)
Per I-JEPA abbiamo seguito la formulazione senza view augmentation esplicite: invece di costruire due viste tramite augmentations aggressive, selezioniamo un blocco di contesto (scala ~0.85–1.0) e M blocchi target non sovrapposti (scala ~0.15–0.20). Eventuali trasformazioni colore (HED-jitter o normalizzazione cromatica) sono applicate simmetricamente a contesto e target per non introdurre segnali spurî.
MoCo v3 (contrastive, due viste globali)
Per MoCo v3 la costruzione delle viste resta centrata su due global view coerenti, con assi di ablation dedicati a temperatura InfoNCE, momentum EMA, uso multicrop e stain-handling.
Operativamente, abbiamo “validato” visivamente le augmentations (montage) per assicurarci che:
il tessuto rimanesse informativo (no distruzione totale della morfologia),
lo shift cromatico simulasse variabilità realistica H&E,
blur/solarization non introducessero pattern artificiali dominanti.

E.3 Ablations: struttura, tracciamento e lettura dei risultati
L’ablation study è stato progettato per rispondere a una domanda semplice: quali leve cambiano davvero le metriche cliniche, e quali invece introducono solo costo computazionale.
Design per-modello (assi mirati)
Nel documento di progetto abbiamo formalizzato gli assi principali per ciascun metodo:
MoCo v3: temperatura InfoNCE, momentum EMA, dimensione projector, multicrop, stain-handling.
DINO v3-style: temperature student/teacher, #local crop, scale global/local, capacità projector, pacchetti augmentazione.
iBOT: mask ratio + schedule, temperature/Sinkhorn, #prototipi, teste condivise/separate, multicrop, stain-handling.
I-JEPA: scala/aspect del contesto, #/scala target, capacità predictor, EMA teacher, scelte spazio di previsione e pipeline colore.
Codifica e riproducibilità
Ogni run di ablation è codificata come configurazione versionata exp_*_ablXX.yaml (o equivalente), lanciabile via SLURM; questa scelta rende ogni risultato tracciabile e replicabile (stesso seed/config → stesso run).
Output di valutazione e “single source of truth” per il confronto
Per rendere confrontabili i run, abbiamo standardizzato gli artefatti di evaluation. In particolare, lo script di aggregazione patient-level:
scansiona una root di esperimenti, trova l’ultima eval con predictions.csv, aggrega e scrive risultati in .../per_patient/, aggiornando anche un runs_summary_patient.csv globale.
implementa due metodi (prob_sum, vote) e esclude sempre NOT_TUMOR, usando tutte le patch disponibili.
produce patient_predictions.csv, metrics_patient.json, confusion matrix patient-level e file di conteggi (pazienti valutabili vs solo-non-tumore).
Metriche tracciate (patient-level) per ranking/selection
Nel calcolo patient-level vengono salvate almeno: accuracy, balanced accuracy, macro-F1, e—quando disponibili logit/probabilità—anche macro-AUC OvR, macro-AUPRC e top-2 accuracy.
Queste metriche alimentano direttamente il ranking tra run, mentre la confusion matrix patient-level supporta l’analisi dei failure mode (in particolare chRCC↔ONCO).

E.4 Selezione modello: criteri quantitativi + vincoli clinici
La selezione non è stata “massimizza una metrica e basta”. È stata una procedura a vincoli:
Vincolo di sicurezza clinica (tumor recall)
L’obiettivo principale è alzare il “worst-case”: min recall sulle classi tumorali (ccRCC, pRCC, chRCC, ONCO). È esplicitamente una metrica target del progetto a livello paziente.
Ottimo globale (Macro-F1 / macro-AUC / macro-AUPRC)
Una volta soddisfatto (o massimizzato) il vincolo sul recall minimo, consideriamo metriche globali per evitare modelli che “vincono” solo su una classe:
macro-F1 e balanced accuracy (robuste a sbilanciamento),
macro-AUC OvR e macro-AUPRC quando disponibili.
Stabilità e generalizzazione (validation-driven)
La scelta del checkpoint avviene con early stopping su validation, riducendo overfit e leakage; il test viene usato come misura finale e non come set di tuning.
Analisi di errore guidata dal dominio
Per i candidati migliori, la confusione chRCC vs ONCO è considerata critica: non basta avere Macro-F1 alto se il modello collassa sulle classi più difficili/rare. (Questa parte si collega naturalmente alla sezione XAI: la selezione finale viene anche “validata” qualitativamente sulle evidenze morfologiche osservate).

F. Evaluation — obiettivo e criteri
La fase di evaluation ha l’obiettivo di produrre una misura comparabile, riproducibile e “paper-ready” delle performance dei modelli (MoCo v3, DINO v3, iBOT, I-JEPA, più baseline supervised/transfer), mantenendo traccia degli artefatti necessari per:
confronto quantitativo tra approcci SSL,
analisi degli errori (in particolare ONCO vs CHROMO e separazione tumore vs non-tumore),
collegamento con explainability patch/ROI tramite metadati e coordinate.
Abbiamo strutturato l’evaluation su due livelli:
Patch-level (primario): ogni patch è un campione indipendente, con label tra {ccRCC, pRCC, CHROMO, ONCO, NOT_TUMOR}; l’ordine classi di default è fissato e può essere letto dal JSON di metriche .
Patient-level (secondario, post-process): le patch vengono aggregate per paziente per ottenere una predizione più coerente con l’uso clinico, con regole esplicite e ripetibili .

F.1 Metriche (patch-level)
A patch-level abbiamo calcolato e salvato le seguenti metriche globali:
Accuracy: accuratezza complessiva.
Balanced Accuracy: media delle recall per classe (mitiga imbalance).
Macro-F1: media semplice dell’F1 sulle classi (penalizza modelli che “collassano” sulle classi rare).
Queste sono calcolate direttamente da y_true e y_pred .
Quando disponibili i logits (salvati), calcoliamo anche:
Macro-AUROC (One-vs-Rest) su probabilità softmax, binarizzando le label multi-classe.
Macro-AUPRC (average precision macro) sempre sulle probabilità.
Queste metriche vengono computate e registrate nel JSON di evaluation .
Nota operativa importante: salviamo anche logits_test.npy per ricostruire analisi successive (es. curve ROC/PR dettagliate, thresholding, calibration) senza dover rieseguire l’inferenza .
Min tumor recall (KPI clinico interno): oltre alle metriche aggregate, abbiamo definito un KPI sintetico utile per evidenziare comportamenti “pericolosi”:
min tumor recall = minimo della recall tra le classi tumorali {ccRCC, pRCC, CHROMO, ONCO} (escludendo NOT_TUMOR).
Operativamente si estrae dal report_per_class.json generato con classification_report .

F.2 Patch-level results: tabella principale “paper-ready”
Ogni run di evaluation produce un file metrics_<model_tag>.json con dentro:
metadati di esperimento/modello,
dizionario metrics (Acc, BalAcc, Macro-F1, Macro AUROC OvR, Macro AUPRC, e opzionalmente ECE),
class_names usate.
Da questi JSON abbiamo costruito la tabella comparativa “6 modelli × metriche” (formato paper-ready). La struttura consigliata (senza numeri qui, perché dipendono dai run) è:
Model
Acc
BalAcc
Macro-F1
Macro-AUROC(OvR)
Macro-AUPRC
Min tumor recall

I valori provengono da:
metrics_<model_tag>.json (metriche globali)
report_per_class.json (recall per classe → min tumor recall)

F.3 Confusion matrix (best model) e lettura degli errori ONCO/CHROMO
Per ciascun modello valutato generiamo:
confusion matrix patch-level via confusion_matrix(y_true, y_pred) e salvataggio PNG cm_<model_tag>.png .
report per classe (precision/recall/f1/support) in report_per_class.json .
Nel progetto, questa visualizzazione è stata usata per:
quantificare la confusione tra ONCO e CHROMO (celle fuori diagonale tra queste due classi),
separare errori “tumore→non tumore” e “non tumore→tumore”, utili a capire se il modello sta usando shortcut (es. background/stain/artefatti) più che morfologia.

F.4 Patient-level aggregation: majority voting / prob-sum e motivazione
Poiché in clinica la decisione è per paziente (o almeno per WSI/ROI), abbiamo introdotto un passaggio di aggregazione a posteriori basato sulle predizioni patch-level.
Regole esplicite (riproducibili):
Escludiamo sempre NOT_TUMOR e usiamo tutte le patch disponibili .
Ground truth paziente:
se esiste almeno una patch tumorale, la label paziente è la moda delle sole label tumorali;
se tutte le patch sono NOT_TUMOR, il paziente viene marcato non_tumor_only e non entra nelle metriche tumor-only .
Predizione paziente (due metodi):
prob_sum (default): sommiamo le probabilità softmax su tutte le patch, azzeriamo NOT_TUMOR e scegliamo l’argmax tra le classi tumorali ;
vote: majority vote delle predizioni patch ignorando NOT_TUMOR, con fallback se non ci sono voti .
Output patient-level (per run):
patient_predictions.csv con conteggi patch, confidenza e supporto per classe (se prob_sum)
metrics_patient.json e confusion matrix tumor-only cm_patient_<model>.png
file riassuntivo globale runs_summary_patient.csv aggiornato automaticamente, utile per confronti rapidi tra run/ablation
Le metriche patient-level vengono calcolate solo sui pazienti tumor-evaluable, includendo accuracy, balanced_accuracy, macro_f1 e (per prob_sum) macro_auc_ovr, macro_auprc, top2_accuracy .

F.5 ROC/PR curve: cosa abbiamo salvato e come ottenere le curve
Nella pipeline automatica abbiamo:
calcolato Macro-AUROC(OvR) e Macro-AUPRC e salvato i risultati nel JSON ;
salvato logits_test.npy .
Le curve ROC/PR non sono esportate come PNG in automatico in questa traccia, ma sono ricostruibili in modo deterministico dai logits salvati:
softmax(logits_test.npy) → probabilità per classe,
label_binarize(y_true) → ground truth OvR,
plotting ROC/PR per classe + macro-average (confronto best SSL vs baseline).

4.1 Entry-point e orchestrazione (HPC)
Entry-point SLURM “one-shot”: run_all_explainability.sbatch alloca GPU/CPU/RAM e lancia lo script orchestratore .
Parametri principali (override via env): EXP_ROOT, MODEL_NAME, BACKBONE_NAME, flag ONLY_SPATIAL/ONLY_CONCEPT, e opzionale aggregazione VLM con soglia MIN_VLM_CONF .
Script orchestratore: run_all_explainability.sh:
Scansiona le ablation exp_* sotto EXP_ROOT.
Esegue il modulo python3 -m explainability.run_explainability (per generare config e sottomettere job) .
Opzionalmente, può aggregare output VLM (directory JSON) in un CSV candidato “pulito” per la fase concept .
Orchestratore Python: explainability/run_explainability.py:
Per ogni ablation:
Compila config_xai.yaml e config_concept.yaml inserendo:
outputs_root (root ablation),
i checkpoint ssl_backbone_ckpt e ssl_head_ckpt,
il path alla evaluation (eval_run_dir) .
Sottomette i job spatial/xai_generate.sbatch e concept/xai_concept.sbatch .
Questa scelta garantisce che XAI sia sempre allineata alla run di evaluation corretta (stessi ckpt e stessi artefatti) senza intervento manuale.

4.2 Prerequisito: alignment con evaluation (perché è cruciale)
La fase di evaluation salva:
predictions.csv arricchito con wds_key e metadati per poter recuperare esattamente la patch nel WebDataset e riallinearne predizione/logit/XAI .
logits_test.npy (e opzionalmente embeddings) come input per selezione e diagnostica XAI .
Questo vincolo è fondamentale in pathology: senza chiave stabile (qui wds_key) si rischia di produrre spiegazioni su patch diverse da quelle effettivamente valutate.

4.3 Spatial XAI (attention rollout + fallback) — “Dove guarda il modello?”
Script: spatial/xai_generate.py
Obiettivo: generare spiegazioni spaziali su un sottoinsieme informativo di patch del test set, selezionate da predictions.csv .
(a) Selezione patch per audit (guidata da outcome)
Nel config spatial/config_xai.yaml abbiamo definito una selezione per classe:
topk_tp, topk_fp, topk_fn, lowconf_topk
con un vincolo di copertura min_per_class .
Questo produce un set bilanciato di esempi:
TP: per vedere pattern corretti e “cosa usa” quando funziona.
FP/FN: per auditare errori (es. ONCO↔CHROMO).
Low confidence: per capire se l’incertezza è legata a tessuto ambiguo, artefatti, background, ecc.
(b) Metodi di saliency attivati
Nel config abilitiamo più metodi in parallelo:
attn_rollout (primario per ViT),
gradcam,
ig (Integrated Gradients) .
Parametri chiave di attention rollout:
head_fusion: "mean"
discard_ratio: 0.9
.
Il codice implementa:
IG con n_steps configurabile e baseline “black” .
GradCAM su layer target definito da config (nel nostro caso backbone.model.norm) .
Attention rollout per ViT tramite “monkey patching” dei blocchi Attention (quando il backbone è ViT/timm) .
(c) Output generati (riproducibili e indicizzati)
Per ogni patch selezionata, lo script salva overlay PNG (es. attn_rollout.png, gradcam.png, ig.png) e scrive una riga su index.csvcon:
wds_key,
true/pred,
conf,
metodi usati e path delle immagini,
“selection_reason” .
(d) Risorse HPC
Job submission: spatial/xai_generate.sbatch (GPU A40, 1 GPU, 8 CPU, 48GB RAM, 24h) .
Motivo: anche se la batch-size è 1 (per generare saliency per-patch), alcuni metodi (IG) sono computazionalmente costosi.

4.4 Concept XAI (PLIP/VLM + concept bank) — “Che concetti istologici emergono?”
Script: concept/xai_concept.py
Obiettivo: produrre una spiegazione semantica sotto forma di ranking di concetti istologici per patch, riusando lo stesso caricatore modello/ckpt della evaluation e (opzionalmente) una lista di candidati derivata da VLM .
(a) Selezione patch (coerente con spatial)
La selezione per classe è speculare alla spatial:
topk_tp/fp/fn, lowconf_topk, min_per_class .
(b) Concept bank e metadati (ontology-driven)
La “base concettuale” viene letta da un CSV (meta_csv) che contiene:
lista concetti (nel trace: 18),
group,
class_label (utile per distinguere concetti più diagnostici vs potenziali confound) .
Il concept bank può essere costruito da ontologia YAML + dev WebDataset con ontology/build_concept_bank.py (estrae embedding immagini per concetto e salva un CSV) .
(c) Similarità e ranking
Nel config scegliamo:
tipo similarità cosine (sui feature embedding) .
topk concetti da salvare per patch (es. 10) .
Output per patch:
concept_scores.json con lista concept_scores (nome, score, group, class_label),
più metadati di audit: true_label, pred_label, conf, e selection_reason .
index.csv con top concetti serializzati e chiave wds_key .
Questo rende possibile, a posteriori, correlare:
errore di classe (es. ONCO predetto CHROMO),
con i concetti “attivati” (es. pattern che il modello associa a CHROMO),
e con il tipo di selezione (FP/FN/low-conf).
(d) Integrazione opzionale con VLM (candidate mining)
Abbiamo previsto una modalità in cui un VLM produce risposte strutturate JSON del tipo:
concept_present (bool),
confidence,
rationale (spiegazione testuale) .
Il prompt è “vincolato” a un ruolo di patologo (“board-certified renal pathologist”) e chiede output JSON stretto ; l’implementazione client usa un system role “expert pathologist assistant” nelle chiamate chat .
Poi uno step di aggregazione costruisce concept_candidates_rcc.csv partendo da:
spatial index.csv (quali patch analizzare),
directory vqa_json,
filtro su confidenza minima (min_confidence) .
Questa parte serve a “restringere” l’analisi concettuale a concetti robusti e ad alta confidenza (utile quando vuoi auditare pochi concetti ad alto segnale).
(e) Risorse HPC
Job submission: concept/xai_concept.sbatch (gpu_a40, 1 GPU, 8 CPU, 48GB RAM, 24h) .

4.5 Chiusura tecnica: cosa funziona, cosa no, e limiti (senza “vendere fumo”)
Cosa funziona (nel nostro setup):
La selezione TP/FP/FN/low-conf rende l’audit mirato e riproducibile per classe .
L’attention rollout è naturale per backbone ViT e produce overlay immediatamente interpretabili “a colpo d’occhio” .
Il concept XAI produce un output strutturato (JSON + index) che collega predizione ↔ concetti ↔ motivazione di selezione .
Limiti intrinseci (da dichiarare in modo onesto in presentazione/paper):
Le mappe di attenzione/saliency non sono prova causale: indicano correlazioni del modello, non “verità clinica”.
Concept bank: copre solo i concetti presenti in ontologia/dataset; concetti mancanti → spiegazioni incomplete .
VLM: anche con JSON strict e soglia di confidenza, resta un componente potenzialmente non deterministico e dipendente dal dominio.
Next steps (futuro lavoro, non ancora “fatto”):
Validazione qualitativa con patologi: agreement su regioni salienti e top-concepts.
Aggregazione patient-level delle spiegazioni (consistenza concetti su più patch/ROI).
Miglioramento ontology + shortlist “diagnostic vs confounds” con validazione quantitativa.
