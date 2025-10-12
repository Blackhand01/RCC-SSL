# RCC_WSIs Dataset — Structure & Mapping Guide

## 1. Overview

Il dataset **RCC_WSIs** è una collezione di **Whole-Slide Images (WSI)** digitali provenienti da campioni istologici di tumori renali, con le rispettive **annotazioni XML** e tabelle Excel di corrispondenza tra *slide* e *pazienti*.
Serve per sviluppare sistemi di segmentazione, classificazione o correlazione tra regioni istologiche e diagnosi.

---

## 2. Directory Structure

| Folder               | Content Type                        | Description                                                           |
| -------------------- | ----------------------------------- | --------------------------------------------------------------------- |
| `Annotations_chromo e CHROMO` | `.svs`, `.tif`, `.xlsx`             | WSI cromofobi + file `CHROMO_patients_correspondence.xlsx`            |
| `Annotations_onco e ONCHO`   | `.svs`, `.tif`, `.xlsx`             | WSI oncocitoma + file `ONCO_patients_correspondence.xlsx`             |
| `ccRCC e pre/ccRCC`  | `.svs`, `.tif`                      | WSI del carcinoma a cellule chiare                                    |
| `pRCC e pre/pRCC`    | `.scn`, `.xml`, `.xlsx`             | WSI del carcinoma papillare (pRCC)                                    |
| `pRCC/pRCC_xml e pre/pRCC/pRCC_xml`| `.xml`                | Annotazioni strutturate per le slide pRCC                             |
| `ccRCC/ccRCC_xml e pre/ccRCC/ccRCC_xml`          | `.xml`  | Annotazioni strutturate per le slide ccRCC                            |

Quindi abbiamo 4 classi tumorali: **CHROMO**, **ONCO**, **ccRCC** e **pRCC**.
Solo negli xml abbiamo le annotazioni di aree non tumorali.


---

## 3. Excel Correspondence Tables

### Common Structure

Tutti gli Excel hanno lo stesso schema:

| ID (slide range) | Patient_ID |
| ---------------- | ---------- |
| 1-3              | HPXXXXXXX  |

#### CHROMO_patients_correspondence.xlsx

| ID    | Patient_ID |
| ----- | ---------- |
| 1-2   | HP17008718 |
| 3-7   | HP20002300 |
| 8     | HP18014084 |
| 9-11  | HP19012316 |
| 12-13 | HP20.2506  |

#### ONCO_patients_correspondence.xlsx

| ID    | Patient_ID |
| ----- | ---------- |
| 1-3   | HP20002450 |
| 4-7   | HP20001530 |
| 8-13  | HP18005453 |
| 14-21 | HP20.5602  |
| 22-25 | HP18090209 |


## 4. Annotation Files (`*.xml`)

Le annotazioni XML, prodotte con software come Aperio o QuPath, descrivono **regioni d’interesse (ROI)** su ciascuna slide.
Struttura tipica:

```xml
<Annotations>
  <Annotation Id="1" Name="tumor" ReadOnly="0">
    <Regions>
      <Region Id="1" Type="0" Text="Tumor region" GeoShape="Points">
        <Vertices>
          <Vertex X="37279.31" Y="4662.18"/>
          <Vertex X="37319.55" Y="4588.89"/>
          ...
        </Vertices>
      </Region>
    </Regions>
  </Annotation>
  <Annotation Id="2" Name="non_tumor">
    <Regions>...</Regions>
  </Annotation>
</Annotations>
```

---