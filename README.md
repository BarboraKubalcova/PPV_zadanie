# Pokročilé počítačocé videnie - zadanie

Úlohou tohto zadania je prepojiť LLM s metódami počítačového videnia. Použitý vizuálny model: 
[CheXangent-2-3b](https://huggingface.co/StanfordAIMI/CheXagent-2-3b). Použitý dataset: 
[Slake](https://www.med-vqa.com/slake/).

## Dataset
Dataset **SLAKE** ( A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual 
Question Answering) je dataset určený na dotazovanie špecificky medicínskych otázok týkajúcich sa 
x-ray snímok hrudíka. Ku každému obrázku je priradená otázka s prislúchajúcou odpoveďou (forma áno/nie alebo krátky text). 

Pred použitím kódu je potrebné dataset stiahnuť
[tu](https://link.jscdn.cn/googledrive/aHR0cHM6Ly9kcml2ZS5nb29nbGUuY29tL2ZpbGUvZC8xRVowV3BPNVo2QkpVcUMzaVBCUUpKUzFJTldTTXNoN1Uvdmlldz91c3A9c2hhcmluZw==.zip).

Štruktúra priečinku s datasetom je nasledovná
```
./dataset/
└── Slake/
    ├── __MACOSX/
    └── Slake1.0/
        ├── imgs/
        ├── KG/
        ├── test.json
        ├── train.json
        └── validate.json
```

## Model
Klinický model trénovaný na odpovedanie otázok ohľadom x-ray snímkov hrudníka a iných častí ľudského tela. 
