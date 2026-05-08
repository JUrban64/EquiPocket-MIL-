# P2Rank Embedding Pipeline & MIL Classification

Toto je end-to-end architektura pro extrakci kapes pomocí `p2rank`, trénování E(3)-ekvivariantních GNN modelů a následnou klasifikaci pomocí Attention-based Multiple Instance Learning (MIL).

## Architektura

Architektura se skládá ze 3 hlavních pilířů:
1. **Data Extractor & Graph Builder:** Z PDB souborů kapes vyextrahuje sekvence a 3D souřadnice. Ty jsou zkonvertovány na `PyTorch Geometric` grafy a kofaktorové třídy jsou automaticky olabelovány na bag-level (např. celý protein váže NAD).
2. **EGNN (E(3)-Equivariant Graph Neural Network):** Grafová síť zpracovává 3D koordináty atomů a ESM embeddingy. Je invariantní vůči rotacím a translacím, čímž vznikají vysoce kvalitní, na rotaci nezávislé embeddingy kapes.
3. **Attention-based MIL Classifier:** Namísto ručního určování, která p2rank kapsa je ta pravá, model dostane všechny kapsy daného proteinu najednou (tzv. "Bag"). Pomocí Attention mechanismu si model *sám* odvodí, která kapsa je pro daný enzym klíčová, přidělí jí vysokou váhu a provede klasifikaci na cílový kofaktor (NAD, FAD, ATP...).

---

## Jak pipeline použít

### 1. Extrakce Ground Truth a tvorba grafů
Nejprve musíme vyrobit grafy z krystalografických struktur, na kterých budeme trénovat samotné EGNN.
Tento příkaz automaticky vytáhne správné ligandy, vytvoří `binding_sites_by_protein.json` a rozseká je do tenzorových grafů:
```bash
python build_p2rank_dataset.py
python binding_site_graph.py \
    --input-json binding_sites_by_protein.json \
    --output-dir gt_graph_batches
```

### 2. Trénink EGNN (Reprezentační učení)
Nyní naučíme EGNN rozpoznávat správný tvar kapes nad krystalografickými daty (GT grafy). Model se uloží jako `gnn_model_e3.pt`.
```bash
python train_E3.py \
    --epochs 10 \
    --hidden-dim 64 \
    --num-heads 4 \
    --num-classes 6
```
*(Tréninkové logy se automaticky ukládají do `egnn_training.log`)*

### 3. Příprava P2Rank dat (Neoznačené kapsy)
Přejdeme k p2rank kapsám. Vytvoříme z nich grafy (přiřadí se globální labely z nadřazených složek):
```bash
python binding_site_graph.py \
    --input-json p2rank_pockets_dataset.json \
    --output-dir p2rank_graph_batches
```

### 4. Generování Embeddingů
Přeženeme p2rank grafy přes natrénované EGNN a uložíme výsledné tenzory:
```bash
python generate_embeddings_E3.py \
    --model-path gnn_model_e3.pt \
    --graph-manifest p2rank_graph_batches/manifest.json \
    --output-path p2rank_egnn_embeddings.pt
```

### 5. Trénink MIL Klasifikátoru
Nakonec spustíme Attention-based Multiple Instance Learning síť, která se podívá na vygenerované embeddingy a naučí se sama nacházet tu správnou kapsu pro každý protein.
```bash
python train_mil_classifier.py \
    --embeddings-path p2rank_egnn_embeddings.pt \
    --epochs 50
```
*(Tréninkové logy včetně Attention vah vybraných kapes se ukládají do `mil_training.log`)*
