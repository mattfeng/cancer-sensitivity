# Predicting the sensitivity of cancer cell types to different drugs

## Dependencies
- `conda install -c conda-forge rdkit`

## Preprocessing

### Compounds metadata
- original source of compound info: https://www.cancerrxgene.org/compounds#t-all

## TODO
- integrate latent embedding of cell line genome (mutations)
  - model should be able to learn how immortalization of a cell line influences biology and therefore drug sensitivities; model may need primary cells for comparison, but overall the idea is to use machine learning to correct for the cell line's imperfect modeling of a primary cell
