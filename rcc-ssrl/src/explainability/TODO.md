# Apply Patch to Explainability Pipeline

## Steps to Complete

- [x] 1. Fix Stage 0 in run_full_xai.sh: Change the check for concept bank existence to also verify it's non-empty (more than 1 line).
- [x] 2. Update config_concept.yaml: Add train_dir to webdataset, remove duplicate similarity block.
- [ ] 3. Modify xai_concept.py: Update the data loading to scan both train and test WebDataset directories for features.
- [ ] 4. Remove empty concepts_rcc_v1.csv: Since it's only header, delete it so Stage 0 regenerates.
