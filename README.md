# STEREO and STEREO-Lit

This repository is the code for paper: Bridging Chemical Literature and Molecular Graphs: LLM-Augmented Enantioselectivity Prediction for Iridium-Catalyzed Asymmetric C-H Borylation.

## Environment

```bash
pip install -r requirements.txt
```

## Data

Download the Zenodo data archive and place the CSV files in `release_data/`:

```text
release_data/borylation_training_dataset.csv
release_data/borylation_extrapolation_dataset.csv
```

The extrapolation table is intentionally not stored in this Git repository
because it is larger than GitHub's 100 MB file limit. Obtain it from the
Zenodo record associated with the paper before running extrapolation.

The training table uses `btr_id`; the extrapolation table uses `bvs_id`.

## Train and test

```bash
python train.py --variant stereo --data /path/to/borylation_training_dataset.csv \
  --epochs 20 --batch-size 16 --device cuda \
  --summary-path outputs/training_summary.json \
  --model-save-path outputs/training_model.pt --log-dir outputs/training_run
```

```bash
python train.py --variant stereo-lit --data release_data/borylation_training_dataset.csv \
  --epochs 20 --batch-size 16 --seed 42 --split-seed 42 --device cuda \
  --summary-path outputs/stereo_lit_summary.json \
  --model-save-path outputs/stereo_lit_model.pt --log-dir outputs/stereo_lit_run
```

## Extrapolate

```bash
PYTHONPATH=src python src/predict_extrapolation.py \
  --summary outputs/training_summary.json --checkpoint outputs/training_model.pt \
  --training-data release_data/borylation_training_dataset.csv \
  --input release_data/borylation_extrapolation_dataset.csv \
  --output outputs/extrapolation_predictions.csv --device cuda --batch-size 64
```

## License

Code is released under the MIT License. The released tabular data are
licensed under CC BY 4.0; the full extrapolation archive is distributed via
Zenodo. Please cite the associated research article when using this release.
