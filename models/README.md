# Models

The lightweight demo model is saved as:

```text
models/lightweight_eot_model.json
```

It is trained with:

```bash
python3 src/train_lightweight_classifier.py --data data/ami_turn_samples.csv
```

This JSON model is small enough to keep in the repository so the local browser demo works without installing scikit-learn.

The optional scikit-learn model is written here by:

```bash
python src/train_classifier.py
```

The generated `.joblib` file and `models/metrics.json` are ignored by Git because they can be recreated.
