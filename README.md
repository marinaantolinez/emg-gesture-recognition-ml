# Hand gesture recognition with surface EMG signals

> **EN:** Classical machine learning pipeline to recognize six hand and wrist gestures from 8-channel surface EMG (MYO armband), evaluated on **subjects never seen during training**. SVM (RBF) reaches a macro F1 of 0.83 on 8 held-out subjects (baseline: 0.05), with inference in ~3 ms per window.

Applied project of the subject **Computational Learning (Introduction to Machine Learning with Python)**, MLDS program, National University of Colombia.

**Authors:** Marina Antolínez Cabrero · Jorge Esau Tierradentro Cruz

🎥 [Explanatory video (YouTube)](https://www.youtube.com/watch?v=eVvN9gcQzFg)

📄 [Presentation (PDF)](presentation/Presentation_EMG_gestures.pdf)

📓[[Final Notebook](final-project/Final_Project_EMG_gestures.ipynb)

- - -

## The question

A prosthesis or exoskeleton only responds well if it knows what gesture the user intends to make. Surface electromyography (EMG) allows this to be inferred, but the signal is noisy and varies from person to person.

**Can a classic machine learning model recognize six gestures in people it never saw during training?**

Gestures: rest, closed fist, wrist flexion, wrist extension, radial deviation and ulnar deviation.

It is a **feasibility study** with an *offline* rating, not a final device.

## Data

[EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures) (UCI Machine Learning Repository, ID 481, CC BY 4.0 license).

- 36 subjects with an 8-channel MYO wristband
- 72 records and more than 4.2 million reads
- The "extended palm" class is only performed by 2 subjects, so it is excluded from the main model and used as an "unknown gesture"

The dataset is **not included** in the repository (≈ 250 MB). The notebook automatically downloads it from UCI the first time it runs.

## Method

1. Cleaning the records (1 null removed; 89% of repeated readings collapse, revealing an effective frequency of ≈ 125 Hz).
2. Segmentation in 200 ms windows with 50% overlap (10,090 windows × 25 readings × 8 channels).
3. Extraction of 40 time domain features: RMS, MAV, WL, ZC and SSC per channel.
4. Comparison of a *baseline* (most frequent class) and five models: logistic regression, kNN, SVM (RBF), Random Forest and HistGradientBoosting.
5. Hyperparameter search with cross-validation grouped by subject (`StratifiedGroupKFold`).
6. Single evaluation on 8 reserved subjects.
7. Generalization analysis: Leave-One-Subject-Out, user calibration, unknown
gesture, and latency.

**All validation is done per subject.** Overlapping windows for the same person are
almost identical, and mixing them between training and testing would inflate the results.

## Results

| Indicator | Result |
| :--- | :--- |
| Baseline (most frequent class) | F1-macro 0.049 · accuracy 0.174 |
| SVM RBF ($C = 1, \gamma = 0.03$), cross-validation per subject | F1-macro 0.821 ± 0.076 |
| **8 never-before-seen test subjects** | **F1-macro 0.829 · accuracy 0.829 · AUC macro 0.955** |
| Leave-One-Subject-Out (36 subjects) | Global macro F1 0.830; per subject from 0.32 to 0.99 |
| Normalization per subject (brief calibration) | Test macro F1 0.829 → 0.909 |
| Inference latency | 3 ms per window, with ≈ 100 ms available |

### Main findings

- **Non-linear models outperform linear models**, but SVM, Random Forest and HistGradientBoosting are practically tied.
- **The limit is set by the variability between people, not the amount of data:** learning curves flatten and performance per subject varies enormously.
- **A brief calibration per user is the most promising improvement**, although it was only evaluated with 8 subjects.
- **The model is viable in real time:** inference costs ~3 ms versus ~100 ms
available between windows.
- The errors are concentrated among gestures that share musculature, especially wrist flexion and ulnar deviation.

### Limitations

- Only 36 subjects, and 8 in the test set, so small differences between models are not conclusive.
- Closed set classifier: always chooses a gesture, even if it is unknown (82% of the "extended palm" was classified as ulnar deviation).
- *Offline* evaluation: not tested with a real device nor was the latency of the entire system measured.

### Next steps

- Evaluate the normalization per subject with LOSO on the 36 subjects.
- Add an "unknown" class or a confidence threshold.
- Test 1D networks over the windowed signal.
- Estimate confidence intervals by resampling subjects.

## Repository structure

The project was developed in phases throughout the course. Each folder contains the notebook from its phase, and the final notebook integrates all the work.

```
├── phase-1-understanding-the-business/     # Project framework, scope, risk plan, data loading (SPANISH)
├── phase-2-understanding-the-data/         # Data quality and relationship between channels (SPANISH)
├── phase-3-preparation-of-the-data/        # Cleaning and segmentation in windows (SPANISH)
├── phase-4-modeling/                       # Features, models, hyperparameters, and evaluation (SPANISH) 
├── final-project/ (ENGLISH)
├── presentation/                           # Project presentation (PDF) (SPANISH) 
├── requirements.txt
└── README.md
```

> If you only want to see the result, start with [`final-project/`](final-project/). Notebooks retain the outputs of the original execution, so GitHub displays them with tables and graphs without needing to run anything.

## How to run it

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook proyecto-final/Proyecto_final_EMG_gestos.ipynb
```

The first load cell downloads and unzips the dataset into the folder `emg_gestures/`
It requires an internet connection and about 250 MB of disk space (about 650 MB in
memory when loading all records).

- **Professor:** [Fabio Augusto Gonzalez](https://dis.unal.edu.co/~fgonza/)
- **Teaching assistant:** [Rosa Alejandra Superlano Esquibel](https://www.linkedin.com/in/alejandrasuperlano/)
- **Virtualization Coordinator:** [Edder Hernández Forero](https://www.linkedin.com/in/edder-hernandez-forero-28aa8b207/)
- **Dataset:** *EMG Data for Gestures*, UCI Machine Learning Repository (CC BY 4.0).

**National University of Colombia** · Faculty of Engineering

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
