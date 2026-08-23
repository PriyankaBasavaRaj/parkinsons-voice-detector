# Voice & Parkinson's, a signal-processing demo

A fully client-side web app that estimates Parkinson's-related vocal tremor
from a short recorded voice sample, trained on the Oxford Parkinson's Disease
Detection Dataset. Everything, audio capture, feature extraction, and model
inference, runs in the browser. No server, no upload, nothing leaves the
page.

**[Try the live demo →](#)** *(link once deployed, see Deployment below)*

---

## Why this dataset is a trap (and how this project avoids it)

The [UCI Parkinson's dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)
has 195 voice recordings, but only from **31 people**, each contributing
~6 recordings. Nearly every public tutorial on this dataset splits rows
randomly into train/test, which means recordings from the *same person*
end up on both sides of the split. The model doesn't learn "what does
Parkinsonian voice sound like", it partly learns "what does this specific
person's voice sound like," and reports inflated accuracy (often 95%+)
that has nothing to do with real generalization.

This project splits **by patient** (`GroupKFold`, grouped by subject ID),
so no patient's recordings ever appear in both a training fold and a
validation fold. The honest result is lower, and that's the point:

| Split method | Accuracy | What it actually measures |
|---|---|---|
| Random row split (common in tutorials) | ~95%+ | Partly memorized patient identity |
| Patient-grouped split (this project) | ~74% (full features) | Actual generalization to new people |

## Two models, one honest tradeoff

**`models/model.pkl`**, trained on all 22 dataset features (jitter/shimmer
variants, HNR, and nonlinear dynamics measures RPDE/DFA/D2/spread1/spread2/PPE).
~74% accuracy, 0.76 AUC under grouped CV. This is the "best you can do with
this dataset" model, but RPDE/DFA/D2 require Praat-grade signal processing
(`parselmouth`) that isn't practical to reproduce in browser JavaScript.

**`models/model_browser.pkl`**, retrained on only 6 features that a
browser can realistically compute from raw audio via the Web Audio API:
mean/max/min pitch, jitter %, shimmer, and an approximated HNR. ~69%
accuracy, 0.66–0.71 AUC. **This is the model the live app actually uses.**

Going fully static (no backend, deployable on GitHub Pages for free,
zero cold-starts) cost about 5 points of accuracy. That tradeoff, and
being explicit about it, is more interesting than chasing a better
number, so it's documented here rather than hidden.

## What's approximated in the browser

The in-browser feature extraction (`docs/feature-extraction.js`) uses a
simple autocorrelation-based pitch tracker, not a full clinical DSP suite:

- **Pitch (F0)**: windowed autocorrelation, 40ms frames, 50% overlap
- **Jitter**: mean absolute cycle-to-cycle F0 difference, as % of mean F0
- **Shimmer**: mean absolute cycle-to-cycle amplitude difference
- **HNR**: *approximated* from frame-to-frame pitch stability, not a
  true spectral harmonic-to-noise ratio (which needs FFT-based harmonic
  analysis). Calibrated to sit in a plausible range, but should be read
  as a rough proxy, not a lab-grade measurement.

## Limitations & next steps

- **Sample size.** 31 people is not enough to make any clinical claim.
  With more data, this easily extends to a much larger, more diverse
  training set.
- **Feature engineering vs. raw audio.** A natural next step is skipping
  hand-engineered features entirely and training a CNN directly on
  spectrograms, likely more robust, but needs far more data and a real
  backend (no longer browser-only).
- **HNR proxy.** Swapping in a proper FFT-based harmonic analysis in JS
  (rather than the pitch-stability approximation used now) would tighten
  the browser model's accuracy gap versus the full-feature model.

## This is not a medical device

This is a portfolio and educational project demonstrating grouped
cross-validation, feature engineering tradeoffs, and shipping an ML model
client-side. It is not validated for, and must not be used for, medical
diagnosis. If you have concerns about Parkinson's disease, please consult
a doctor.

---

## Project structure

```
parkinsons-voice-detector/
├── data/
│   └── parkinsons.data          # Oxford Parkinson's dataset (CC BY 4.0)
├── src/
│   ├── data.py                  # loading + patient-grouped train/test split
│   ├── train.py                 # full 22-feature model (Python-only use)
│   └── train_browser_model.py   # reduced 6-feature model used by the app
├── models/
│   ├── model.pkl                 # full-feature sklearn model
│   ├── model_browser.pkl         # reduced-feature sklearn model
│   ├── model.js                  # model_browser.pkl exported to JS (m2cgen)
│   └── scaler_config.json        # StandardScaler mean/scale for JS scoring
├── docs/
│   ├── index.html                 # the static web app (GitHub Pages entry point)
│   ├── feature-extraction.js      # browser-side pitch/jitter/shimmer/HNR extraction
│   ├── model.js                   # copy of the exported model used by the app
│   └── scaler_config.json         # copy used by the app
└── requirements.txt
```

## Running the Python pipeline locally

```bash
pip install -r requirements.txt

# Full-feature model (all 22 features, Python-only)
python src/train.py

# Reduced-feature model that the browser app actually uses
python src/train_browser_model.py
```

Both scripts print grouped cross-validation results (accuracy, PD recall/
precision, ROC-AUC, confusion matrix) before saving the final model.

If you retrain `model_browser.pkl`, re-export it to JS and refresh the
scaler config the app uses:

```bash
pip install m2cgen
python -c "
import joblib, m2cgen as m2c
data = joblib.load('models/model_browser.pkl')
code = m2c.export_to_javascript(data['pipeline'].named_steps['clf'])
open('docs/model.js', 'w').write(code)
"
```
(then update `docs/scaler_config.json` with the new scaler's `mean_`/`scale_`)

## Running the app locally

The app is fully static, no build step, no server-side code. Any local
web server works (it must be a server, not `file://`, for `fetch()` to
load `scaler_config.json`):

```bash
cd docs
python -m http.server 8000
# open http://localhost:8000
```

## Deployment (GitHub Pages)

1. Push this repo to GitHub (see commands below).
2. In the repo settings → **Pages** → set source to the `main` branch,
   folder `/docs`.
3. GitHub gives you a URL like `https://<username>.github.io/<repo-name>/`
  , that's the link to put on your portfolio site.

## Dataset citation

Little, M. (2007). Parkinsons [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C59C74. Licensed CC BY 4.0.
