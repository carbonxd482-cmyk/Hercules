# HERCULES

AI-assisted astronomical triage for NASA's Kepler dataset. It doesn't find planets. It looks at thousands of observations and tells researchers which ones are worth a closer look first.

## The problem

Kepler-era surveys produce way more candidate signals than anyone can manually review with equal attention. Somewhere in a pile of 9,564 observations, a handful are genuinely worth investigating and most aren't. HERCULES is a triage layer for that pile — not a replacement for the astronomer, just a way to put the most promising signals at the front of the queue.

## How it works


OBSERVE  →  CLASSIFY  →  PRIORITIZE  →  INVESTIGATE


1. **Observe** — raw Kepler observations and transit measurements go in.
2. **Classify** — a trained classifier estimates whether a signal looks like a candidate, a confirmed planet, or a false positive.
3. **Prioritize** — candidate probability gets combined with signal quality, transit coverage, and data completeness into a single Scientific Priority Score.
4. **Investigate** — the highest-scoring targets get surfaced first, for a human to actually look at.

That score is the whole product. It is **not** a confirmation probability, and the app says so everywhere it matters — a high score means "look at this," not "this is a planet."

## Model history

The current classifier (V3-C) didn't come out of nowhere. A few iterations got there:

- **V1** — baseline Random Forest, ~79.5% accuracy.
- **V2** — ran a small tournament (Random Forest, Extra Trees, HistGradientBoosting, Logistic Regression) on 21 hand-picked scientific features. HistGradientBoosting won: 80.66% accuracy, 77.22% macro F1.
- **V3** — tested whether extra engineered features actually helped. Stellar density alone didn't. Measurement-uncertainty features bumped accuracy slightly but hurt candidate recall. The winner (V3-C, "scientific signals," 28 features) had the best macro F1 and became the production classifier.
- **V4 / V4.2** — took the V3-C classifier, ran it against the full dataset, and layered a separate prioritization engine on top that combines candidate probability with signal quality, transit coverage, and data completeness.

Every one of the 21–28 features going into the model was checked against the raw catalog columns first — `koi_disposition`, `koi_pdisposition`, and the `koi_fpflag_*` fields are all present in the raw NASA data but were deliberately kept out of training, because they're effectively the answer key.

## Where it's honestly weak

Candidate recall sits around 55–57% in testing. That's not hidden anywhere — it means HERCULES misses a real fraction of actual candidates, not just false positives. Overall accuracy looks fine (~80%) mostly because false positives are the majority class in this dataset, which is exactly the kind of number that looks better than it is if you don't dig into it. Anyone using this for actual research should treat the priority queue as a starting point, not a filter that's safe to trust blindly.

## The app

Five sections, all pulling from the same ranking output:

- **Command Center** — the pipeline funnel and the current top-ranked candidate.
- **Target Explorer** — search any observation and see its full scientific dossier: classification, priority score, signal components, raw transit/orbital/planetary/stellar measurements.
- **Candidate Hunter** — filter the whole candidate set by priority, probability, and signal quality; export the full filtered set to CSV.
- **Scientific Analytics** — classification and priority distributions, signal-vs-probability relationships.
- **About** — what the system actually is and isn't claiming.

Built on Streamlit. Every page reads from one ranking CSV (`outputs/hercules_v4_2_rankings.csv`) so the numbers can't disagree with each other across screens.

## Running it

```
cd Hercules
streamlit run src/app.py
```

Needs `outputs/hercules_v4_2_rankings.csv` to exist — that's the output of the V4.2 scoring pipeline, not something the app generates itself. A `.streamlit/config.toml` sets the color theme; without it Streamlit falls back to its own default accent color instead of the app's.

## Stack

Python, pandas, Streamlit, matplotlib for the charts. No JS framework, no database — one CSV in, one dashboard out.

## What's next

The obvious follow-up work: getting candidate recall up without wrecking precision, auditing whether the priority score formula should be gated more strictly by predicted class (a false positive with strong signal characteristics can currently still score high), and eventually moving from catalog-level features to actual light-curve analysis instead of pre-computed transit parameters.

---

*Data: NASA Exoplanet Archive, Kepler Objects of Interest catalog.*

Live demo: temporarily unavailable due to dataset hosting limitations.
Local demo: run the V4.2 scientific priority engine with the included pipeline.
