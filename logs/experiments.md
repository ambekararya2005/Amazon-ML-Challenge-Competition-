# Experiment log

## 2026-09-25 — Project setup
- Created layout: `code/business_entity_resolution/{src,README.md,requirements.txt}`, `output/`, `cache/`, `logs/`.
- Added root `CLAUDE.md` with standing rules; `logs/submissions.md` template; `.gitignore` excludes dataset, cache, output, venv.
- Venv `.venv/` on Python 3.11.9: pandas 3.0.6, pyarrow 25.0.1, numpy 2.4.6, scikit-learn 1.9.1, scipy 1.17.1,
  rapidfuzz 3.14.6, sparse_dot_topn 1.2.0, lightgbm 4.7.0, anyascii 0.3.3, tqdm 4.70.1, psutil 7.2.2. All imports + smoke tests pass.
- No modelling code yet. Key numbers: n/a.
