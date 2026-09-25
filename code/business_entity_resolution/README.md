# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline that, for every Source 1 entity, finds all matching Source 2 / Source 3 records and writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

> Status: project scaffold only. Pipeline stages and exact run commands are added as they are built.

## Environment

- Python 3.11.9
- Hardware used: Windows 11, 16 GB RAM, 8 CPU cores, NVIDIA RTX 4050 Laptop GPU

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r code/business_entity_resolution/requirements.txt
```

## Data

Place the organiser data under `student_resource/dataset/{train,test}/` (unchanged). No external data,
APIs or downloaded dictionaries are used.

## Reproduce (end to end)

_To be filled in: data → normalise → blocking → matching → output, with the runtime of each stage._

## Layout

```
src/            all source code
README.md       this file
requirements.txt pinned dependencies
```

Stage outputs are cached as Parquet in `cache/` (pass `--force` to recompute); runtimes and peak memory
are logged to `logs/`.
