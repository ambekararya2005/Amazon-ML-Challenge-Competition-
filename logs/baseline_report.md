# Baseline report

Stage 1: top-5 per query by cosine_C + 0.2 x shared_keys_A; validation recall 0.9744 -> 0.96555 (@5).

Best config: mode=o2o, weights (name, addr, num) = (0.3, 0.6, 0.1), threshold = 0.64

| validation | macro F0.5 | mean precision | mean recall | singleton acc | mean pred size |
|---|---|---|---|---|---|
| best (o2o) with one-to-one | 0.9641 | 0.9862 | 0.9329 | 0.9341 | 3.260 |
| best (o2o) with one-to-one | India | 0.9529 | 0.9843 | 0.9122 | 0.9281 | 3.197 |
| best (o2o) with one-to-one | US | 0.9716 | 0.9875 | 0.9467 | 0.9381 | 3.302 |
| best (o2o) without one-to-one | 0.9119 | 0.9244 | 0.9343 | 0.7965 | 3.623 |
| best (o2o) without one-to-one | India | 0.8762 | 0.8925 | 0.9142 | 0.7306 | 3.766 |
| best (o2o) without one-to-one | US | 0.9357 | 0.9457 | 0.9477 | 0.8404 | 3.528 |
| best no-one-to-one config | 0.9488 | 0.9820 | 0.9027 | 0.9285 | 3.170 |
| best no-one-to-one config | India | 0.9318 | 0.9785 | 0.8730 | 0.9212 | 3.082 |
| best no-one-to-one config | US | 0.9601 | 0.9843 | 0.9225 | 0.9334 | 3.229 |

| test country | S1 | % empty predictions | mean predicted per S1 | mean candidates per S1 |
|---|---|---|---|---|
| France | 259,452 | 0.52 | 4.979 | 27.650 |
| India | 809,986 | 0.31 | 5.182 | 29.121 |
| US | 663,106 | 0.04 | 5.288 | 28.781 |
