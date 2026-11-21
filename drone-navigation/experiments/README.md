# Experiments: new datasets, stress test, missing frames

The task data has no ground truth, so these scripts check the pipeline in three extra ways.
They use the unchanged `drone_nav` code. Results go to `outputs/experiments/`.

```bash
python experiments/synthetic_flights.py   # ~35 s: 15 generated flights with ground truth
python experiments/stress_test.py         # ~60 s: 3,100 degraded frames, find the breaking point
python experiments/drop_frames.py         # ~15 s: remove frames from the task data
```

## 1. Generated flights with ground truth (`synthetic_flights.py`)

5 random smooth flights × 50 frames over the task map. Each frame has a known centre, any heading
from 0° to 360°, and altitude changing by up to ±8 %. Frames are degraded at three levels:

- **realistic**: ±20 % exposure, ±10 % colour cast, noise σ 6, blur σ 1, JPEG 75
- **harsh**: ±40 % exposure, ±20 % colour cast, noise σ 15, blur σ 2, JPEG 40, clouds up to 25 % of the frame

| Level | Localised | Centre error median / max | Heading error max | Scale error median |
|---|---|---|---|---|
| clean | 250/250 | 0.39 / 0.55 px | 0.02° | 0.006 % |
| realistic | 250/250 | 0.39 / 0.55 px | 0.02° | 0.007 % |
| harsh | 250/250 | 0.39 / 0.56 px | 0.05° | 0.014 % |

The median number of inliers drops from 362 (clean) to 207 (harsh), with a minimum of 54. That is
still well above the threshold of 20.

## 2. Breaking point (`stress_test.py`)

One degradation at a time, 2 flights × 50 frames per setting. A frame counts as correct if its centre
is within 10 px of the truth.

| Degradation | Still 100 % correct up to | Fails at |
|---|---|---|
| Sensor noise σ | 80 (the maximum tested) | – |
| Gaussian blur σ | 4 px (99 % at 6 px) | 8 px: 84 %, 10 px: 44 % |
| Cloud cover | 20 % (99 % at 40 %) | 60 %: 91 %, 75 %: 46 % |
| Contrast (haze) | ×0.5 | ×0.3: 86 %, ×0.2: 1 % |
| Contrast + 1–99 % stretch | ×0.05 (the maximum tested) | – |

- **No wrong position was accepted in 3,100 stressed frames.** When a frame is too damaged, the
  inlier and scale checks reject it, so it gets interpolated instead of corrupting the route.
- **Haze can be fixed cheaply.** Low contrast falls below SIFT's contrast threshold. A per-frame
  1–99 percentile contrast stretch brings back 100 % even at 5 % contrast. This is a proposed change
  and is not in the pipeline yet.

## 3. Missing frames (`drop_frames.py`)

The reference is the full run on the 50 task frames. Frames are removed and their positions are
interpolated over the frame number. Each frame is localised independently, so removing frames never
moves the others: there is no drift.

Random removal (first and last frames kept, 50 trials each). Errors are for the removed frames; one
frame footprint is 162 px.

| Removed | Linear: median / max | Cubic spline: median / max |
|---|---|---|
| 10 % | 3.1 / 8.1 px | 0.1 / 0.3 px |
| 30 % | 3.8 / 25.1 px | 0.1 / 0.7 px |
| 50 % | 6.5 / 78.7 px | 0.2 / 5.8 px |
| 60 % | 9.4 / 118.9 px | 0.2 / 7.4 px |

Worst error for one contiguous gap, over every position on the route:

| Gap | 1 | 3 | 5 | 8 | 10 | 15 frames |
|---|---|---|---|---|---|---|
| Linear | 4.2 | 16.1 | 36.0 | 78.7 | 116.6 | 237.2 px |
| Cubic | 0.3 | 0.5 | 1.7 | 8.3 | 10.8 | 42.3 px |

**End-to-end through `main.py`:** 14 of the 50 frames (28 %, including a 6-frame gap) were replaced
with black, pure-noise or featureless images. All 14 were rejected and marked `interpolated`, and no
good frame was rejected. The other 36 frames moved by at most 0.006 px. The interpolated positions
are 3.3 px off at the median and 7.5 px at most.

**Takeaway:** linear interpolation is fine for gaps of up to ~3 frames. A cubic spline over the
frame number is 10–30× more accurate on this route and stays under 11 px for gaps of up to 10 frames.
That makes it the next improvement to make.

## Caveat

The generated frames are cut from the same map, so they test geometry, camera effects and missing
data, not a change of season or sensor. Public UAV-to-satellite datasets would be the next step for
testing that.
