# spectron_cv_duality

Test assignment for It-Jim, *Classic CV — Drone navigation* (standard level): reconstructing a
drone's flight route by localising its aerial frames on a satellite map with SIFT + OpenCV.

The project lives in [`drone-navigation/`](drone-navigation/). See its
[README](drone-navigation/README.md) for the approach, how to run it, and results.

```bash
cd drone-navigation
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py --map data/global_map.png --frames data/crops --out outputs
```
