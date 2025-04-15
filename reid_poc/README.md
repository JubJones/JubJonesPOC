```angular2html
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If python 3.12 or distutils error while installing the requirements
```
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
```

calibrate_homography:
```angular2html
python reid_poc/calibrate_homography.py c09
```