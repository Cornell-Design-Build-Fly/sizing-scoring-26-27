# DBF Sizing Script Competition Year 26-27

Python dependencies for this repo are tracked in `requirements.txt`. Install them into a virtual environment before running project code.

## Install the packages

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Update `requirements.txt`

If you install a new package, refresh `requirements.txt` so the repo stays in sync with your environment.

```powershell
python -m pip freeze > requirements.txt
```

If you only want to refresh the file after several installs, run just the last command once you are done adding packages.

The current `requirements.txt` was generated from the packages already installed in this project's virtual environment.
