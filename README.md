# Bangla Text Analysis (Streamlit + PySpark)

A Streamlit app for Bangla text analysis (EDA, ANN/LSH search, clustering) powered by PySpark.

## Repository
**GitHub:** https://github.com/saifkhancse/Bangla-Text-Analysis

---

## Prerequisites
- **Python** 3.8–3.12 (3.10+ recommended)
- **Java JDK** 11 or 17 (required by PySpark)
- **Git** (optional if you prefer ZIP download)

> Windows users: having JDK 17 on PATH works well with PySpark 3.5.x.

---

## 1) Get the code

### Option A — Clone with Git
```bash
git clone https://github.com/saifkhancse/Bangla-Text-Analysis.git
cd Bangla-Text-Analysis
```

### Option B — Download ZIP
- Click **Code ▸ Download ZIP** on the repo page, unzip, then `cd` into the folder.

---

## 2) Create & activate a virtual environment

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -U pip
```

### macOS / Linux (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

---

## 3) Install requirements
Make sure you’re in the project root (where `requirements.txt` lives).
```bash
pip install -r requirements.txt
```

> If you see a Java/PySpark error, ensure Java is installed and `JAVA_HOME` is set:
>
> **Windows (PowerShell)**
> ```powershell
> $env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-17"
> $env:PATH="$env:JAVA_HOME\bin;$env:PATH"
> ```
>
> **macOS**
> ```bash
> export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
> export PATH="$JAVA_HOME/bin:$PATH"
> ```
>
> **Linux (Ubuntu/Debian)**
> ```bash
> sudo apt-get update && sudo apt-get install -y openjdk-17-jdk
> export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
> export PATH="$JAVA_HOME/bin:$PATH"
> ```

---

## 4) Run the app
From the project root:
```bash
streamlit run app.py
```
Then open the URL shown in the terminal (usually http://localhost:8501).

> If the default port is busy:
> ```bash
> streamlit run app.py --server.port 8502
> ```

---

## Notes
- Place your data where the app expects it (see on-screen prompts/paths).  
- First run may build caches/models and take a bit longer.
- Stop the app with **Ctrl+C** in the terminal.

---

## Troubleshooting
- **`JAVA_HOME`/PySpark errors** → set `JAVA_HOME` as shown above.
- **`PATH_NOT_FOUND` for data** → update the data path in the app UI or move data into the suggested folder.
- **Firewall prompt on Windows** → allow local network access for Streamlit.

---

## License
See `LICENSE` (if present) or the repository page.
