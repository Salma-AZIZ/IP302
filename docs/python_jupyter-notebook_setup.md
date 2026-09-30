# Setting Up Your Python & Jupyter Environment

Welcome! To participate in our coding sessions, you need a working Python environment with **Jupyter Notebook**. 

Choose **ONE** of the three setup options below based on your preference and computer specifications.

---

## Quick Comparison: Which Option Should You Choose?

| Option | Setup Difficulty | Best For | Internet Needed? |
| :--- | :--- | :--- | :--- |
| **Option 1: Anaconda** | ⭐ Easy (Recommended) | Absolute beginners who want everything pre-installed | No |
| **Option 2: Python + Pip** | ⭐⭐ Moderate | Users who prefer a lightweight setup or already have Python | No |
| **Option 3: Google Colab** | ⭐ Instant | Emergency backup / Zero installation | Yes (Cloud-based) |

---

## Option 1: Anaconda (Recommended – All-in-One Package)

Anaconda installs Python, Jupyter Notebook, and all essential Data Science libraries (Pandas, NumPy, Scikit-learn, Matplotlib) in a single installer.

### Step-by-Step Installation:
1. Go to the [Anaconda Download Page](https://www.anaconda.com/download/success).
2. Download the installer for your operating system (Windows, Mac, or Linux).
3. Run the installer and follow the prompts:
   * **Windows users:** Keep default options checked.
4. Once completed, restart your computer.

### How to Launch Jupyter Notebook:
* **Method 1 (GUI):** Open **Anaconda Navigator** from your applications list and click **Launch** under Jupyter Notebook.
* **Method 2 (Terminal/Prompt):** Open **Anaconda Prompt** (Windows) or **Terminal** (Mac/Linux) and type:
  ```bash
  jupyter notebook

```

---

## Option 2: Install Python & Jupyter Separately (Lightweight)

Use this method if you do not want the large Anaconda distribution and prefer a minimal setup.

### Step 1: Install Python

1. Go to the [Official Python Downloads Page](https://www.python.org/downloads/).
2. Download and run the latest Python 3.x installer.
3. ⚠️ **CRITICAL (Windows Users):** On the very first installer screen, check the box that says **"Add python.exe to PATH"** before clicking Install.

### Step 2: Install Jupyter Notebook

1. Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux).
2. Upgrade `pip` and install Jupyter:
```bash
python -m pip install --upgrade pip
pip install notebook

```



### How to Launch Jupyter Notebook:

Open your Terminal / Command Prompt and type:

```bash
jupyter notebook

```

---

## Option 3: Google Colab (No Installation Required)

If you encounter installation issues during class or have computer hardware constraints, you can run Python notebooks directly in your browser using Google Colab.

1. Go to [Google Colab](https://colab.research.google.com/).
2. Sign in with any Google account.
3. Click **New Notebook** to start coding immediately.
4. *Note:* Notebooks created in Colab are saved directly to your Google Drive in the `Colab Notebooks` folder. You will need to download the `.ipynb` file to your computer to push it to your Git repository.

---

## First Steps in Jupyter Notebook

### 1. Where to Launch Jupyter (Local Users)

By default, Jupyter launches in your user home folder. To open Jupyter directly inside your class folder:

```bash
# Navigate to your project folder first
cd Documents/ML_with_Python_submission

# Launch Jupyter from this location
jupyter notebook

```

### 2. Basic Shortcuts & Controls

* **Create a Cell:** Click the `+` button in the toolbar, or press `Esc` then `B` (below) / `A` (above).
* **Run a Cell & Move Next:** Press `Shift + Enter`.
* **Run a Cell & Stay:** Press `Ctrl + Enter`.
* **Cell Types:**
* **Code:** For writing Python code.
* **Markdown:** For writing text, notes, and documentation.



### 3. Verify Your Environment

Create a new notebook, type the following code in the first cell, and press `Shift + Enter`:

```python
import sys
print(f"Python Version: {sys.version}")
print("Environment setup successful! Ready for Machine Learning.")

```

---

## 🛠️ Verification & Troubleshooting

### Check Version Numbers in Terminal:

```bash
python --version
pip --version
jupyter notebook --version

```

### Common Issues & Quick Fixes:

* **`'jupyter' is not recognized as an internal command` (Windows):**
Re-run the Python installer, select **Modify**, and ensure **Add Python to environment variables / PATH** is checked.
* **Command not found after installation:**
Close your terminal window completely and reopen it (or restart your computer).

```