
# Student Guide: Git, GitHub & Submitting Your Machine Learning Exercises

During our course sessions, you will practice concepts of **Machine Learning in Python** within a limited time and share your Jupyter Notebooks with the instructor.

To facilitate this workflow and give you hands-on experience with standard industry practices, we will use **Git** and **GitHub**. This guide will walk you step-by-step through setting up your project, creating a private repository, authenticating via a Personal Access Token, and submitting your work on **ILIAS**.

---

## Overview of Steps

1. **Install Git & Create a GitHub Account**
2. **Create a Private Repository on GitHub** 
3. **Add Your Instructor as a Collaborator**
4. **Create Your Local Project Folder, Generate a Personal Access Token (PAT), & Push Your First Commit**
5. **Future Workflow: Adding Notebooks & Submitting Links on ILIAS**

---

## Step 1: Initial Setup

### A. Create a GitHub Account

1. Go to [github.com](https://github.com) and sign up for a free account if you do not already have one.
2. *Recommended:* Use your student email address. You can also sign up for the [GitHub Student Developer Pack](https://education.github.com/pack) to get free developer tools.

### B. Install Git

* **Windows:** Download and install [Git for Windows](https://git-scm.com/download/win). Use default installation settings.
* **Mac:** Open **Terminal** and type `git --version`. Follow the prompt to install Command Line Tools if prompted.
* **Linux:** Open your terminal and run `sudo apt install git` (Ubuntu/Debian) or the equivalent package manager command for your distribution.

### C. Configure Your Git Identity

Open your Terminal (Mac/Linux) or **Git Bash** (Windows) and set your identity:

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your-email@example.com"

```

---

## Step 2: Create a Private Repository on GitHub

1. Log in to [github.com](https://github.com).
2. Click the **`+`** icon in the top-right corner and select **New repository**.
3. Fill in the details:
* **Repository name:** `ML_with_Python_submission`
* **Visibility:** Select **Private** *(Important: Keeps your code private from other students)*.
* **Initialize repository section:** **Leave ALL boxes unchecked** (Do **NOT** add a README, .gitignore, or license).


4. Click **Create repository**.

---

## Step 3: Add Your Instructor as a Collaborator

Because your repository is private, your instructor needs explicit permission to view and grade your work.

1. On your repository page, click the **Settings** tab at the top right.
2. In the left sidebar, click **Collaborators**.
3. Click the green **Add people** button.
4. Enter your instructor’s GitHub username or email:
> **Instructor's GitHub Username :** `Salma-AZIZ`


5. Click **Add [Salma-AZIZ] to this repository**.

---

## Step 4: Create Your Local Project Folder, Setup Personal Access Token, & First Commit

---

### Part A: Local Folder Setup Commands (Run in Terminal / Git Bash)

1. **Navigate to your desired directory and create the folder:**
```bash
# Move to where you want your project stored (e.g., Documents)
cd Documents

# Create the submission folder
mkdir ML_with_Python_submission

# Navigate into the folder
cd ML_with_Python_submission

```


2. **Initialize Git in your folder:**
```bash
git init

```


3. **Create the `README.md` file containing your identification info:**
Run these lines directly in your terminal/git bash to create a `README.md` file with your full name:
```bash
echo "# ML_with_Python_submission" >> README.md
echo "First Name: [Your First Name]" >> README.md
echo "Last Name: [Your Last Name]" >> README.md

```


*(Replace `[Your First Name]` and `[Your Last Name]` with your actual name so the instructor can identify your submission).*
4. **Stage, commit, and link your repository:**
```bash
# Stage the README file
git add README.md

# Create your first commit
git commit -m "first commit"

# Rename branch to main
git branch -M main

# Link your local folder to your GitHub repository
# (Replace YOUR-USERNAME with your actual GitHub username)
git remote add origin https://github.com/YOUR-USERNAME/ML_with_Python_submission.git

# Push your local commit to GitHub
git push -u origin main

```



---

### Part B: GitHub Authentication (Creating Your Personal Access Token)

When you run `git push`, GitHub will ask for your credentials. **GitHub NO LONGER accepts your regular account password for terminal pushes.** Instead, you must generate a **Personal Access Token (PAT)** and use it as your password.

#### How to Generate Your Token on GitHub:

1. Log in to [github.com](https://github.com).
2. Click your **profile picture** in the top-right corner and go to **Settings**.
3. Scroll down the left sidebar and click **Developer settings** (at the very bottom).
4. Under **Personal access tokens**, click **Tokens (classic)**.
5. Click **Generate new token** > **Generate new token (classic)**.
6. In the **Note** box, enter a name (e.g., `ML Course Token`).
7. In the **Expiration** dropdown, select **No expiration** *(This ensures the token remains active for the entire course without expiring)*.
8. Under **Select scopes**, check the box for **`repo`** (Full control of private repositories).
9. Scroll down and click the green **Generate token** button.

> ⚠️ **CRITICAL:** GitHub will show your token **ONCE ONLY**. You cannot view it again after leaving the page! Copy the token immediately and save it in a safe place on your computer (e.g., a secure text document or password manager).

#### How to Use the Token in Terminal:

When running `git push -u origin main`:

* **Username:** Enter your GitHub username.
* **Password:** **Paste the Personal Access Token** you just generated (NOT your GitHub account password).

*(Note: When pasting passwords/tokens into terminal, characters may be hidden. Just paste and hit Enter).*

---

## Step 5: Workflow for Future Sessions & ILIAS Submissions

In upcoming practical sessions, you will complete hands-on Machine Learning exercises in **Jupyter Notebooks** (`.ipynb` files).

### How to push your completed Jupyter Notebooks during sessions:

1. Save your Jupyter Notebook (`.ipynb` file) inside your local `ML_with_Python_submission` folder.
2. Open Terminal / Git Bash inside that folder and run:
```bash
# Check modified files
git status

# Stage all new and updated files
git add .

# Commit changes with a brief note
git commit -m "Add session 1 jupyter notebook"

# Push changes to GitHub
git push

```


*(If prompted for credentials, enter your username and paste your Personal Access Token).*

### Submitting on ILIAS:

1. Go to your repository page on [github.com](https://github.com).
2. Copy the URL from your browser address bar (e.g., `https://github.com/YOUR-USERNAME/ML_with_Python_submission`).
3. Log in to **ILIAS**, navigate to the relevant assignment submission task, and paste the URL into the submission field.

---

## ⚠️ Checklist Before Finalizing

* Repository visibility is set to **Private**.
* Instructor is added under **Settings > Collaborators** (`[INSERT_INSTRUCTOR_USERNAME_HERE]`).
* `README.md` contains your real **First Name** and **Last Name**.
* Created a **Personal Access Token (Classic)** with **No Expiration** and **`repo` scope**.
* Saved the Token locally in a safe place.
* Pushed your `.ipynb` notebook file to GitHub before submitting on ILIAS.



