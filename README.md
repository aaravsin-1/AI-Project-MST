# if on windows run on wsl--linux based
- still might not work better to simply used a linux based laptop or refactor the code to use a different library


# create a virtual environment
1️⃣ Install Python 3.11 (if not already installed)
**in case of fedora**
`sudo dnf install python3.11 python3.11-devel python3.11-pip`
**in case of ubuntu**
`sudo apt install python3.11 python3.11-devel python3.11-pip`
**Verify:**
python3.11 --version

2️⃣ Create a virtual environment with Python 3.11
From your project root:
`python3.11 -m venv venv`

3️⃣ Activate the virtual environment
`source venv/bin/activate`
You should now see:
(venv) :...
Confirm Python version:
`python --version`
✅ Should say Python 3.11.x

4️⃣ Upgrade pip (important)
`pip install --upgrade pip`

5️⃣ Install requirements again
pip install -r requirements.txt



