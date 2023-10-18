# Requirements

## Ubuntu/Debian

### Install *pyenv* and Python 3.9.0


Source: https://realpython.com/intro-to-pyenv/#installing-pyenv

**Build Dependencies:**

```shell
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
```
**Install pyenv **
```shell
curl https://pyenv.run | bash
```
**Add the following lines to .bashrc:**

```shell
export PATH="/home/$USER/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

**Install Python 3.9.0:**

```shell
pyenv install -v 3.9.0
```
**Upgrade pyenv**

```shell
cd $(pyenv root)
git pull
```
### Install *pipenv*
https://realpython.com/pipenv-guide/

```shell
pip install pipenv
```

## Windows 10

### Install Python 3.9.0

Install Python from here:
https://www.python.org/downloads/release/python-390/

### Install *pyenv*

```shell
pip install pyenv-win
```
### Install *pipenv*

Follow the instructions here:
https://www.pythontutorial.net/python-basics/install-pipenv-windows/

Add the following variable to the User Environment Variables
PIPENV_VENV_IN_PROJECT=1

# Running the code

## Start the python-environment
1. Open a new window of the "Windows Powershell (x86)"
2. Navigate to the repository containing the Piplock file
3. Start the python environment:

```shell
pipenv install --python C:\Users\Matthias\.pyenv\pyenv-win\versions\3.9.0\python.exe
```
