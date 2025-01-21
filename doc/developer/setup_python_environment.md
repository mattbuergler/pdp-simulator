# Setup a virtual python environment with Pipenv

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

## Windows

### Install Python 3.9.0

Install Python from here:
https://www.python.org/downloads/release/python-390/

### Install *pyenv*
In the command prompt run:

```shell
pip install pyenv-win --target %USERPROFILE%\\.pyenv
```

If you run into an error with the above command use the folllowing instead:

```shell
pip install pyenv-win --target %USERPROFILE%\\.pyenv --no-user --upgrade
```

Add system settings via PowerShell:

Adding PYENV, PYENV_HOME and PYENV_ROOT to your Environment Variables:

```powershell

[System.Environment]::SetEnvironmentVariable('PYENV',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")

[System.Environment]::SetEnvironmentVariable('PYENV_ROOT',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")

[System.Environment]::SetEnvironmentVariable('PYENV_HOME',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
```

Now adding the following paths to your USER PATH variable in order to access the pyenv command


```powershell
[System.Environment]::SetEnvironmentVariable('path', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('path', "User"),"User")
```

### Install *pipenv*

Follow the instructions here:
https://www.pythontutorial.net/python-basics/install-pipenv-windows/

Add the following variable to the User Environment Variables
PIPENV_VENV_IN_PROJECT=1

