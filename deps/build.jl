"""PyCall has its own little Python install, and we need to make sure that we
install all the right dependencies in that environment. Note that you need to
restart Julia after running this in order to be able to @pyimport the installed
packages. Run this with `] build`.
"""

import PyCall: pyimport

# See https://stackoverflow.com/questions/12332975/installing-python-module-within-code and https://gist.github.com/Luthaf/368a23981c8ec095c3eb.
const PIP_PACKAGES = ["taichi==0.7.12", "matplotlib==3.3.4"]

sys = pyimport("sys")
subprocess = pyimport("subprocess")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", PIP_PACKAGES...])
