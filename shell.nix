let
  # Last updated: 2/4/21
  pkgs = import (fetchTarball("https://github.com/NixOS/nixpkgs/archive/2c58a9dac0bd5f5620394c7c6f623355f9f476d2.tar.gz")) {};

  # Rolling updates, not deterministic.
  # pkgs = import (fetchTarball("channel:nixpkgs-unstable")) {};
in pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3.pkgs.pip
    pkgs.julia
    pkgs.ffmpeg
  ];
  shellHook = ''
    # Hacks to make taichi work:
    # See https://nixos.wiki/wiki/Packaging/Quirks_and_Caveats#ImportError:_libstdc.2B.2B.so.6:_cannot_open_shared_object_file:_No_such_file.
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib/:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.xorg.libX11}/lib/:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.ncurses5.out}/lib/:$LD_LIBRARY_PATH"

    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}
