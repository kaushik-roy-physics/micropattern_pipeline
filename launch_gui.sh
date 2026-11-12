#!/bin/bash
# =============================================================================
# Start the pipeline GUI (macOS / Linux).
#   macOS : double-click launch_gui.command   (or: bash launch_gui.command)
#   Linux : bash launch_gui.sh
# The first start builds a private Python environment in your home folder
# (a few minutes, needs internet). Later starts take seconds. The GUI opens in
# your web browser and is only reachable from this computer.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${GUI_VENV:-$HOME/.micropattern_pipeline_gui_venv}"

pause_exit() { echo ""; echo "$1"; read -r -p "Press Enter to close." _ || true; exit 1; }

# A usable environment has pip. Ubuntu's system Python lacks it unless the
# python3.X-venv package is installed, and venv then leaves a half-built folder
# behind; such a folder is removed and the next Python on the list is tried.
venv_ok() { [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; }

if ! venv_ok; then
    [ -f "$VENV/pyvenv.cfg" ] && { echo "Removing the incomplete environment in $VENV ..."; rm -rf "$VENV"; }
    found=""
    for cand in python3.12 python3.11 python3.10 python3 python; do
        command -v "$cand" >/dev/null 2>&1 &&
            "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null || continue
        found=1
        echo "First start: creating the GUI's Python environment in $VENV with $(command -v "$cand") ..."
        if "$cand" -m venv "$VENV" >/dev/null 2>&1 && venv_ok; then break; fi
        rm -rf "$VENV"
    done
    [ -n "$found" ] || pause_exit "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/ and try again."
    venv_ok || pause_exit "Could not create $VENV: no Python here can build an environment with pip. On Ubuntu/Debian run: sudo apt install python3-venv"
fi
if ! cmp -s "$HERE/requirements-gui.txt" "$VENV/requirements-gui.txt"; then
    echo "Installing / updating packages (one-off, a few minutes) ..."
    "$VENV/bin/python" -m pip install --upgrade pip --quiet
    "$VENV/bin/python" -m pip install -r "$HERE/requirements-gui.txt" \
        || pause_exit "Package installation failed - check the internet connection."
    cp "$HERE/requirements-gui.txt" "$VENV/requirements-gui.txt"
fi

# Streamlit asks for an e-mail address on its very first run and waits at that
# prompt before serving anything; an existing credentials file skips it.
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

echo "Starting the GUI - it opens in your browser. Close this window to stop it."
exec "$VENV/bin/python" -m streamlit run "$HERE/gui/app.py" \
    --server.address localhost --browser.gatherUsageStats false "$@"
