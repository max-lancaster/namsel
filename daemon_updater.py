# encoding: utf-8
"""Auto-updater for the Namsel OCR daemon: version.py config, release lookup over
http.client (B310-safe _https_get), and the tkinter download/install GUI.

Split out of daemon.py (which only calls check_for_updates_async from its
update thread) so the daemon stays a lean OCR worker.
"""
import contextlib
import os
import sys
import time

# Own rate limit. The daemon is started on demand and can be launched many times in
# one session; without this, every launch paid for a network round-trip.
SELF_CHECK_STAMP = '.namsel_update_check'
SELF_CHECK_TTL_SEC = 24 * 60 * 60
# Short, so a slow or unreachable feed can never hold up an OCR daemon that is
# otherwise ready to work. The check is optional by design.
HTTP_TIMEOUT_SEC = 8


def _log(msg):
    """Diagnostics go to STDERR, never stdout.

    The updater runs on a background thread while stdout carries the daemon's
    JSON-line protocol; a print() landing between a request and its response would
    be consumed by the client as that request's reply.

    A frozen GUI build has no console (sys.stderr is None), and a closed or broken
    stream raises OSError / ValueError; either way the message has nowhere to go, and a
    diagnostic must never take the updater thread down with it.
    """
    stream = sys.stderr
    if stream is None:
        return
    with contextlib.suppress(OSError, ValueError):
        stream.write("%s\n" % msg)
        stream.flush()


def _installer_filename(download_url):
    """Filename for the downloaded installer: URL basename, query stripped, platform extension ensured."""
    # The release feed proxies assets through .../download-asset?...&filename=<real name>,
    # so the basename is the FUNCTION name and the real name is in the query. Read the
    # query first; stripping it blindly yields "download-asset.exe" for every product.
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(download_url)
    filename = (parse_qs(parsed.query).get("filename", [""])[0] or "").strip()
    if not filename:
        filename = parsed.path.split("/")[-1]
    filename = filename.split("?")[0]
    if not filename.endswith((".exe", ".pkg")):
        is_windows = sys.platform.startswith("win")
        filename = filename + (".exe" if is_windows else ".pkg")
    return filename


def _updater_version_file():
    """Path of version.py: next to the frozen exe / this file, else the parent folder."""
    import sys as _sys
    if getattr(_sys, 'frozen', False):
        installer_dir = os.path.dirname(_sys.executable)
    else:
        installer_dir = os.path.dirname(os.path.abspath(__file__))
    version_file = os.path.join(installer_dir, 'version.py')
    # Also check parent folder (Documents\namsel) if not found next to exe
    if not os.path.exists(version_file):
        version_file = os.path.join(os.path.dirname(installer_dir), 'version.py')
    return version_file


def _install_dir():
    """The Namsel install directory — the folder holding version.py and the stamp."""
    return os.path.dirname(_updater_version_file())


def _stamp_is_fresh(name, ttl_sec):
    """True when the stamp file exists and was written less than ttl_sec ago."""
    try:
        return (time.time() - os.path.getmtime(os.path.join(_install_dir(), name))) < ttl_sec
    except OSError:
        return False


def _touch_stamp(name):
    """Record that a check just happened. Failure is harmless — worst case we re-check."""
    try:
        with open(os.path.join(_install_dir(), name), 'w') as f:
            f.write(str(int(time.time())))
    except OSError:
        pass


def _should_skip_check():
    """(skip, reason) — why this updater should not run right now, if it shouldn't.

    Both gates are resolved without touching the network:
      1. NAMSEL_SKIP_UPDATE_CHECK — an explicit opt-out for headless, CI, packaged
         and air-gapped use, where a tkinter window must never appear.
      2. This updater checked recently — don't re-ask the feed on every daemon start.
    """
    if os.environ.get('NAMSEL_SKIP_UPDATE_CHECK'):
        return True, 'NAMSEL_SKIP_UPDATE_CHECK is set'
    # No version stamp means this is a source checkout, not an installed build: there is
    # nothing to compare a release against (the default would be a bare "1.0.0.0", which
    # every published release beats), and offering a source user a downloadable installer
    # is the wrong action anyway. Packaged builds ship version.py and do get checked.
    if not os.path.exists(_updater_version_file()):
        return True, 'no version.py -- running from source, not an installed build'
    if _stamp_is_fresh(SELF_CHECK_STAMP, SELF_CHECK_TTL_SEC):
        return True, 'already checked within the last %dh' % (SELF_CHECK_TTL_SEC // 3600)
    return False, None


def _strip_tag_prefix(tag_name):
    """Version string from a release tag, stripping the app-specific prefix.

    Handles 'namsel-v1.0.1' -> '1.0.1', plain 'v1.0.1' -> '1.0.1', and unprefixed
    tags unchanged. Release tags are published as namsel-v<x.y.z.b>; without this
    the whole tag reached _parse_version, failed int(), and every release compared
    as (0,0,0,0) — so no update was ever offered.
    """
    if tag_name.startswith("namsel-v"):
        return tag_name[len("namsel-v"):]
    if tag_name.startswith("v"):
        return tag_name[1:]
    return tag_name


def _load_updater_config():
    """(CURRENT_VERSION, GITHUB_RELEASES_API, BRANCH_NAME) — defaults overridden by version.py."""
    current = "1.0.0.0"
    api = "https://glossariobudacompact.netlify.app/.netlify/functions/releases"
    branch = "Namsel"
    version_file = _updater_version_file()
    if os.path.exists(version_file):
        try:
            with open(version_file, 'r') as f:
                for line in f.read().split('\n'):
                    if line.startswith('VERSION ='):
                        current = line.split('=')[1].strip().strip('"\'')
                    elif line.startswith('GITHUB_RELEASES_API ='):
                        api = line.split('=')[1].strip().strip('"\'')
                    elif line.startswith('BRANCH_NAME ='):
                        branch = line.split('=')[1].strip().strip('"\'')
        except Exception as e:
            _log(f"[daemon] version file unreadable, using defaults: {e}")
    return current, api, branch


def _https_get(url, headers=None, timeout=15, max_redirects=5):
    """GET over http.client (https/http only), following redirects.

    Returns (connection, response); the caller must close the connection.
    Replaces urllib.request.urlopen so no generic URL opener (file:// etc.)
    is ever in play (Bandit B310).
    """
    import http.client
    import urllib.parse
    for _ in range(max_redirects):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme == "https":
            conn = http.client.HTTPSConnection(parts.netloc, timeout=timeout)
        elif parts.scheme == "http":
            conn = http.client.HTTPConnection(parts.netloc, timeout=timeout)
        else:
            raise ValueError("unsupported URL scheme: %s" % url)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        if resp.status in (301, 302, 303, 307, 308):
            url = resp.getheader("Location")
            resp.read()
            conn.close()
            if not url:
                raise IOError("redirect without Location header")
            continue
        return conn, resp
    raise IOError("too many redirects: %s" % url)

def _fetch_branch_release(api, branch):
    """The NEWEST release dict for this branch, or None.

    Previously this returned the first feed entry whose target_commitish matched,
    which is only correct while the feed happens to be ordered newest-first. The
    default Netlify feed does sort that way, but GITHUB_RELEASES_API is overridable
    via version.py and GitHub's own /releases orders by created_at — where a release
    drafted early and published late sorts below an older one. Version-compare every
    candidate instead of trusting the server's order.
    """
    import json
    conn, response = _https_get(
        api,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "NamselOCR-AutoUpdater/1.0"
        },
        timeout=HTTP_TIMEOUT_SEC
    )
    try:
        if response.status != 200:
            return None
        data = json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()
    releases = data if isinstance(data, list) else [data]
    best = None
    best_version = None
    for r in releases:
        if not isinstance(r, dict) or r.get("target_commitish") != branch:
            continue
        version = _parse_version(_strip_tag_prefix(r.get("tag_name", "")))
        if best_version is None or version > best_version:
            best, best_version = r, version
    return best


def _parse_version(v):
    """A 4-part comparable tuple.

    Padding matters because Python compares tuples element-wise and a longer tuple
    with an equal prefix sorts GREATER: unpadded, a 4-part tag "1.0.1.0" would
    compare greater than an installed 3-part "1.0.1" — the same version — and the
    update prompt would reappear on every launch forever.
    """
    try:
        parts = tuple(int(p) for p in v.split("."))
    except Exception:
        return (0, 0, 0, 0)
    return (parts + (0, 0, 0, 0))[:4]


def _find_platform_asset_url(release):
    """browser_download_url of the first .exe (Windows) / .pkg (mac) asset, or None."""
    search_ext = ".exe" if sys.platform.startswith("win") else ".pkg"
    for asset in release.get("assets", []):
        if isinstance(asset, dict) and asset.get("name", "").lower().endswith(search_ext):
            return asset.get("browser_download_url")
    return None


def _format_release_notes(release):
    release_notes = release.get("body", "Notas de versão não disponíveis")
    if release_notes:
        import re
        release_notes = release_notes.replace("\\n", "\n")
        release_notes = re.sub(r"(?m)^[*-]\s+", "• ", release_notes)
    return release_notes


def check_for_updates_async():
    """Check for updates in background thread and show GUI if update available."""
    try:
        skip, reason = _should_skip_check()
        if skip:
            _log("[daemon] update check skipped: %s" % reason)
            return
        _touch_stamp(SELF_CHECK_STAMP)
        current_version, api, branch = _load_updater_config()
        release = _fetch_branch_release(api, branch)
        if not release:
            return
        tag_name = release.get("tag_name", "")
        if not tag_name:
            return
        latest_version = _strip_tag_prefix(tag_name)
        if _parse_version(latest_version) <= _parse_version(current_version):
            return  # No update needed
        download_url = _find_platform_asset_url(release)
        if not download_url:
            return
        show_update_gui(latest_version, download_url, _format_release_notes(release))
    except Exception as e:
        _log(f"[daemon] update check skipped: {e}")  # update check is optional


def _build_notes_area(body, release_notes):
    """Release-notes label + read-only scrolled text inside the body frame."""
    import tkinter as tk
    from tkinter import scrolledtext

    tk.Label(body, text="Novidades:",
             bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 8))

    notes_text = scrolledtext.ScrolledText(
        body, height=12, width=65,
        font=("Segoe UI", 9), bg="white", fg="#333",
        relief=tk.FLAT, borderwidth=1, wrap=tk.WORD, state=tk.DISABLED
    )
    notes_text.pack(fill=tk.BOTH, expand=True)
    notes_text.config(state=tk.NORMAL)
    notes_text.insert(1.0, release_notes)
    notes_text.config(state=tk.DISABLED)


def _build_update_window(version, release_notes):
    """Build the static update-window chrome; returns the widgets the
    download/install logic needs to drive."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Gerenciador de Atualizações - Namsel OCR")
    root.geometry("600x550")
    root.resizable(False, False)
    root.configure(bg="#f5f5f5")

    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 600) // 2
    y = (root.winfo_screenheight() - 550) // 2
    root.geometry(f"600x550+{x}+{y}")

    # Header
    header = tk.Frame(root, bg="#2c3e50", height=70)
    header.pack(fill=tk.X)
    header.pack_propagate(False)

    tk.Label(header, text="  Gerenciador de Atualizações - Namsel OCR",
             bg="#2c3e50", fg="white",
             font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=20, pady=20)

    # Body
    body = tk.Frame(root, bg="#f5f5f5", padx=24, pady=20)
    body.pack(fill=tk.BOTH, expand=True)

    # Version info
    version_frame = tk.Frame(body, bg="#f5f5f5")
    version_frame.pack(fill=tk.X, pady=(0, 16))

    tk.Label(version_frame, text="Atualização disponível:",
             bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
    tk.Label(version_frame, text=f"v{version}",
             bg="#f5f5f5", fg="#27ae60", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(8, 0))

    _build_notes_area(body, release_notes)

    # Progress frame
    progress_frame = tk.Frame(body, bg="#f5f5f5", height=60)
    progress_frame.pack(fill=tk.X, pady=(12, 0))
    progress_frame.pack_propagate(False)

    progress_label = tk.Label(progress_frame, text="", bg="#f5f5f5", font=("Segoe UI", 9), fg="#555")
    progress_label.pack(anchor=tk.W, pady=(0, 4))

    progress_bar = ttk.Progressbar(progress_frame, mode="determinate", length=540)

    # Buttons
    btn_frame = tk.Frame(root, bg="#ececec", pady=14)
    btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

    close_btn = tk.Button(btn_frame, text="Fechar", width=12,
                          command=root.destroy, font=("Segoe UI", 9))
    close_btn.pack(side=tk.RIGHT, padx=(0, 20))

    action_btn = tk.Button(btn_frame, text="Baixar Atualização", width=18,
                           bg="#27ae60", fg="white", activebackground="#229954",
                           font=("Segoe UI", 9, "bold"), relief=tk.FLAT, cursor="hand2")
    action_btn.pack(side=tk.RIGHT, padx=(0, 8))

    return root, progress_label, progress_bar, action_btn, close_btn


def show_update_gui(version, download_url, release_notes):
    """Show update GUI in main thread."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        import threading
        import tempfile
        import shutil
        from pathlib import Path
        # Used only for the macOS `open` launch below (arg-array, no shell).
        import subprocess  # nosec

        root, progress_label, progress_bar, action_btn, close_btn = \
            _build_update_window(version, release_notes)

        # Download logic
        installer_path = [None]  # Mutable container
        
        def update_progress(percent):
            progress_bar['value'] = percent
            progress_label.config(text=f"Baixando... {percent}%")
        
        def do_download():
            try:
                filename = _installer_filename(download_url)

                # Download to temp
                temp_dir = Path(tempfile.mkdtemp(prefix="namsel-update-"))
                temp_file = temp_dir / filename
                
                conn, response = _https_get(
                    download_url,
                    headers={"User-Agent": "NamselOCR-AutoUpdater/1.0"},
                    timeout=60
                )
                try:
                    file_size = int(response.getheader("Content-Length") or 0)
                    downloaded = 0
                    chunk_size = 8192

                    with open(temp_file, "wb") as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if file_size > 0:
                                progress = int((downloaded * 100) / file_size)
                                root.after(0, lambda p=progress: update_progress(p))
                finally:
                    conn.close()
                
                # Move to Downloads
                downloads_dir = Path.home() / "Downloads"
                downloads_dir.mkdir(exist_ok=True)
                final_path = downloads_dir / filename
                
                if final_path.exists():
                    final_path.unlink()
                
                shutil.move(str(temp_file), str(final_path))
                temp_dir.rmdir()
                
                installer_path[0] = final_path
                root.after(0, on_download_complete)
            
            except Exception as e:
                root.after(0, lambda msg=str(e): on_download_error(msg))  # bind now: Python clears `e` when the except block exits
        
        def on_download_complete():
            progress_bar['value'] = 100
            progress_label.config(text="Download concluído!", fg="#27ae60")
            action_btn.config(text="Instalar Agora", command=install_update,
                            bg="#e67e22", activebackground="#d35400")
            close_btn.config(text="Instalar Depois")
        
        def on_download_error(error):
            progress_label.config(text=f"Erro: {error}", fg="#e74c3c")
            action_btn.config(text="Tentar Novamente", command=download_update,
                            bg="#e74c3c", activebackground="#c0392b")
        
        def download_update():
            action_btn.config(state=tk.DISABLED)
            progress_label.config(text="Preparando download...")
            progress_bar['value'] = 0
            progress_bar.pack(fill=tk.X)
            threading.Thread(target=do_download, daemon=True).start()
        
        def install_update():
            if not installer_path[0] or not installer_path[0].exists():
                messagebox.showerror("Erro", "Arquivo do instalador não encontrado")
                return
            
            response = messagebox.askyesno(
                "Instalar Atualização",
                f"O instalador será executado agora.\n\n"
                f"Namsel OCR será atualizado para v{version}.\n\n"
                f"Continuar?",
                icon=messagebox.QUESTION
            )
            
            if response:
                try:
                    if sys.platform.startswith("win"):
                        # ShellExecute the .exe (native launch — no subprocess, no shell,
                        # and installer elevation manifests are honored like a double-click).
                        # Reviewed-safe: a user-confirmed, app-downloaded installer.
                        os.startfile(str(installer_path[0]))  # nosec  # nosemgrep
                    elif sys.platform == "darwin":
                        # arg-array (no shell) launch of a user-confirmed, app-downloaded installer.
                        subprocess.Popen(["open", str(installer_path[0])])  # nosec  # nosemgrep
                    
                    messagebox.showinfo(
                        "Instalador Executado",
                        "O instalador foi iniciado.\n\n"
                        "Siga as instruções na tela para concluir a atualização."
                    )
                    root.destroy()
                except Exception as e:
                    messagebox.showerror("Erro", f"Falha ao executar o instalador:\n{e}")
        
        action_btn.config(command=download_update)
        root.mainloop()
        
    except Exception as e:
        # GUI can't be shown (headless daemon / missing display) — log and continue.
        _log(f"[updater] update GUI unavailable: {e}")
