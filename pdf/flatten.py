"""
PDF flattening utilities for docsqz.

Includes:
- Acrobat macOS flattening via keyboard automation + spool watching
- Aspose Cloud flattening (XFA → AcroForm → flatten → download)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import os.path as op
import time
import subprocess

# ==============================================================
# Config: prefer DOCSQZ.config, else fall back to safe defaults
# ==============================================================

# Fallbacks = your original working values
_FALLBACK = {
    "PRINTER":       os.getenv("DOCSQZ_PRINTER", "PDFwriter"),
    "SPOOL_DIR":     Path(os.getenv("DOCSQZ_SPOOL_DIR", "/private/var/spool/pdfwriter/vahedi")),
    "SPOOL_TIMEOUT": int(os.getenv("DOCSQZ_SPOOL_TIMEOUT", "15")),
    "UI_DEADLINE":   float(os.getenv("DOCSQZ_UI_DEADLINE", "5.0")),
}

# Original profile timings
FAST = dict(d_after_open=0.82, d_after_cmdp=0.45, d_before_space=0.03,
            d_after_space=0.08, d_after_type=0.14, d_between_returns=0.10,
            tabs_to_printer=0)

TURBO = dict(d_after_open=0.05, d_after_cmdp=0.05, d_before_space=0.02,
             d_after_space=0.03, d_after_type=0.05, d_between_returns=0.04,
             tabs_to_printer=0)

SAFE = dict(d_after_open=0.90, d_after_cmdp=1.20, d_before_space=0.05,
            d_after_space=0.30, d_after_type=0.22, d_between_returns=0.30,
            tabs_to_printer=0)

_DEFAULT_PROFILE_ORDER: List[Tuple[str, Dict[str, float]]] = [("turbo", TURBO), ("fast", FAST), ("safe", SAFE)]

# Try to import project config (case-sensitive). If missing, use fallbacks.
try:
    from DOCSQZ.config import (  # type: ignore
        PRINTER, SPOOL_DIR, PROFILE_ORDER,
        SPOOL_TIMEOUT, UI_DEADLINE,
        ASPOSE_BASE, ASPOSE_CLIENT_ID, ASPOSE_CLIENT_SECRET, ASPOSE_STORAGE_NAME,
    )
except Exception:
    PRINTER       = _FALLBACK["PRINTER"]
    SPOOL_DIR     = _FALLBACK["SPOOL_DIR"]
    SPOOL_TIMEOUT = _FALLBACK["SPOOL_TIMEOUT"]
    UI_DEADLINE   = _FALLBACK["UI_DEADLINE"]
    PROFILE_ORDER = _DEFAULT_PROFILE_ORDER
    # Aspose fallbacks (these must be set via env or passed in)
    ASPOSE_BASE           = os.getenv("ASPOSE_BASE", "https://api.aspose.cloud")
    ASPOSE_CLIENT_ID      = os.getenv("ASPOSE_CLIENT_ID", "")
    ASPOSE_CLIENT_SECRET  = os.getenv("ASPOSE_CLIENT_SECRET", "")
    ASPOSE_STORAGE_NAME   = os.getenv("ASPOSE_STORAGE_NAME", None)

# ======================================================================
# --- Acrobat macOS flattening path (keyboard automation + spool watch)
# ======================================================================

# Acrobat/Reader labels to try
APP_LABEL_CANDIDATES = [
    "Adobe Acrobat", "Adobe Acrobat DC",
    "Adobe Acrobat Reader DC", "Adobe Acrobat Reader",
    "Acrobat Reader",
]

# --- Core helpers ---
def _osascript(lines: List[str]) -> Tuple[int, str, str]:
    """Run AppleScript lines and return (rc, stdout, stderr)."""
    cmd = ["osascript"] + sum([["-e", L] for L in lines], [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()

def _sh(cmd: str) -> Tuple[int, str, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()

def printer_available(printer_name: str) -> bool:
    # system_profiler first; fallback to lpstat if available
    rc, out, _ = _sh('system_profiler SPPrintersDataType | sed -n "s/^ *Printer Name: //p"')
    names = {n.strip() for n in out.splitlines() if n.strip()} if rc == 0 else set()
    if printer_name in names:
        return True
    rc, out, _ = _sh("command -v lpstat >/dev/null 2>&1 && lpstat -p || true")
    return (rc == 0) and (printer_name in out)

def detect_acrobat_app_and_process() -> Tuple[str, str, str]:
    """Return (app_label, bundle_id, ui_process_name) for Acrobat/Reader."""
    app_label = bundle_id = None  # type: ignore[assignment]
    for label in APP_LABEL_CANDIDATES:
        rc, out, _ = _osascript([f'id of application "{label}"'])
        if rc == 0 and out:
            app_label, bundle_id = label, out.strip()
            break
    if not app_label:
        raise RuntimeError("No Acrobat/Reader app found.")
    rc, out, err = _osascript([
        f'tell application "{app_label}" to activate',
        'tell application "System Events"',
        '  set foundName to ""',
        '  repeat 100 times',
        f'    if exists (application process 1 whose bundle identifier is "{bundle_id}") then',
        f'      set foundName to name of (application process 1 whose bundle identifier is "{bundle_id}")',
        '      exit repeat',
        '    end if',
        '    delay 0.1',
        '  end repeat',
        'end tell',
        'return foundName',
    ])
    if rc != 0 or not out:
        raise RuntimeError(f"Could not discover Acrobat UI process. {err}")
    return app_label, bundle_id, out.strip()

def wait_for_output(spool_dir: Path, before_names: set, timeout_s: int) -> Optional[Path]:
    """Wait for a new, stable .pdf to appear in the spool_dir."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(0.6)
        try:
            names = os.listdir(spool_dir)
        except FileNotFoundError:
            continue
        new = [n for n in names if n.lower().endswith(".pdf") and n not in before_names]
        if new:
            paths = sorted((spool_dir / n for n in new), key=lambda p: p.stat().st_mtime, reverse=True)
            p = paths[0]
            s1 = p.stat().st_size
            time.sleep(0.4)
            if p.exists() and p.stat().st_size == s1 and s1 > 0:
                return p
    return None

# --- Guards / watchdog ---
def is_print_dialog_open(ui_process: str) -> bool:
    """Restore the original, reliable detection (sheet OR window has a Print button)."""
    rc, out, _ = _osascript([
        'tell application "System Events"',
        f'  tell application process "{ui_process}"',
        '    set hasSheet to false',
        '    set hasWin  to false',
        '    try',
        '      if exists sheet 1 of window 1 then set hasSheet to true',
        '    end try',
        '    try',
        '      if exists button "Print" of window 1 then set hasWin to true',
        '    end try',
        '    return (hasSheet as string) & "," & (hasWin as string)',
        '  end tell',
        'end tell',
    ])
    if rc != 0 or not out:
        return False
    s_sheet, s_win = out.split(",")
    return (s_sheet == "true") or (s_win == "true")

def accept_save_panel_if_present(ui_process: str) -> bool:
    rc, out, _ = _osascript([
        'tell application "System Events"',
        f'  tell application process "{ui_process}"',
        '    set hasSave to false',
        '    try',
        '      if exists sheet 1 of window 1 then',
        '        try',
        '          if exists text field 1 of sheet 1 of window 1 then set hasSave to true',
        '        end try',
        '      end if',
        '    end try',
        '    if hasSave then',
        '      try',
        '        key code 36 -- Return',
        '      end try',
        '    end if',
        '    return hasSave as string',
        '  end tell',
        'end tell',
    ])
    return (rc == 0 and out == "true")

def dismiss_print_dialog(ui_process: str) -> bool:
    _osascript([
        'tell application "System Events"',
        f'  tell application process "{ui_process}"',
        '    try',
        '      click button "Cancel" of sheet 1 of window 1',
        '    on error',
        '      try',
        '        click button "Cancel" of window 1',
        '      on error',
        '        key code 53 -- Escape',
        '      end try',
        '    end try',
        '  end tell',
        'end tell',
    ])
    return not is_print_dialog_open(ui_process)

def close_front_document(app_label: str):
    _osascript([f'tell application "{app_label}" to close front document saving no'])

def hard_reset_acrobat(app_label: str, ui_process: str, wait_s: float = 4.0) -> bool:
    _osascript([f'tell application "{app_label}" to quit'])
    for _ in range(int(wait_s * 10)):
        rc, out, _ = _osascript([
            'tell application "System Events"',
            f'  return (exists application process "{ui_process}") as string',
            'end tell'
        ])
        if rc == 0 and out == "false":
            return True
        time.sleep(0.1)
    _osascript(['do shell script "pkill -x Acrobat || pkill -x \\"Adobe Acrobat\\" || true"'])
    time.sleep(1.0)
    rc, out, _ = _osascript([
        'tell application "System Events"',
        f'  return (exists application process "{ui_process}") as string',
        'end tell'
    ])
    return (rc == 0 and out == "false")

# --- Keyboard flow ---
def print_keyboard_once(
    ui_process: str,
    printer_name: str,
    d_after_open=0.8,
    d_after_cmdp=1.0,
    d_before_space=0.05,
    d_after_space=0.2,
    d_after_type=0.2,
    d_between_returns=0.2,
    tabs_to_printer=0,
    dry_run=False
):
    """Bring to front → ⌘P → [Tab x N] → Space (open printer) → type → Return → Return."""
    safe_printer = printer_name.replace('"', '\\"')
    lines = [
        'tell application "System Events"',
        f'  tell application process "{ui_process}" to set frontmost to true',
        'end tell',
        f'delay {max(d_after_open, 0):.2f}',
        'tell application "System Events" to keystroke "p" using command down',
        f'delay {max(d_after_cmdp, 0):.2f}',
        'tell application "System Events"',
    ]
    if tabs_to_printer > 0:
        lines += [
            f'  repeat {int(tabs_to_printer)} times',
            '    keystroke tab',
            '    delay 0.12',
            '  end repeat',
        ]
    lines += [
        f'  delay {max(d_before_space, 0):.2f}',
        '  key code 49',
        f'  delay {max(d_after_space, 0):.2f}',
        f'  keystroke "{safe_printer}"',
        f'  delay {max(d_after_type, 0):.2f}',
        '  key code 36',
        f'  delay {max(d_between_returns, 0):.2f}',
    ]
    lines += (['  key code 53'] if dry_run else ['  key code 36'])
    lines += ['end tell']
    return _osascript(lines)

# --- Guarded trial ---
def run_trial_guarded(
    pdf: Path,
    printer: str,
    spool_dir: Path,
    timeouts: Dict[str, float],
    delays: Dict[str, float],
    dry_run: bool = False,
    allow_hard_reset: bool = True,
    close_app_after: bool = False,          # <— NEW
) -> Dict[str, object]:

    t0 = time.time()

    def _ok(phase: str, **kw):
        d = {"ok": True, "phase": phase, "elapsed": time.time() - t0, "err": ""}
        d.update(kw); return d

    def _fail(phase: str, msg: str, **kw):
        d = {"ok": False, "phase": phase, "elapsed": time.time() - t0, "err": msg}
        d.update(kw); return d

    # ----- Prechecks
    spool_timeout = int(timeouts.get("spool_timeout", SPOOL_TIMEOUT))
    ui_deadline_s = float(timeouts.get("ui_deadline", UI_DEADLINE))

    if not pdf.exists():
        return _fail("precheck", f"PDF not found: {pdf}")
    if not printer_available(printer):
        return _fail("precheck", f'Printer "{printer}" not found')
    if not spool_dir.exists():
        return _fail("precheck", f"Spool dir missing: {spool_dir}")

    app_label, _, ui_proc = detect_acrobat_app_and_process()

    # Clean start
    _osascript([f'tell application "{app_label}" to activate'])
    if is_print_dialog_open(ui_proc):
        dismiss_print_dialog(ui_proc)
    close_front_document(app_label)

    # Open target PDF
    rc, _, err = _osascript([
        f'tell application "{app_label}" to activate',
        f'tell application "{app_label}" to open POSIX file "{pdf.resolve()}"',
        f'delay {max(delays.get("d_after_open", 0.8), 0):.2f}',
    ])
    if rc != 0:
        return _fail("open", err or "failed to open PDF in Acrobat")

    before = set(os.listdir(spool_dir)) if spool_dir.exists() else set()

    # Print via keyboard
    rc, _, err = print_keyboard_once(
        ui_process=ui_proc,
        printer_name=printer,
        d_after_open=delays.get("d_after_open", 0.8),
        d_after_cmdp=delays.get("d_after_cmdp", 1.0),
        d_before_space=delays.get("d_before_space", 0.05),
        d_after_space=delays.get("d_after_space", 0.2),
        d_after_type=delays.get("d_after_type", 0.2),
        d_between_returns=delays.get("d_between_returns", 0.2),
        tabs_to_printer=delays.get("tabs_to_printer", 0),
        dry_run=dry_run,
    )
    if rc != 0:
        try: dismiss_print_dialog(ui_proc)
        except: pass
        try: close_front_document(app_label)
        except: pass
        if allow_hard_reset:
            try: hard_reset_acrobat(app_label, ui_proc)
            except: pass
        return _fail("ui", err or "keyboard print failed")

    # Dry run: exit early
    if dry_run:
        try: close_front_document(app_label)
        except: pass
        return _ok("dry_run")

    # Watchdog: wait for print dialog to close (up to ui_deadline), then spool (up to spool_timeout)
    t_ui_deadline = time.time() + ui_deadline_s
    while time.time() < t_ui_deadline:
        try:
            accept_save_panel_if_present(ui_proc)
        except:
            pass
        if not is_print_dialog_open(ui_proc):
            break
        time.sleep(0.2)
    else:
        # UI never closed
        dismiss_print_dialog(ui_proc)
        close_front_document(app_label)
        if allow_hard_reset:
            hard_reset_acrobat(app_label, ui_proc)
        return _fail("ui_watchdog", "print dialog did not close in time")

    # Spool wait (original behavior)
    out_path = wait_for_output(spool_dir, before, timeout_s=spool_timeout)
    if not out_path:
        dismiss_print_dialog(ui_proc)
        if allow_hard_reset:
            hard_reset_acrobat(app_label, ui_proc)
        close_front_document(app_label)
        return _fail("spool_wait", f"timeout waiting for spool ({spool_dir})")

    close_front_document(app_label)
    if close_app_after:                      # <— NEW
        _osascript([f'tell application "{app_label}" to quit'])
        _osascript(['delay 0.2'])
    return _ok("done", spooled=str(out_path))

# --- Public wrappers ---
def flatten_one_auto(
    pdf: str | Path,
    profiles: List[Tuple[str, Dict[str, float]]] = PROFILE_ORDER,
    spool_timeout: int = SPOOL_TIMEOUT,
    ui_deadline: float = UI_DEADLINE,
    hard_reset: bool = True,
    close_app_after: bool = False,
    verbose = False,# <— NEW
) -> Dict[str, object]:
    """Try profiles in order (TURBO → FAST → SAFE). Rename output to *_flattened.pdf."""
    pdf = Path(pdf).expanduser().resolve(strict=True)
    for name, delays in profiles:
        res = run_trial_guarded(
            pdf=pdf,
            printer=PRINTER,
            spool_dir=SPOOL_DIR,
            timeouts={"spool_timeout": spool_timeout, "ui_deadline": ui_deadline},
            delays=delays,
            dry_run=False,
            allow_hard_reset=hard_reset,
            close_app_after=close_app_after, # <— NEW
        )
        if not verbose:
                print('PDF flattened')
        else:
            print(f"[{name}] ok={res['ok']} phase={res.get('phase')} "
                  f"elapsed={round(res.get('elapsed',0),2)} err={res.get('err','')}")
        if res["ok"]:
            sp = Path(res["spooled"])
            target = pdf.with_name(pdf.stem + "_flattened.pdf")
            try:
                sp.rename(target)
            except Exception:
                target.write_bytes(sp.read_bytes())
                try: sp.unlink()
                except: pass
            res["output"] = str(target)
            if verbose:
                print(f"[✓] -> {target}")
            return res
    raise RuntimeError("All profiles failed (keyboard path). Consider click-based fallback if needed.")

# ======================================================================
# --- Aspose Cloud flattening path
# ======================================================================

def aspose_get_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 120,
) -> str:
    """
    Fetch OAuth token from Aspose Cloud.
    You must supply client_id/secret (or set env/ASPOSE_*).
    """
    import requests
    cid = (client_id or ASPOSE_CLIENT_ID or "").strip()
    csec = (client_secret or ASPOSE_CLIENT_SECRET or "").strip()
    if not cid or not csec:
        raise RuntimeError("Aspose credentials missing (client_id/client_secret).")
    r = requests.post(
        f"{base}/connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": cid,
            "client_secret": csec,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["access_token"]

def aspose_upload_file(local_path: str, remote_path: str,
                       storage_name: Optional[str] = None,
                       token: Optional[str] = None,
                       base: str = ASPOSE_BASE,
                       timeout: int = 300) -> None:
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    with open(local_path, "rb") as f:
        r = requests.put(
            f"{base}/v3.0/pdf/storage/file/{remote_path}",
            headers={"Authorization": f"Bearer {tok}"},
            params={"storageName": storage},
            data=f,
            timeout=timeout,
        )
    r.raise_for_status()

def aspose_convert_xfa_to_acroform(input_remote_path: str, out_remote_path: str,
                                   storage_name: Optional[str] = None,
                                   token: Optional[str] = None,
                                   base: str = ASPOSE_BASE,
                                   timeout: int = 600) -> None:
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    name_in  = op.basename(input_remote_path)
    folder_in = op.dirname(input_remote_path)
    r = requests.put(
        f"{base}/v3.0/pdf/{name_in}/convert/xfatoacroform",
        headers={"Authorization": f"Bearer {tok}"},
        params={"folder": folder_in, "outPath": out_remote_path, "storageName": storage},
        timeout=timeout,
    )
    r.raise_for_status()

def aspose_flatten_in_place(remote_path: str,
                            storage_name: Optional[str] = None,
                            token: Optional[str] = None,
                            base: str = ASPOSE_BASE,
                            timeout: int = 600) -> None:
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    name_conv   = op.basename(remote_path)
    folder_conv = op.dirname(remote_path)
    r = requests.put(
        f"{base}/v3.0/pdf/{name_conv}/fields/flatten",
        headers={"Authorization": f"Bearer {tok}"},
        params={"folder": folder_conv, "storageName": storage},
        timeout=timeout,
    )
    r.raise_for_status()

def aspose_download_file(remote_path: str, local_out: str,
                         storage_name: Optional[str] = None,
                         token: Optional[str] = None,
                         base: str = ASPOSE_BASE,
                         timeout: int = 600) -> str:
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    r = requests.get(
        f"{base}/v3.0/pdf/storage/file/{remote_path}",
        headers={"Authorization": f"Bearer {tok}"},
        params={"storageName": storage},
        stream=True, timeout=timeout,
    )
    r.raise_for_status()
    os.makedirs(op.dirname(local_out) or ".", exist_ok=True)
    with open(local_out, "wb") as out:
        for chunk in r.iter_content(1 << 20):
            out.write(chunk)
    return local_out

def flatten_xfa_pdf_with_aspose(
    local_in: str,
    in_remote: str = "incoming/input.pdf",
    out_remote: str = "converted/xfa_to_acroform.pdf",
    local_out: str = "your_xfa_flattened.pdf",
    storage_name: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> str:
    """
    End-to-end: upload -> XFA->AcroForm -> flatten in place -> download.
    Returns local_out path.
    """
    tok = aspose_get_token(client_id=client_id, client_secret=client_secret)
    aspose_upload_file(local_in, in_remote, storage_name=storage_name, token=tok)
    aspose_convert_xfa_to_acroform(in_remote, out_remote, storage_name=storage_name, token=tok)
    aspose_flatten_in_place(out_remote, storage_name=storage_name, token=tok)
    return aspose_download_file(out_remote, local_out, storage_name=storage_name, token=tok)

# ======================================================================
# --- Public wrappers for package API
# ======================================================================
def flatten_acrobat_macos(pdf_path: str | Path, close_app_after: bool = False) -> Path:
    """
    Public wrapper: flatten using Acrobat macOS (keyboard automation).
    Returns Path to the flattened PDF.
    """
    res = flatten_one_auto(pdf_path, close_app_after=close_app_after)
    return Path(res["output"])

def flatten_aspose(pdf_path: str | Path,
                   client_id: Optional[str] = None,
                   client_secret: Optional[str] = None) -> Path:
    """
    Public wrapper: flatten using Aspose Cloud.
    Returns Path to the flattened PDF.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    out_path = pdf_path.with_name(pdf_path.stem + "_flattened.pdf")
    flatten_xfa_pdf_with_aspose(str(pdf_path), local_out=str(out_path),
                                client_id=client_id, client_secret=client_secret)
    return out_path
