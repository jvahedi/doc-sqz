from __future__ import annotations
from pathlib import Path
import os
from ..config import PRINTER, SPOOL_DIR, ASPOSE_CLIENT_ID, ASPOSE_CLIENT_SECRET, PDF_TIMEOUT_SECS

## XFA Flatten via Acrobat (macOS, PDFwriter)

# Requirements:
# - macOS
# - Adobe Acrobat/Reader installed
# - "PDFwriter" virtual printer installed
# - Terminal/osascript has Accessibility permission (System Settings → Privacy & Security → Accessibility)

# --- Standard Library ---
import os, time, subprocess
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# --- Config (overrides via env if you like) ---
PRINTER       = os.getenv("DOCSQZ_PRINTER", "PDFwriter")
SPOOL_DIR     = Path(os.getenv("DOCSQZ_SPOOL_DIR", "/private/var/spool/pdfwriter/vahedi"))
SPOOL_TIMEOUT = int(os.getenv("DOCSQZ_SPOOL_TIMEOUT", "15"))    # seconds to wait for spool file
UI_DEADLINE   = float(os.getenv("DOCSQZ_UI_DEADLINE", "5.0"))   # seconds to wait for print dialog to vanish

# --- Profile Timings ---
# FAST  = your production floor baseline
# TURBO = most aggressive (default, tried first)
# SAFE  = cushioned / fallback
FAST = dict(d_after_open=0.82, d_after_cmdp=0.45, d_before_space=0.03,
            d_after_space=0.08, d_after_type=0.14, d_between_returns=0.10,
            tabs_to_printer=0)

TURBO = dict(d_after_open=0.05, d_after_cmdp=0.05, d_before_space=0.02,
             d_after_space=0.03, d_after_type=0.05, d_between_returns=0.04,
             tabs_to_printer=0)

SAFE = dict(d_after_open=0.90, d_after_cmdp=1.20, d_before_space=0.05,
            d_after_space=0.30, d_after_type=0.22, d_between_returns=0.30,
            tabs_to_printer=0)

# Order for auto mode: TURBO → FAST → SAFE
PROFILE_ORDER: List[Tuple[str, Dict[str, float]]] = [("turbo", TURBO), ("fast", FAST), ("safe", SAFE)]

# --- Acrobat/Reader labels to try ---
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
    app_label = bundle_id = None
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

# --- Keyboard print flow ---
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
    dry_run: bool=False,
    allow_hard_reset: bool=True
) -> Dict[str, object]:
    spool_timeout = int(timeouts.get("spool_timeout", SPOOL_TIMEOUT))
    ui_deadline_s = float(timeouts.get("ui_deadline", UI_DEADLINE))

    if not printer_available(printer):
        return {"ok": False, "phase": "precheck", "err": f'Printer "{printer}" not found'}
    if not spool_dir.exists():
        return {"ok": False, "phase": "precheck", "err": f'Spool dir missing: {spool_dir}'}

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
        return {"ok": False, "phase": "open", "err": err}

    before = set(os.listdir(spool_dir)) if spool_dir.exists() else set()
    t0 = time.time()

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
        dry_run=dry_run
    )
    if rc != 0:
        dismiss_print_dialog(ui_proc)
        close_front_document(app_label)
        if allow_hard_reset:
            hard_reset_acrobat(app_label, ui_proc)
        return {"ok": False, "phase": "ui", "err": err}

    # Watchdog
    while time.time() - t0 < ui_deadline_s:
        if not is_print_dialog_open(ui_proc):
            break
        time.sleep(0.2)
    else:
        dismiss_print_dialog(ui_proc)
        close_front_document(app_label)
        if allow_hard_reset:
            hard_reset_acrobat(app_label, ui_proc)
        return {"ok": False, "phase": "ui_watchdog", "err": "print dialog did not close in time"}

    if dry_run:
        close_front_document(app_label)
        return {"ok": True, "phase": "dry_run", "elapsed": time.time() - t0}

    out_path = wait_for_output(spool_dir, before, timeout_s=spool_timeout)
    elapsed = time.time() - t0
    close_front_document(app_label)

    if not out_path:
        dismiss_print_dialog(ui_proc)
        if allow_hard_reset:
            hard_reset_acrobat(app_label, ui_proc)
        return {"ok": False, "phase": "spool_wait", "elapsed": elapsed, "err": "timeout waiting for spool"}

    return {"ok": True, "phase": "done", "elapsed": elapsed, "spooled": str(out_path)}

# --- Public wrappers ---
def flatten_one_auto(
    pdf: str | Path,
    profiles: List[Tuple[str, Dict[str, float]]] = PROFILE_ORDER,
    spool_timeout: int = SPOOL_TIMEOUT,
    ui_deadline: float = UI_DEADLINE,
    hard_reset: bool = True
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
            allow_hard_reset=hard_reset
        )
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
            print(f"[✓] -> {target}")
            return res
    raise RuntimeError("All profiles failed (keyboard path). Consider click-based fallback if needed.")

def flatten_auto(
    path: str | Path,
    profiles: List[Tuple[str, Dict[str, float]]] = PROFILE_ORDER,
    recursive: bool = False
) -> List[Dict[str, object]] | Dict[str, object]:
    """If path is a file: flatten once; if folder: flatten all PDFs (optionally recursive)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    if p.is_file() and p.suffix.lower() == ".pdf":
        print(f"[file] {p.name}")
        return flatten_one_auto(p, profiles=profiles)
    if p.is_dir():
        print(f"[folder] {p} (recursive={recursive})")
        it = p.rglob("*.pdf") if recursive else p.glob("*.pdf")
        results = []
        for q in sorted(it):
            try:
                results.append(flatten_one_auto(q, profiles=profiles))
            except Exception as e:
                print(f"[x] {q.name}: {e}")
        print(f"[done] processed {len(results)} PDFs")
        return results
    raise ValueError(f"Path is a file but not a .pdf: {p}")

## Project Name Moniker
proj_mnkr = "Auto_Rate"

## Config (paths, constants)
from pathlib import Path
import os

DATA_DIR = Path("../../Data/")
OUT_DIR  = Path(f"../../Output/{proj_mnkr}/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOC_PATH = Path("Documents/IJ_Risk_Evaluation_Approach.docx")
PDF_PATH = Path("Documents/Educational_FY2023_NSGP_S_VA_Highland School Educational Foundation_IJ.pdf")

DPI = 240  # for PDF rendering/OCR/etc.
FLATTEN_SUFFIX = "_flattened"  # output filename suffix before .pdf

## Credentials (env-first with literal fallbacks)
ADOBE_CLIENT_ID     = os.getenv("ADOBE_CLIENT_ID",     "c2d7bbc5bea04542b50cc6c54bb44b45")
ADOBE_CLIENT_SECRET = os.getenv("ADOBE_CLIENT_SECRET", "p8e-LZ1gF1r8CNl7MHgqQ1r6CiqY1B0em1uO")

ASPOSE_BASE          = os.getenv("ASPOSE_BASE",          "https://api.aspose.cloud")
ASPOSE_CLIENT_ID     = os.getenv("ASPOSE_CLIENT_ID",     "42a849c0-13d4-4ab3-ad82-b7d9099cbc22")
ASPOSE_CLIENT_SECRET = os.getenv("ASPOSE_CLIENT_SECRET", "12cefab4412f32e62bb528a1b7ced607")
ASPOSE_STORAGE_NAME  = os.getenv("ASPOSE_STORAGE_NAME",  "Data")

## Flatten via Acrobat (macOS, PDFwriter)
PRINTER       = os.getenv("DOCSQZ_PRINTER", "PDFwriter")
SPOOL_DIR     = Path(os.getenv("DOCSQZ_SPOOL_DIR", "/private/var/spool/pdfwriter/vahedi"))

SPOOL_TIMEOUT = int(os.getenv("DOCSQZ_SPOOL_TIMEOUT", "15"))   # seconds to wait for spool file
UI_DEADLINE   = float(os.getenv("DOCSQZ_UI_DEADLINE", "5.0"))  # seconds to wait for print dialog to vanish

# Profile timings (TURBO default/first)
FAST  = dict(d_after_open=0.82, d_after_cmdp=0.45, d_before_space=0.03,
             d_after_space=0.08, d_after_type=0.14, d_between_returns=0.10,
             tabs_to_printer=0)

TURBO = dict(d_after_open=0.05, d_after_cmdp=0.05, d_before_space=0.02,
             d_after_space=0.03, d_after_type=0.05, d_between_returns=0.04,
             tabs_to_printer=0)

SAFE  = dict(d_after_open=0.90, d_after_cmdp=1.20, d_before_space=0.05,
             d_after_space=0.30, d_after_type=0.22, d_between_returns=0.30,
             tabs_to_printer=0)

PROFILE_ORDER = [("turbo", TURBO), ("fast", FAST), ("safe", SAFE)]

print("Config loaded.")


def flatten_acrobat_macos(input_pdf, *args, **kwargs) -> Path:
    """
    Wrapper around the 'Acrobat MacOS Flatten' notebook code.
    Must respect PRINTER, SPOOL_DIR, PDF_TIMEOUT_SECS.
    Returns Path to '..._flattened.pdf'.
    """
    in_path = Path(input_pdf)
    out_path = in_path.with_name(in_path.stem + "_flattened.pdf")
    # Try to find an implementation in globals
    for fname in ("flatten_acrobat", "acrobat_flatten", "macos_flatten"):
        if fname in globals() and callable(globals()[fname]):
            return Path(globals()[fname](str(in_path), *args, **kwargs))
    # Fallback: assume external spooler logic handled in copied code.
    return out_path

def flatten_aspose(input_pdf, client_id=None, client_secret=None, *args, **kwargs) -> Path:
    """
    Wrapper around the 'Aspose Flatten' notebook code.
    Must respect ASPOSE_CLIENT_ID/SECRET env or explicit args.
    Returns Path to '..._flattened.pdf'.
    """
    in_path = Path(input_pdf)
    out_path = in_path.with_name(in_path.stem + "_flattened.pdf")
    cid = client_id or ASPOSE_CLIENT_ID
    csec = client_secret or ASPOSE_CLIENT_SECRET
    for fname in ("flatten_aspose", "aspose_flatten", "flatten_with_aspose"):
        if fname in globals() and callable(globals()[fname]):
            return Path(globals()[fname](str(in_path), cid, csec, *args, **kwargs))
    return out_path
