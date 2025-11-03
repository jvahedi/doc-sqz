"""
Configuration module for docsqz.
Preserves environment variables, defaults, and timing profiles.
"""

from pathlib import Path
import os

# -----------------------
# Project paths
# -----------------------
proj_mnkr = "Auto_Rate"

DATA_DIR = Path("../../Data/")
OUT_DIR  = Path(f"../../Output/{proj_mnkr}/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOC_PATH = Path("Documents/IJ_Risk_Evaluation_Approach.docx")
PDF_PATH = Path("Documents/Educational_FY2023_NSGP_S_VA_Highland School Educational Foundation_IJ.pdf")

DPI = 240  # for PDF rendering/OCR/etc.
FLATTEN_SUFFIX = "_flattened"  # output filename suffix before .pdf

# -----------------------
# Credentials (env-first with literal fallbacks)
# -----------------------
ADOBE_CLIENT_ID     = os.getenv("ADOBE_CLIENT_ID",     
                                #"c2d7bbc5bea04542b50cc6c54bb44b45"
                                "a5573ab4685248388189f6fa8e6b6b87"
                               )
ADOBE_CLIENT_SECRET = os.getenv("ADOBE_CLIENT_SECRET", 
                                # "p8e-LZ1gF1r8CNl7MHgqQ1r6CiqY1B0em1uO"
                                "p8e-tm12IhU9nBhOosw7aqlKo4K3zpm8_Jw6"
                               )

ASPOSE_BASE          = os.getenv("ASPOSE_BASE",          "https://api.aspose.cloud")
ASPOSE_CLIENT_ID     = os.getenv("ASPOSE_CLIENT_ID",     "42a849c0-13d4-4ab3-ad82-b7d9099cbc22")
ASPOSE_CLIENT_SECRET = os.getenv("ASPOSE_CLIENT_SECRET", "12cefab4412f32e62bb528a1b7ced607")
ASPOSE_STORAGE_NAME  = os.getenv("ASPOSE_STORAGE_NAME",  "Data")

# -----------------------
# Flatten via Acrobat (macOS, PDFwriter)
# -----------------------
PRINTER       = os.getenv("DOCSQZ_PRINTER", "PDFwriter")
SPOOL_DIR     = Path(os.getenv("DOCSQZ_SPOOL_DIR", "/private/var/spool/pdfwriter/vahedi"))

SPOOL_TIMEOUT = int(os.getenv("DOCSQZ_SPOOL_TIMEOUT", "15"))   # seconds to wait for spool file
UI_DEADLINE   = float(os.getenv("DOCSQZ_UI_DEADLINE", "5.0"))  # seconds to wait for print dialog to vanish

# -----------------------
# Profile timings (TURBO default/first)
# -----------------------
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

#print("Config loaded.")
