"""One-off helper: rebuild a tests.xlsx from the Synthetic_Tests.pdf printout.
Your real MVB sheet is already an Excel file, so you will not need this for real data.
   python tools/make_sample_tests.py Synthetic_Tests.pdf data/tests.xlsx"""
import re, sys
import pandas as pd
import pdfplumber

src, dst = sys.argv[1], sys.argv[2]
with pdfplumber.open(src) as pdf:
    text = "\n".join(p.extract_text() for p in pdf.pages)
text = text.split("Evidence / Page Ref", 1)[1]
text = re.sub(r"\s*\n\s*", " ", text)
parts = re.split(r"(?<![A-Za-z0-9\-_])T(\d{1,3})\s+(?=[\w\-]+\.pdf)", text)
rows = []
for i in range(1, len(parts), 2):
    tid, body = f"T{parts[i]}", parts[i + 1].strip()
    files = re.findall(r"[\w\-\.]+?\.pdf", body)
    rest = body[body.rfind(".pdf") + 4:].strip()
    codes = re.match(r"^((?:[A-Z]{3}-\d{3}\s*,?\s*)+)", rest)
    code_str = re.sub(r"\s+", " ", codes.group(1)).strip().rstrip(",") if codes else ""
    rest = rest[len(codes.group(1)):].strip() if codes else rest
    stem = re.sub(r"^[A-Za-z]{3}-?\d{3}_", "", files[0][:-4]).strip("_").split("_")
    words = rest.split()
    if len(files) == 1 and [w.lower() for w in words[:len(stem)]] == [w.lower() for w in stem]:
        doctype, test = " ".join(words[:len(stem)]), " ".join(words[len(stem):])
    else:                                   # multi-file rows: doc types are listed twice, then the test text
        m = re.match(r"^(.*?(?:Assessment|Documentation|Procedure))(?:,\s*)?(.*?(?:Assessment|Documentation|Procedure))?\s+(.*)$", rest)
        doctype, test = (m.group(1) + (", " + m.group(2) if m.group(2) else ""), m.group(3)) if m else ("", rest)
    rows.append({"Test Name": tid, "File Name": ", ".join(files), "Document Code(s)": code_str,
                 "Document Types": doctype, "Tests": test, "Result": "", "Evidence / Page Ref": ""})
# The three multi-document rows have cells that run together in the printout; set them explicitly.
OVERRIDES = {
 "T10": ("SAN-001_Sanctions_Program.pdf, SAN002_Sanctions_Risk_Assessment.pdf", "SAN-001, SAN-002",
         "Sanctions Risk Assessment, Sanctions Risk Assessment", "Existing customer rescreening occurs multiple times per day."),
 "T54": ("PEN-001_Penetration_Testing_Procedure.pdf, ISP001_Information_Security_Policy.pdf", "PEN-001, ISP-001",
         "Penetration Testing Procedure", "Vulnerability remediation for medium and low severity findings identified during penetration tests has an SLA of less than 100 days."),
 "T79": ("AML-001_AML_Program_Documentation.pdf, TXM001_Transaction_Monitoring_Documentation.pdf", "AML-001, TXM-001",
         "AML Program Documentation, Transaction Monitoring Documentation", "What dollar amount triggers review for potential structuring/suspicious activity?"),
}
for r in rows:
    if r["Test Name"] in OVERRIDES:
        r["File Name"], r["Document Code(s)"], r["Document Types"], r["Tests"] = OVERRIDES[r["Test Name"]]
df = pd.DataFrame(rows)
df.to_excel(dst, index=False)
print(len(df), "tests ->", dst)
