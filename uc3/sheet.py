"""Read MVB's test sheet and map each test to the document file(s) it names.

Matching is done on the FILE NAME column (not the document-code column) because the
sample sheet has mismatches (e.g. T50-T53 list ISP-001 files but the code ISO-001).
Cells that run several file names together are split on the '.pdf' boundary.
"""
import re
from dataclasses import dataclass, field
import pandas as pd

FILE_RE = re.compile(r"[\w\-\.]+?\.pdf", re.I)
CODE_RE = re.compile(r"^([A-Za-z]{2,4})[-_ ]?(\d{3})")


def code_from_filename(name: str):
    """'SAN002_Sanctions_Risk_Assessment.pdf' -> 'SAN-002' (hyphen-insensitive)."""
    m = CODE_RE.match(name.strip())
    return f"{m.group(1).upper()}-{m.group(2)}" if m else None


@dataclass
class Test:
    test_id: str
    text: str
    files: list = field(default_factory=list)
    codes: list = field(default_factory=list)      # normalized codes derived from file names
    declared_codes: str = ""                       # raw 'Document Code(s)' cell, for mismatch warnings
    row: int = 0                                   # 0-based row in the sheet


def _find_col(df, *names):
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in lowered:
            return lowered[n]
    raise KeyError(f"Test sheet is missing a column like {names!r}. Found: {list(df.columns)}")


def load_sheet(path):
    return pd.read_excel(path, dtype=str).fillna("")


def load_tests(path):
    df = load_sheet(path)
    c_id = _find_col(df, "test name", "test id", "test")
    c_files = _find_col(df, "file name", "file names", "file")
    c_text = _find_col(df, "tests", "test description", "requirement")
    c_codes = next((c for c in df.columns if "document code" in str(c).lower()), None)
    tests = []
    for i, r in df.iterrows():
        tid = str(r[c_id]).strip()
        if not tid:
            continue
        files = FILE_RE.findall(str(r[c_files]))
        codes = [c for c in (code_from_filename(f) for f in files) if c]
        tests.append(Test(tid, str(r[c_text]).strip(), files, codes,
                          str(r[c_codes]).strip() if c_codes else "", i))
    return tests


def select_tests(tests, test_ids):
    """Keep requested tests in their original workbook order."""
    wanted = {str(test_id).strip().upper() for test_id in test_ids}
    return [test for test in tests if test.test_id.upper() in wanted]


def test_selector_options(sheet_df, tests):
    """Return test IDs sorted by category and requirement, with readable labels."""
    columns = {str(column).strip().lower(): column for column in sheet_df.columns}
    category_col = columns.get("document types") or columns.get("category")
    name_col = next(
        (columns[name] for name in ("tests", "test description", "requirement")
         if name in columns),
        None,
    )
    labels = {}
    for test in tests:
        row = sheet_df.iloc[test.row]
        name = str(row[name_col]).strip() if name_col else ""
        labels[test.test_id] = f"{name} ({test.test_id})" if name else test.test_id

    ordered_tests = sorted(
        tests,
        key=lambda test: (
            str(sheet_df.iloc[test.row][category_col]).strip().casefold()
            if category_col else "",
            str(sheet_df.iloc[test.row][name_col]).strip().casefold()
            if name_col else "",
            test.test_id.casefold(),
        ),
    )
    return list(dict.fromkeys(test.test_id for test in ordered_tests)), labels


def test_selector_categories(sheet_df, tests):
    """Group test IDs under their workbook document-type categories."""
    columns = {str(column).strip().lower(): column for column in sheet_df.columns}
    category_col = columns.get("document types") or columns.get("category")
    categories = {}
    for test in tests:
        category = (
            str(sheet_df.iloc[test.row][category_col]).strip()
            if category_col else ""
        )
        names = list(dict.fromkeys(
            name.strip() for name in category.split(",") if name.strip()
        )) or ["Uncategorized"]
        for name in names:
            categories.setdefault(name, []).append(test.test_id)
    return {
        category: list(dict.fromkeys(test_ids))
        for category, test_ids in sorted(
            categories.items(),
            key=lambda item: item[0].casefold(),
        )
    }


def recommend_tests(tests, uploaded_codes, available_codes):
    """Recommend tests linked to uploads whose required documents are all available."""
    uploaded = {str(code).strip().upper() for code in uploaded_codes}
    available = {str(code).strip().upper() for code in available_codes}
    return [
        test for test in tests
        if test.codes
        and uploaded.intersection(test.codes)
        and set(test.codes).issubset(available)
    ]


def sheet_rows_for_tests(sheet_df, tests):
    """Return only the original workbook rows corresponding to selected tests."""
    return sheet_df.iloc[[test.row for test in tests]].copy()
