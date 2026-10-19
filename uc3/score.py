"""Score results against the team's answer key.
   python -m uc3.score --make-template      (creates data/answer_key.csv from the latest results)
   python -m uc3.score                      (scores output/results.xlsx)"""
import argparse
import pandas as pd
import yaml
from .sheet import code_from_filename, FILE_RE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results")
    ap.add_argument("--key")
    ap.add_argument("--make-template", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf8"))
    res_path, key_path = a.results or cfg["output_file"], a.key or cfg["answer_key"]
    res = pd.read_excel(res_path, sheet_name="Results", dtype=str).fillna("")
    id_col = next(c for c in res.columns if c.strip().lower() in ("test name", "test id", "test"))

    if a.make_template:
        pd.DataFrame({"Test Name": res[id_col], "Expected": "", "Notes": ""}).to_csv(key_path, index=False)
        print(f"Wrote template {key_path}. Fill 'Expected' with PASS / FAIL / UNCLEAR for the tests you have graded by hand.")
        return

    key = pd.read_csv(key_path, dtype=str).fillna("")
    key["Expected"] = key["Expected"].str.strip().str.upper()
    key = key[key["Expected"].isin(["PASS", "FAIL", "UNCLEAR"])]
    m = key.merge(res[[id_col, "Result", "File Name"]].rename(columns={id_col: "Test Name"}), on="Test Name")
    m["Result"] = m["Result"].str.upper()
    m = m[m["Result"].isin(["PASS", "FAIL", "UNCLEAR"])]
    if m.empty:
        print("No graded tests found in both the key and the results."); return
    n = len(m)
    acc = (m["Expected"] == m["Result"]).mean()
    print(f"Scored {n} test(s). Overall accuracy: {acc:.0%}\n")
    print("Confusion matrix (rows = expected, columns = model):")
    print(pd.crosstab(m["Expected"], m["Result"]).reindex(index=["PASS", "FAIL", "UNCLEAR"], columns=["PASS", "FAIL", "UNCLEAR"], fill_value=0), "\n")
    wp = ((m["Result"] == "PASS") & (m["Expected"] != "PASS")).sum()
    wf = ((m["Result"] == "FAIL") & (m["Expected"] != "FAIL")).sum()
    print(f"Wrong PASS (model said PASS, expected FAIL/UNCLEAR): {wp}  <- the risky error")
    print(f"Wrong FAIL (model said FAIL, expected PASS/UNCLEAR): {wf}")
    print(f"Missed PASS (expected PASS, model said FAIL/UNCLEAR): {((m['Expected']=='PASS') & (m['Result']!='PASS')).sum()}\n")
    m["Doc"] = m["File Name"].apply(lambda s: ", ".join(c for c in (code_from_filename(f) for f in FILE_RE.findall(s)) if c))
    print("Accuracy by document:")
    print(m.groupby("Doc").apply(lambda g: f"{(g['Expected']==g['Result']).mean():.0%} ({len(g)} tests)", include_groups=False).to_string())
    miss = m[m["Expected"] != m["Result"]][["Test Name", "Expected", "Result"]]
    if len(miss):
        print("\nDisagreements to investigate:"); print(miss.to_string(index=False))


if __name__ == "__main__":
    main()
