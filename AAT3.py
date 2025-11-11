
import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict, Counter
from itertools import combinations

csv_path = "E:\SEM2\DV\data_scopus.csv"
df = pd.read_csv(csv_path, dtype=str, encoding_errors="ignore")
orig_cols = list(df.columns)
df.columns = [c.strip().lower() for c in df.columns]

def pick_col(candidates):
    for c in df.columns:
        for pat in candidates:
            if re.fullmatch(pat, c):
                return c
    return None

col_authors_with_aff = pick_col([r"authors with affiliations", r"authors with affiliations?"])
col_authors = pick_col([r"authors?", r"author names?", r"author"])
col_affiliations = pick_col([r"affiliations?", r"author affiliations?"])
col_aff_country = pick_col([r"affiliation country", r"country", r"countries?"])
col_year = pick_col([r"year", r"publication year", r"py"])
col_title = pick_col([r"title", r"document title"])
col_eid = pick_col([r"eid", r"doi", r"pub id", r"publication id", r"source id"])

# Masks
idx = df.index
mask_year = pd.Series(False, index=idx)
if col_year:
    mask_year = df[col_year].astype(str).str.strip().ne("")

mask_auth = pd.Series(False, index=idx)
if col_authors_with_aff:
    mask_auth = mask_auth | df[col_authors_with_aff].astype(str).str.strip().ne("")
if col_authors:
    mask_auth = mask_auth | df[col_authors].astype(str).str.strip().ne("")

mask_aff = pd.Series(False, index=idx)
if col_authors_with_aff:
    mask_aff = mask_aff | df[col_authors_with_aff].astype(str).str.strip().ne("")
if col_affiliations:
    mask_aff = mask_aff | df[col_affiliations].astype(str).str.strip().ne("")

mask = mask_year & mask_auth & mask_aff
df_clean = df[mask].copy()

def norm_space(x):
    return re.sub(r"\s+", " ", str(x)).strip()

def clean_author(name):
    name = norm_space(name)
    name = re.sub(r"\s*\[\d+\]$", "", name)
    return name

def parse_authors_with_aff(text):
    authors = []
    if not isinstance(text, str) or not text.strip():
        return authors
    parts = [p.strip() for p in re.split(r";|\|", text) if p.strip()]
    for p in parts:
        aff = None
        m_paren = re.search(r"\(([^)]+)\)\s*$", p)
        if m_paren:
            aff = m_paren.group(1).strip()
            name = p[:m_paren.start()].strip().rstrip(",")
        else:
            chunks = [c.strip() for c in p.split(",")]
            if len(chunks) >= 2:
                aff = chunks[-1]
                name = ", ".join(chunks[:-1]).strip()
            else:
                name = p.strip()
        authors.append((clean_author(name), norm_space(aff) if aff else None))
    return authors

def parse_authors_simple(auth_text):
    if not isinstance(auth_text, str) or not auth_text.strip():
        return []
    parts = [clean_author(p) for p in re.split(r";|\|", auth_text) if p.strip()]
    return parts

records = []
for idx, row in df_clean.iterrows():
    yraw = row.get(col_year) if col_year else None
    year = None
    if isinstance(yraw, str):
        m = re.search(r"\d{4}", yraw)
        if m:
            year = int(m.group(0))
    eid = row.get(col_eid) if col_eid else None
    title = row.get(col_title) if col_title else None
    country = row.get(col_aff_country) if col_aff_country else None

    if col_authors_with_aff and isinstance(row.get(col_authors_with_aff), str) and row[col_authors_with_aff].strip():
        pairs = parse_authors_with_aff(row[col_authors_with_aff])
        if pairs:
            authors = [a for a, _ in pairs]
            affs = dict(pairs)
        else:
            authors = parse_authors_simple(row.get(col_authors, ""))
            affs = {}
    else:
        authors = parse_authors_simple(row.get(col_authors, ""))
        affs = {}
    aff_fallback = row.get(col_affiliations) if col_affiliations else None

    if not authors:
        continue

    per_author = []
    for a in authors:
        a_aff = affs.get(a) or aff_fallback
        per_author.append({
            "name": a,
            "affiliation": norm_space(a_aff) if isinstance(a_aff, str) and a_aff.strip() else None,
            "country": norm_space(country) if isinstance(country, str) and country.strip() else None
        })

    records.append({
        "paper_id": eid or (title or f"row_{idx}"),
        "year": year,
        "authors": per_author
    })

aff_counter = defaultdict(Counter)
country_counter = defaultdict(Counter)
pubs_by_author = defaultdict(set)

for rec in records:
    pid = rec["paper_id"]
    for a in rec["authors"]:
        name = a["name"]
        if not name:
            continue
        pubs_by_author[name].add(pid)
        if a["affiliation"]:
            aff_counter[name][a["affiliation"]] += 1
        if a["country"]:
            country_counter[name][a["country"]] += 1

nodes = {}
for name, pubs in pubs_by_author.items():
    aff = (aff_counter[name].most_common(1)[0][0] if aff_counter[name] else None)
    country = (country_counter[name].most_common(1)[0][0] if country_counter[name] else None)
    nodes[name] = {
        "id": name,
        "name": name,
        "affiliation": aff,
        "country": country
    }

edge_weights = defaultdict(int)
years_seen = defaultdict(list)
for rec in records:
    authors = [a["name"] for a in rec["authors"] if a["name"]]
    if len(authors) < 2:
        continue
    for u, v in combinations(sorted(set(authors)), 2):
        key = (u, v)
        edge_weights[key] += 1
        if rec["year"]:
            years_seen[key].append(rec["year"])

links = []
for (u, v), w in edge_weights.items():
    yrs = years_seen.get((u, v), [])
    link = {"source": u, "target": v, "weight": int(w)}
    if yrs:
        link["year_min"] = int(min(yrs))
        link["year_max"] = int(max(yrs))
    links.append(link)

deg = defaultdict(int)
for l in links:
    deg[l["source"]] += 1
    deg[l["target"]] += 1
for name in nodes:
    nodes[name]["degree"] = int(deg.get(name, 0))

country_counts = Counter([n["country"] for n in nodes.values() if n["country"]])
summary = {
    "num_rows_in_csv": int(len(df)),
    "num_rows_used": int(len(df_clean)),
    "num_nodes": int(len(nodes)),
    "num_links": int(len(links)),
    "top_countries": country_counts.most_common(12)
}

out = {"nodes": list(nodes.values()), "links": links, "summary": summary}
out_path = "E:\SEM2\DV\author_network.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

import caas_jupyter_tools
sum_df = pd.DataFrame([summary])
caas_jupyter_tools.display_dataframe_to_user("Author Network Summary", sum_df)

print("Wrote JSON to:", out_path)
print(json.dumps(summary, indent=2))

