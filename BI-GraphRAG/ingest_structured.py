"""
ingest_structured.py — Load structured CSV data into the BI graph as triples.

No LLM involved. Each CSV row is converted to entity-relationship-entity triples
using the same format as LLM extraction, so core/load.py handles both identically.

ID-to-name resolution is done upfront so all triples use human-readable names.
This lets Neo4j MERGE naturally cross-link CSV entities with identically-named
entities extracted from PDFs (e.g. "Joy Gardner" in a meeting note connects to
the Employee node loaded here).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# fallback only — each section passes its own filename
SOURCE = "structured_data"


def _triple(subject, relationship, obj, s_type="Entity", o_type="Entity", source=SOURCE):
    return {
        "subject": str(subject).strip(),
        "relationship": relationship,
        "object": str(obj).strip(),
        "subject_type": s_type,
        "object_type": o_type,
        "source": source,
        "page_number": None,
        "bbox": None,
        "section": None,
    }


def load_csv_triples(data_dir: str | Path) -> list[dict]:
    """
    Read all CSVs from data_dir and return a flat list of triples.
    Order: company → departments → regions → products → employees → customers → orders → order_items
    """
    d = Path(data_dir)
    triples: list[dict] = []

    # ── Build ID → name lookup tables ─────────────────────────────────────────
    def read(name):
        path = d / name
        if not path.exists():
            logger.warning("Missing %s — skipping", name)
            return pd.DataFrame()
        return pd.read_csv(path)

    companies   = read("company.csv")
    departments = read("departments.csv")
    regions     = read("regions.csv")
    products    = read("products.csv")
    employees   = read("employees.csv")
    customers   = read("customers.csv")
    orders      = read("orders.csv")
    order_items = read("order_items.csv")

    # id → name maps
    dept_name   = dict(zip(departments["id"], departments["name"]))   if not departments.empty   else {}
    region_name = dict(zip(regions["id"],     regions["name"]))       if not regions.empty       else {}
    prod_name   = dict(zip(products["id"],    products["name"]))      if not products.empty      else {}
    emp_name    = dict(zip(employees["id"],   employees["name"]))     if not employees.empty     else {}
    cust_name   = dict(zip(customers["id"],   customers["name"]))     if not customers.empty     else {}
    company_name = companies.iloc[0]["name"] if not companies.empty else "Company"

    # ── Company ────────────────────────────────────────────────────────────────
    for _, row in companies.iterrows():
        name = row["name"]
        triples += [
            _triple(name, "IS_A", row.get("industry", "Company"), "Organization", "Concept", "company.csv"),
            _triple(name, "FOUNDED_IN", str(int(row["founded_year"])), "Organization", "Concept", "company.csv"),
        ]

    # ── Departments ────────────────────────────────────────────────────────────
    for _, row in departments.iterrows():
        name = row["name"]
        triples += [
            _triple(name, "IS_A", "Department", "Organization", "Concept", "departments.csv"),
            _triple(name, "PART_OF", company_name, "Organization", "Organization", "departments.csv"),
        ]

    # ── Regions ────────────────────────────────────────────────────────────────
    for _, row in regions.iterrows():
        triples.append(_triple(row["name"], "IS_A", "Region", "Location", "Concept", "regions.csv"))

    # ── Products ───────────────────────────────────────────────────────────────
    for _, row in products.iterrows():
        name = row["name"]
        triples += [
            _triple(name, "IS_A", row["category"], "Topic", "Concept", "products.csv"),
            _triple(name, "PRICED_AT", f"${float(row['price']):,.2f}", "Topic", "Measurement", "products.csv"),
        ]

    # ── Employees ──────────────────────────────────────────────────────────────
    for _, row in employees.iterrows():
        name = row["name"]
        dept  = dept_name.get(row.get("department_id", ""), "")
        mgr   = emp_name.get(row.get("manager_id", ""), "")

        triples.append(_triple(name, "IS_A", "Employee", "Person", "Concept", "employees.csv"))
        triples.append(_triple(name, "HAS_ROLE", row["job_title"], "Person", "Concept", "employees.csv"))
        triples.append(_triple(name, "EMPLOYED_BY", company_name, "Person", "Organization", "employees.csv"))
        if dept:
            triples.append(_triple(name, "WORKS_IN", dept, "Person", "Organization", "employees.csv"))
        if mgr:
            triples.append(_triple(name, "REPORTS_TO", mgr, "Person", "Person", "employees.csv"))
        if pd.notna(row.get("hire_date")):
            triples.append(_triple(name, "HIRED_ON", str(row["hire_date"]), "Person", "Concept", "employees.csv"))

    # ── Customers ──────────────────────────────────────────────────────────────
    for _, row in customers.iterrows():
        name   = row["name"]
        region = region_name.get(row.get("region_id", ""), "")

        triples.append(_triple(name, "IS_A", "Customer", "Organization", "Concept", "customers.csv"))
        triples.append(_triple(name, "REPRESENTS", row["company_name"], "Organization", "Organization", "customers.csv"))
        if region:
            triples.append(_triple(name, "LOCATED_IN", region, "Organization", "Location", "customers.csv"))
        if pd.notna(row.get("customer_since")):
            triples.append(_triple(name, "CUSTOMER_SINCE", str(row["customer_since"]), "Organization", "Concept", "customers.csv"))

    # ── Orders ─────────────────────────────────────────────────────────────────
    for _, row in orders.iterrows():
        customer  = cust_name.get(row.get("customer_id", ""), "")
        employee  = emp_name.get(row.get("employee_id", ""), "")
        amount    = f"${float(row['total_amount']):,.2f}"

        if customer and employee:
            triples.append(_triple(employee, "SOLD_TO", customer, "Person", "Organization", "orders.csv"))
            triples.append(_triple(customer, "ORDER_WORTH", amount, "Organization", "Measurement", "orders.csv"))

    # ── Order items ────────────────────────────────────────────────────────────
    orders_idx = orders.set_index("id") if not orders.empty else pd.DataFrame()
    for _, row in order_items.iterrows():
        order_id = row.get("order_id", "")
        product  = prod_name.get(row.get("product_id", ""), "")
        if not product or orders_idx.empty or order_id not in orders_idx.index:
            continue
        order_row = orders_idx.loc[order_id]
        customer  = cust_name.get(order_row.get("customer_id", ""), "")
        employee  = emp_name.get(order_row.get("employee_id", ""), "")
        qty       = int(row.get("quantity", 1))

        if customer:
            triples.append(_triple(customer, "PURCHASED", product, "Organization", "Topic", "order_items.csv"))
        if employee:
            triples.append(_triple(employee, "SOLD", product, "Person", "Topic", "order_items.csv"))
        if qty > 1:
            triples.append(_triple(product, "ORDERED_QTY", str(qty), "Topic", "Measurement", "order_items.csv"))

    # Deduplicate: same triple from multiple orders should only appear once
    seen = set()
    unique: list[dict] = []
    for t in triples:
        key = (t["subject"], t["relationship"], t["object"])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    logger.info("CSV ingestion: %d unique triples from %s", len(unique), d)
    return unique


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    d = sys.argv[1] if len(sys.argv) > 1 else "Fake_BI_data/data/structured"
    triples = load_csv_triples(d)
    print(f"\n{len(triples)} triples. Sample:")
    for t in triples[:10]:
        print(f"  {t['subject']} --{t['relationship']}--> {t['object']}")
