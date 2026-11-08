import argparse, json, re, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

USER_AGENT = "FinScale Research haraccyber@gmail.com"  # <-- EDIT THIS
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Sector-diverse ticker universe (~150 names) for the hackathon data-expansion pass.
# CIKs are resolved dynamically from SEC's own ticker map (see resolve_ciks()) rather than
# hand-typed here, since mistyping a CIK silently mislabels/fails that company's filings.
DEFAULT_TICKERS = sorted(set([
    # Technology
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "CRM", "ADBE", "CSCO", "IBM",
    "INTC", "AMD", "QCOM", "TXN", "AVGO", "NFLX", "PYPL", "SHOP", "UBER", "ABNB",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK", "AXP", "COF",
    "USB", "PNC", "TFC", "MET", "PRU", "AIG", "ALL", "TRV", "CB", "PGR",
    "SPGI", "MCO", "ICE", "CME", "NDAQ",
    # Healthcare
    "JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "BMY", "AMGN", "GILD", "CVS",
    # Energy
    "XOM", "CVX", "COP", "SLB", "VLO", "MPC", "PSX", "OXY", "HAL", "BKR",
    "EOG", "DVN", "KMI", "WMB", "OKE",
    # Consumer
    "WMT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "TGT", "DG", "DLTR",
    "TJX", "ROST", "PG", "KO", "PEP", "CL", "KMB", "EL", "MDLZ", "GIS",
    "KHC", "CAG",
    # Industrials / Materials
    "BA", "LMT", "RTX", "GE", "CAT", "DE", "UPS", "FDX", "HON", "MMM",
    "EMR", "ITW", "PH", "ETN", "DOW", "DD", "LIN", "APD", "ECL", "NEM",
    "FCX", "NUE", "X", "AA",
    # Communications / Media / Travel
    "DIS", "CMCSA", "VZ", "T", "TMUS", "BKNG", "MAR", "HLT", "LYFT",
    # Utilities
    "SO", "DUK", "NEE", "D", "AEP", "EXC",
    # Autos
    "F", "GM", "TSLA",
]))

FILINGS_PER_COMPANY = 2
OUT = Path("data/raw"); OUT.mkdir(parents=True, exist_ok=True)
CIK_CACHE = Path("data/.cik_map.json")


def get(url, timeout=60):
    time.sleep(0.2)  # SEC limit is 10 req/s; stay well under
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r


def resolve_ciks(tickers):
    """Resolve tickers -> 10-digit zero-padded CIKs via SEC's official ticker map.

    Avoids hand-typing CIKs for a large ticker universe, which is easy to get wrong silently.
    Caches the full SEC map locally so repeated runs (or runs against more tickers) don't
    re-fetch a multi-MB file every time.
    """
    if CIK_CACHE.exists():
        full_map = json.loads(CIK_CACHE.read_text(encoding="utf-8"))
    else:
        data = get("https://www.sec.gov/files/company_tickers.json").json()
        full_map = {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}
        CIK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CIK_CACHE.write_text(json.dumps(full_map), encoding="utf-8")
    resolved, missing = {}, []
    for t in tickers:
        cik = full_map.get(t.upper())
        if cik:
            resolved[t.upper()] = cik
        else:
            missing.append(t)
    if missing:
        print(f"[warn] no CIK found for: {', '.join(missing)}")
    return resolved


def list_10k_filings(cik, n):
    data = get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    rec = data["filings"]["recent"]
    out = []
    for form, acc, doc, date in zip(rec["form"], rec["accessionNumber"],
                                    rec["primaryDocument"], rec["filingDate"]):
        if form == "10-K":
            out.append((acc, doc, date))
        if len(out) >= n:
            break
    return out


def extract_risk_factors(html):
    text = BeautifulSoup(html, "lxml").get_text("\n")
    text = re.sub(r"\n\s*\n+", "\n", text)
    # Find all "Item 1A" starts; the real section is the longest span
    # between an "Item 1A" heading and the next "Item 1B" / "Item 2".
    starts = [m.start() for m in re.finditer(r"item\s*1a\.?\s*risk\s*factors", text, re.I)]
    best = ""
    for s in starts:
        m = re.search(r"item\s*(1b|2)\.?\s", text[s + 200:], re.I)
        end = s + 200 + m.start() if m else min(len(text), s + 200000)
        chunk = text[s:end]
        if len(chunk) > len(best):
            best = chunk
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers-file", type=Path, default=None,
                    help="text file, one ticker per line; defaults to the ~150-ticker sector-diverse DEFAULT_TICKERS list")
    ap.add_argument("--max-companies", type=int, default=None, help="cap the number of companies (useful for a quick smoke test)")
    ap.add_argument("--filings-per-company", type=int, default=FILINGS_PER_COMPANY, help="10-Ks per company, i.e. years of history")
    args = ap.parse_args()

    tickers = (args.tickers_file.read_text(encoding="utf-8").split()
               if args.tickers_file else DEFAULT_TICKERS)
    if args.max_companies:
        tickers = tickers[: args.max_companies]

    companies = resolve_ciks(tickers)
    for ticker, cik in tqdm(companies.items(), desc="Companies"):
        try:
            filings = list_10k_filings(cik, args.filings_per_company)
        except Exception as e:
            print(f"[skip] {ticker}: {e}"); continue
        for acc, doc, date in filings:
            path = OUT / f"{ticker}_{date}.txt"
            if path.exists():
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
            try:
                rf = extract_risk_factors(get(url).text)
            except Exception as e:
                print(f"[skip] {ticker} {date}: {e}"); continue
            if len(rf) < 5000:
                print(f"[skip] {ticker} {date}: risk section too short"); continue
            path.write_text(rf)

if __name__ == "__main__":
    main()
