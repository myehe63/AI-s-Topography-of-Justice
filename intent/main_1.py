import os
from extractor import run_extraction
from analyzer import run_analysis

BASE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("==============================")
    print("  Justice as a Vector")
    print("==============================\n")

    run_extraction(os.path.join(BASE, "data", "sources.json"))
    print("\n")
    run_analysis(os.path.join(BASE, "data", "extracted.json"))
