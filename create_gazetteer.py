import json
import pandas as pd
import srsly  # spaCy's utility library
import re

# This is the hardcoded database from your original code.
# We will add this to our new pattern file to ensure we keep all old logic.
OFFICER_DATABASE = {
    "names": [],
    "departments": [],
    "locations": []
}

def create_gazetteer_patterns():
    """
    Reads all your JSON data files to create a single, powerful
    pattern file for the spaCy EntityRuler.
    This file ('patterns.jsonl') will solve the NER problem.
    """
    patterns = []
    seen_patterns = set()

    def add_pattern(label, text):
        """Adds a pattern if it hasn't been added before."""
        if not isinstance(text, str) or not text.strip():
            return
        pattern_key = (label, text.lower())
        if pattern_key not in seen_patterns:
            patterns.append({"label": label, "pattern": text})
            seen_patterns.add(pattern_key)

    print("Starting pattern generation...")

    # 1. Add patterns from your original hardcoded database
    for name in OFFICER_DATABASE["names"]:
        add_pattern("OFFICER", name)
    for dept in OFFICER_DATABASE["departments"]:
        add_pattern("DEPARTMENT", dept)
    for loc in OFFICER_DATABASE["locations"]:
        add_pattern("LOCATION", loc)
    print(f"Loaded {len(patterns)} patterns from original OFFICER_DATABASE.")

    # 2. Add patterns from OdishaIPCCrimedata.json
    try:
        with open('OdishaIPCCrimedata.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        for field in data.get('fields', []):
            label = field.get('label', '')
            if "IPC" in label:
                match = re.match(r'^(.*?)\s*\((.*?IPC)\)$', label)
                if match:
                    add_pattern("CRIME_TYPE", match.group(1).strip())
                    add_pattern("LAW_SECTION", match.group(2).strip())
                else:
                    add_pattern("CRIME_TYPE", label)

        print(f"Loaded {len(patterns)} patterns after processing OdishaIPCCrimedata.json.")

    except Exception as e:
        print(f"Could not process OdishaIPCCrimedata.json: {e}")

    # 3. Add patterns from DistrictReport.json
    try:
        with open('DistrictReport.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        for district_name in data.keys():
            add_pattern("LOCATION", district_name)

        print(f"Loaded {len(patterns)} patterns after processing DistrictReport.json.")

    except Exception as e:
        print(f"Could not process DistrictReport.json: {e}")

    # 4. Add patterns from mock_cctnsdata.json
    try:
        df = pd.read_json('mock_cctnsdata.json')

        # Districts
        for district in df['district'].unique():
            add_pattern("LOCATION", district)

        # Police Stations
        for station in df['police_station'].unique():
            add_pattern("LOCATION", station)

        # Officers
        for officer_id in df['investigating_officer_id'].unique():
            add_pattern("OFFICER", officer_id)

        # Crime types
        for crime in df['crime_type'].unique():
            if "(Section" in crime:
                match = re.match(r'^(.*?)\s*\((.*?IPC)\)$', crime)
                if match:
                    add_pattern("CRIME_TYPE", match.group(1).strip())
                    add_pattern("LAW_SECTION", match.group(2).strip())
                else:
                    add_pattern("CRIME_TYPE", crime)
            else:
                add_pattern("CRIME_TYPE", crime)

        print(f"Loaded {len(patterns)} patterns after processing mock_cctnsdata.json.")

    except Exception as e:
        print(f"Could not process mock_cctnsdata.json: {e}")

    # SAVE patterns.jsonl
    srsly.write_jsonl("patterns.jsonl", patterns)
    print(f"\nSUCCESS: Created 'patterns.jsonl' with {len(patterns)} total unique patterns.")

if __name__ == "__main__":
    create_gazetteer_patterns()
