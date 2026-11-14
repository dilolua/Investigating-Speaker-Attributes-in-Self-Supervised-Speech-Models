"""
build_metadata_from_clean_transcripts.py

Builds a sentence-level metadata table directly from Eleanor's
clean-transcript files (00cleanTranscripts), joined to real gender/dialect
labels using pin_lookup_table2.txt + mixer6_demographics.csv.

This produces the METADATA TABLE ONLY (sentence text + start/end times +
speaker labels). It does NOT require the matching audio .wav files to
exist yet -- audio is only needed later, as a separate step, to actually
cut clips using these timestamps. This script also records whether a
matching wav file is already available or not, so you know which rows are
ready to extract embeddings from right away vs. which still need audio.

Chain of joins:
  1. Clean-transcript filename -> old_subj_id (e.g.
     "clean_20090713_091520_LDC_92695_CH02_trans.txt" -> old_subj_id=92695)
  2. pin_lookup_table2.txt: old_subj_id -> new_subj_id
  3. mixer6_demographics.csv: new_subj_id -> gender, dialect_region, age,
     height, weight, smoker
"""

import csv
from pathlib import Path

import pandas as pd


CLEAN_TRANSCRIPT_DIR = Path("mixer6/00cleanTranscripts")
PIN_LOOKUP_FILE = Path("pin_lookup_table2.txt")
DEMOGRAPHICS_CSV = Path("mixer6_demographics.csv")
WAV_DIR = Path("mixer6/wav")            # LDC/HIVEROOM-style wav files
STOPS_WAV_DIR = Path("mixer6/stops_wav")  # _stops-style wav files (newID_date_oldID_gender.wav)

OUTPUT_METADATA_CSV = Path("metadata_mixer6_sentences.csv")


# Dialect region grouping (same placeholder as before)


US_CENSUS_REGIONS = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South",
    "SC": "South", "VA": "South", "DC": "South", "WV": "South", "AL": "South",
    "KY": "South", "MS": "South", "TN": "South", "AR": "South", "LA": "South",
    "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "UT": "West", "WY": "West", "AK": "West", "CA": "West",
    "HI": "West", "OR": "West", "WA": "West",
}


def state_to_region(state_born: str) -> str:
    if not isinstance(state_born, str):
        return "Unknown"
    return US_CENSUS_REGIONS.get(state_born.strip().upper(), "Unknown/Non-US")


# Load pin lookup + demographics (same as before)


def load_pin_lookup(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.split("_") if p != ""]
            if len(parts) < 3:
                continue
            old_id, new_id, gender = parts[0], parts[1], parts[2]
            if old_id == "XXXXX":
                continue
            if gender not in ("M", "F"):
                gender = None
            rows.append({"old_subj_id": old_id, "new_subj_id": new_id})
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} usable rows from PIN lookup table")
    return df


def load_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["new_subj_id"] = df["subjid"].astype(str)
    if "state_born" in df.columns and "cntry_born" in df.columns:
        df["dialect_region"] = df.apply(
            lambda row: state_to_region(row["state_born"])
            if row["cntry_born"] == "USA" else "Non-US-born",
            axis=1,
        )
    return df


def build_speaker_lookup(pin_lookup: pd.DataFrame,
                          demographics: pd.DataFrame) -> pd.DataFrame:
    merged = pin_lookup.merge(
        demographics[["new_subj_id", "sex", "dialect_region",
                       "age", "ht_cm", "wt_kg", "smoker"]],
        on="new_subj_id", how="left",
    )
    return merged.set_index("old_subj_id")


# Parse clean transcripts + filenames

def parse_transcript_filename(file_stem: str) -> dict:
    """'clean_20090713_091520_LDC_92695_CH02_trans' -> components.
    Handles any session-type token (LDC, HIVEROOM, etc.)."""
    # strip "clean_" prefix and "_trans" suffix
    core = file_stem
    if core.startswith("clean_"):
        core = core[len("clean_"):]
    if core.endswith("_trans"):
        core = core[:-len("_trans")]

    parts = core.split("_")
    if len(parts) < 5:
        return None
    return {
        "date": parts[0], "time": parts[1], "session_type": parts[2],
        "old_subj_id": parts[3], "channel": parts[4],
    }


def parse_clean_transcript(file_path: Path) -> list[dict]:
    utterances = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader):
            if len(row) < 3:
                continue
            try:
                start, end = float(row[0]), float(row[1])
            except ValueError:
                continue
            text = row[2].strip()
            utterances.append({"sent_index": line_num, "start": start,
                                "end": end, "text": text})
    return utterances

 
# Find available audio, checking BOTH naming conventions,
# matched by (old_subj_id, date) together (not old_subj_id alone -- a
# speaker's _stops wav and a given transcript may be from different
# recording sessions/dates, and using the wrong date's audio with a
# transcript's timestamps would cut completely wrong content).


STOPS_WAV_PATTERN_RE = None  # set below after re import at module load


def scan_available_wavs(wav_dir: Path, stops_wav_dir: Path) -> dict:
    """Build a lookup of (old_subj_id, date) -> wav path, scanning BOTH
    naming conventions."""
    import re
    stops_pattern = re.compile(r"^(\d+)_(\d{8})_(\d+)_([MFU])\.wav$")
    session_pattern = re.compile(r"^(\d{8})_(\d{6})_([A-Z]+)_(\d+)_(CH\d+)\.wav$")

    wav_lookup = {}

    if stops_wav_dir.exists():
        for f in sorted(stops_wav_dir.glob("*.wav")):
            m = stops_pattern.match(f.name)
            if m:
                new_subj_id, date, old_subj_id, gender = m.groups()
                wav_lookup[(old_subj_id, date)] = {"path": f, "source": "_stops naming"}

    if wav_dir.exists():
        for f in sorted(wav_dir.glob("*.wav")):
            m = session_pattern.match(f.name)
            if m:
                date, time, session_type, old_subj_id, channel = m.groups()
                # don't overwrite a _stops match if one already exists
                wav_lookup.setdefault((old_subj_id, date),
                                       {"path": f, "source": f"{session_type} naming"})

    return wav_lookup


def main():
    print("Loading PIN lookup table...")
    pin_lookup = load_pin_lookup(PIN_LOOKUP_FILE)

    print("Loading demographics...")
    demographics = load_demographics(DEMOGRAPHICS_CSV)

    print("Building speaker lookup...")
    speaker_lookup = build_speaker_lookup(pin_lookup, demographics)
    print()

    print("Scanning available audio files (both naming conventions)...")
    wav_lookup = scan_available_wavs(WAV_DIR, STOPS_WAV_DIR)
    print(f"  Found {len(wav_lookup)} usable (old_subj_id, date) audio files\n")

    transcript_files = sorted(CLEAN_TRANSCRIPT_DIR.glob("clean_*_trans.txt"))
    print(f"Found {len(transcript_files)} clean transcript files.\n")

    all_rows = []
    unmatched_speakers = []

    for transcript_path in transcript_files:
        parsed = parse_transcript_filename(transcript_path.stem)
        if parsed is None:
            print(f"  [SKIP] Unrecognized filename: {transcript_path.name}")
            continue

        old_subj_id = parsed["old_subj_id"]
        if old_subj_id not in speaker_lookup.index:
            unmatched_speakers.append(old_subj_id)
            print(f"  [WARN] old_subj_id={old_subj_id} not found in PIN "
                  f"lookup table ({transcript_path.name})")
            continue

        speaker_info = speaker_lookup.loc[old_subj_id]

        # NEW: check both naming conventions, matched by (old_subj_id, date)
        wav_key = (old_subj_id, parsed["date"])
        wav_match = wav_lookup.get(wav_key)
        wav_available = wav_match is not None
        wav_path_str = str(wav_match["path"]) if wav_match else ""
        wav_source = wav_match["source"] if wav_match else ""

        session_stem = (f"{parsed['date']}_{parsed['time']}_"
                         f"{parsed['session_type']}_{old_subj_id}_{parsed['channel']}")

        utterances = parse_clean_transcript(transcript_path)
        print(f"  {transcript_path.name}: {len(utterances)} sentences, "
              f"speaker={speaker_info['new_subj_id']}, "
              f"audio available={wav_available}"
              f"{f' (via {wav_source})' if wav_available else ''}")

        for utt in utterances:
            all_rows.append({
                "speaker_id": speaker_info["new_subj_id"],
                "old_subj_id": old_subj_id,
                "session": session_stem,
                "session_date": parsed["date"],
                "session_type": parsed["session_type"],
                "channel": parsed["channel"],
                "sent_index": utt["sent_index"],
                "sentence_text": utt["text"],
                "gender": speaker_info["sex"],
                "dialect_region": speaker_info["dialect_region"],
                "age": speaker_info["age"],
                "height_cm": speaker_info["ht_cm"],
                "weight_kg": speaker_info["wt_kg"],
                "smoker": speaker_info["smoker"],
                "start": utt["start"],
                "end": utt["end"],
                "wav_available": wav_available,
                "wav_path": wav_path_str,
                "wav_source": wav_source,
            })

    metadata_df = pd.DataFrame(all_rows)
    metadata_df.to_csv(OUTPUT_METADATA_CSV, index=False)

    print(f"\nDone.")
    print(f"  Saved {len(metadata_df)} sentence rows to {OUTPUT_METADATA_CSV}")
    print(f"  Unique speakers: {metadata_df['speaker_id'].nunique()}")
    if not metadata_df.empty:
        print(f"  Rows with audio already available: "
              f"{metadata_df['wav_available'].sum()} / {len(metadata_df)}")
    if unmatched_speakers:
        print(f"  {len(set(unmatched_speakers))} old_subj_ids not found in "
              f"PIN lookup: {sorted(set(unmatched_speakers))}")


if __name__ == "__main__":
    main()
