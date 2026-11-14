"""
cut_clips_from_sentence_metadata.py

Takes metadata_mixer6_sentences.csv (already built, already confirmed by
Eleanor as the dataset to move forward with -- 196 speakers, 140,887
sentences with matching audio) and actually cuts the audio clips, using
the CORRECT sentence boundaries from the clean transcripts (not the
pause-guessed _stops boundaries from the earlier method).

Only processes rows where wav_available == True. Rows without audio are
skipped (they stay in your metadata for reference, but can't be turned
into clips yet).

Output: one short .wav clip per sentence, plus a final metadata CSV
(metadata_mixer6_sentences_with_clips.csv) that includes the clip_path
column -- this is what you'll feed into extract_embeddings_mixer6.py next.
"""

from pathlib import Path

import pandas as pd
from scipy.io import wavfile



INPUT_METADATA = Path("metadata_mixer6_sentences.csv")
OUTPUT_CLIPS_DIR = Path("mixer6_clips_sentences")
OUTPUT_METADATA_CSV = Path("metadata_mixer6_sentences_with_clips.csv")
EXISTING_OUTPUT_TO_RESUME = OUTPUT_METADATA_CSV  # same file -- resumes from
                                                   # wherever the last run
                                                   # (or checkpoint) left off

CHECKPOINT_EVERY = 5000  # save progress periodically, given ~140k clips to cut


def main():
    print("Loading metadata...")
    df = pd.read_csv(INPUT_METADATA)

    matched = df[df["wav_available"] == True].copy()
    print(f"Total rows: {len(df)}")
    print(f"Rows with matching audio (wav_available=True): {len(matched)}")
    print(f"Unique speakers: {matched['speaker_id'].nunique()}\n")

    OUTPUT_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # RESUME SUPPORT. If a previous run already produced output,
    # figure out which sessions (wav_path groups) are already fully done,
    # and skip them entirely this time -- so an interrupted run can be
    # continued instead of redone from scratch.
    already_done_sessions = set()
    existing_rows = []
    if EXISTING_OUTPUT_TO_RESUME.exists():
        existing_df = pd.read_csv(EXISTING_OUTPUT_TO_RESUME)
        already_done_sessions = set(existing_df["wav_path"].unique())
        existing_rows = existing_df.to_dict("records")
        print(f"RESUMING: found {len(existing_df)} clips already cut from "
              f"{len(already_done_sessions)} previously completed sessions.")
        print(f"These will be skipped this run.\n")


    grouped = matched.groupby("wav_path")
    remaining_groups = [(wav_path_str, rows) for wav_path_str, rows in grouped
                         if wav_path_str not in already_done_sessions]
    print(f"Total sessions: {len(grouped)}")
    print(f"Already done (skipped): {len(grouped) - len(remaining_groups)}")
    print(f"Remaining to process: {len(remaining_group)}\n")

    all_rows = list(existing_rows)  # start from what's already done
    failed_rows = []
    sessions_processed = 0
    total_sessions = len(remaining_groups)

    for wav_path_str, session_rows in remaining_groups:
        wav_path = Path(wav_path_str)
        sessions_processed += 1

        if not wav_path.exists():
            print(f"  [SKIP] Session {sessions_processed}/{total_sessions}: "
                  f"file not found: {wav_path}")
            for _, row in session_rows.iterrows():
                failed_rows.append({**row.to_dict(), "error": "wav_file_not_found"})
            continue

        try:
            sample_rate, audio = wavfile.read(str(wav_path))
        except Exception as e:
            print(f"  [SKIP] Session {sessions_processed}/{total_sessions}: "
                  f"failed to read {wav_path}: {e}")
            for _, row in session_rows.iterrows():
                failed_rows.append({**row.to_dict(), "error": f"read_error: {e}"})
            continue

        # the _stops wav files begin EXACTLY at the first
        # reading-task sentence, not at the start of the full session
        # recording. Clean-transcript timestamps are relative to the full
        # (longer) recording, so we subtract each session's own first
        # sentence start time from every timestamp in that session,
        # mapping them onto the _stops wav's own timeline. Confirmed with
        # an exact (0.00s difference) match on real data before rolling
        # this out.
        session_offset = session_rows["start"].min()

        if sessions_processed % 20 == 0 or sessions_processed == total_sessions:
            print(f"  Session {sessions_processed}/{total_sessions}: "
                  f"{wav_path.name} ({len(session_rows)} sentences, "
                  f"offset={session_offset:.2f}s)")

        for _, row in session_rows.iterrows():
            adjusted_start = row["start"] - session_offset
            adjusted_end = row["end"] - session_offset

            start_sample = int(adjusted_start * sample_rate)
            end_sample = int(adjusted_end * sample_rate)

            if end_sample <= start_sample or end_sample > len(audio):
                failed_rows.append({**row.to_dict(), "error": "invalid_time_range"})
                continue

            clip_audio = audio[start_sample:end_sample]

            clip_filename = f"{row['session']}_sent{int(row['sent_index']):04d}.wav"
            clip_path = OUTPUT_CLIPS_DIR / clip_filename

            try:
                wavfile.write(str(clip_path), sample_rate, clip_audio)
            except Exception as e:
                failed_rows.append({**row.to_dict(), "error": f"write_error: {e}"})
                continue

            row_dict = row.to_dict()
            row_dict["clip_path"] = str(clip_path)
            all_rows.append(row_dict)

        # periodic checkpoint save
        if len(all_rows) > 0 and len(all_rows) % CHECKPOINT_EVERY < len(session_rows):
            checkpoint_df = pd.DataFrame(all_rows)
            checkpoint_df.to_csv(OUTPUT_METADATA_CSV, index=False)
            print(f"    [checkpoint saved: {len(all_rows)} clips so far]")

    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(OUTPUT_METADATA_CSV, index=False)

    print(f"\nDone.")
    print(f"  Successfully cut {len(final_df)} clips")
    print(f"  Saved to {OUTPUT_METADATA_CSV}")
    print(f"  Audio clips in {OUTPUT_CLIPS_DIR}/")

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv("clip_cutting_failures.csv", index=False)
        print(f"  {len(failed_df)} rows failed -- logged to clip_cutting_failures.csv")


if __name__ == "__main__":
    main()
