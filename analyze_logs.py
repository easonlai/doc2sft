import json
import os
from collections import Counter

LOG_FILE = "logs/traceability.jsonl"

def analyze_telemetry():
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: Cannot find {LOG_FILE}. Ensure the path is correct.")
        return

    total_retries = 0
    total_quarantines = 0
    error_types = Counter()
    pydantic_issues = Counter()
    quarantined_chunks = []

    print("Scanning enterprise telemetry logs...\n")

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                log = json.loads(line)
                stage = log.get("pipeline_stage")
                
                # Analyze Retry Events
                if stage == "retry_engine":
                    total_retries += 1
                    err_type = log.get("error_type", "UnknownError")
                    error_types[err_type] += 1
                    
                    # Extract the exact schema violations
                    if "pydantic_errors" in log:
                        for p_err in log["pydantic_errors"]:
                            # Map the specific location in the JSON that failed
                            loc = " -> ".join([str(x) for x in p_err.get("loc", ["unknown_location"])])
                            msg = p_err.get("msg", "Unknown violation")
                            
                            # Format: [messages -> 1 -> role] Field required
                            formatted_issue = f"[{loc}] {msg}"
                            pydantic_issues[formatted_issue] += 1

                # Analyze Quarantine Events
                elif stage == "quarantine":
                    total_quarantines += 1
                    file_name = log.get("file_name", "Unknown File")
                    chunk_idx = log.get("chunk_index", "?")
                    quarantined_chunks.append(f"Chunk {chunk_idx} from '{file_name}'")

            except json.JSONDecodeError:
                continue

    # ==========================================
    # PRINT DIAGNOSTIC REPORT
    # ==========================================
    print("=" * 60)
    print("📊 DOC2SFT TELEMETRY DIAGNOSTICS REPORT")
    print("=" * 60)
    print(f"Total Self-Healing Retries Triggered: {total_retries}")
    print(f"Total Chunks Quarantined (Data Lost): {total_quarantines}")
    
    print("\n🔥 ROOT CAUSE ERROR TYPES:")
    if not error_types:
        print("  - No errors detected!")
    for err, count in error_types.most_common():
        print(f"  - {err}: {count} occurrences")
        
    if pydantic_issues:
        print("\n🚨 SPECIFIC PYDANTIC SCHEMA FAILURES (Top 5):")
        for issue, count in pydantic_issues.most_common(5):
            print(f"  - {count}x: {issue}")
            
    if quarantined_chunks:
        print("\n💀 QUARANTINED CHUNKS (Failed 3x in a row):")
        for qc in quarantined_chunks[:5]:
            print(f"  - {qc}")
        if len(quarantined_chunks) > 5:
            print(f"  ...and {len(quarantined_chunks) - 5} more.")
    print("=" * 60)

if __name__ == "__main__":
    analyze_telemetry()