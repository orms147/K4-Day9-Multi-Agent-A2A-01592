import os
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader import DataLoader
from agents import CoordinatorAgent

def process_single_case(filepath, output_dir, trace_file, data_loader):
    filename = os.path.basename(filepath)
    print(f"\\n--- Processing {filename} ---")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        case_json = json.load(f)
        
    start_time = time.time()
    # Khởi tạo một instance CoordinatorAgent mới cho mỗi luồng để tránh xung đột state (dù hiện tại state-less)
    coordinator = CoordinatorAgent(data_loader)
    
    try:
        result_json = coordinator.run_case(case_json)
        
        out_filepath = os.path.join(output_dir, filename)
        with open(out_filepath, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully processed {filename}. Status: {result_json.get('case_assessment', {}).get('case_status')}")
        
        duration = time.time() - start_time
        trace_entry = {
            "case_id": case_json.get("case_id"),
            "status": "success",
            "duration_seconds": round(duration, 2),
            "output": result_json
        }
        with open(trace_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace_entry, ensure_ascii=False) + "\\n")
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        duration = time.time() - start_time
        trace_entry = {
            "case_id": case_json.get("case_id"),
            "status": "error",
            "error_message": str(e),
            "duration_seconds": round(duration, 2)
        }
        with open(trace_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace_entry, ensure_ascii=False) + "\\n")

def main():
    input_dir = "input"
    output_dir = "output"
    trace_file = "trace.jsonl"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if os.path.exists(trace_file):
        os.remove(trace_file)
        
    data_loader = DataLoader("data")
    data_loader.load_all()
    
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    print(f"Found {len(input_files)} cases to process. Running concurrently...")
    
    # Dùng ThreadPoolExecutor để chạy song song (max 10 threads)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_case, fp, output_dir, trace_file, data_loader) for fp in input_files]
        for future in as_completed(futures):
            pass # exceptions are caught inside process_single_case

    print("\\nProcessing complete. Check the 'output' folder and 'trace.jsonl'.")

if __name__ == "__main__":
    main()
