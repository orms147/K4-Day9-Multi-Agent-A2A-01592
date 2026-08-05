import os
import glob
import json
import time
from data_loader import DataLoader
from agents import CoordinatorAgent

def main():
    # Khởi tạo thư mục
    input_dir = "input"
    output_dir = "output"
    trace_file = "trace.jsonl"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Xóa file trace cũ nếu có để ghi đợt chạy mới nhất
    if os.path.exists(trace_file):
        os.remove(trace_file)
        
    # Load data
    data_loader = DataLoader("data")
    data_loader.load_all()
    
    # Khởi tạo Agent
    coordinator = CoordinatorAgent(data_loader)
    
    # Lấy danh sách 50 file input
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    
    print(f"Found {len(input_files)} cases to process.")
    
    for filepath in input_files:
        filename = os.path.basename(filepath)
        print(f"\\n--- Processing {filename} ---")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            case_json = json.load(f)
            
        start_time = time.time()
        
        try:
            # Chạy qua hệ thống multi-agent
            result_json = coordinator.run_case(case_json)
            
            # Lưu file output
            out_filepath = os.path.join(output_dir, filename)
            with open(out_filepath, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
                
            print(f"Successfully processed {filename}. Status: {result_json.get('case_assessment', {}).get('case_status')}")
            
            # Lưu log vào trace.jsonl
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

    print("\\nProcessing complete. Check the 'output' folder and 'trace.jsonl'.")

if __name__ == "__main__":
    main()
