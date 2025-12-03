# run_sweep.py
import sys

def get_mode_for_task(task_id: int):
    """Returns the TRAIN_MODE for a given 1-indexed SLURM Task ID."""
    # Phase 1: Data-Only (Runs 1-10)
    if 1 <= task_id <= 10:
        return 'data'
    
    # Phase 2: Teacher (Runs 11-20)
    elif 11 <= task_id <= 20:
        return 'teacher'
        
    # Phase 3: JVP (Runs 21-30)
    elif 21 <= task_id <= 30:
        return 'jvp'
        
    # Phase 4: Final 1000-Epoch Comparison (Runs 31-33)
    elif task_id == 31:
        return 'teacher'
    elif task_id == 32:
        return 'jvp'
    elif task_id == 33:
        return 'data'
        
    # Default safety
    return 'data'
    

if __name__ == "__main__":
    try:
        # SLURM_ARRAY_TASK_ID is passed as the first argument from the shell script
        task_id = int(sys.argv[1])
        print(get_mode_for_task(task_id))
    except Exception as e:
        # Default if run outside of SLURM
        print("data")