# precompute_graphs_mgpu.py
import os
import multiprocessing as mp
from multiprocessing import Process

# Set spawn method for CUDA compatibility
mp.set_start_method('spawn', force=True)

# --- CONFIG ---
BENCHMARKS = ['all']
GPU_IDS    = [0, 1, 2, 3, 4, 5, 6, 7]   # edit to a subset if needed
DATASET_KW = dict(
    enable_pdb_processing=True,
    force_pdb_processing=False,   # set True if you want to overwrite any existing .pt
    validate_on_init=True,
    skip_invalid_files=True,
    esm_use_cpu=False,            # we're using GPUs; set True if you want CPU-only
    cache_size=0,                 # disable in-RAM cache for bulk jobs
)

def process_split(benchmark: str, train: bool, rank: int, world: int):
    """
    Process one split (train or test) for a given benchmark.
    Shards work by dataset index: each worker handles indices i where i % world == rank.
    """
    try:
        # CRITICAL: Pin this worker to a single GPU BEFORE importing anything torch-related
        gpu_id = GPU_IDS[rank % len(GPU_IDS)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        print(f"[rank {rank}] Using GPU {gpu_id} for benchmark {benchmark}")
        print(f"[rank {rank}] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        # Import torch and protein_dataset AFTER setting CUDA_VISIBLE_DEVICES
        import torch
        from data.protein_dataset import ProteinDataset
        
        torch.cuda.empty_cache()  # Clear any existing cache
        
        # Verify we're using the correct GPU
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            print(f"[rank {rank}] Using CUDA device {device}: {device_name}")
        else:
            print(f"[rank {rank}] Warning: CUDA not available!")
        
        # Additional backup: explicitly set the device in torch
        if torch.cuda.is_available() and len(GPU_IDS) > 1:
            # Since we set CUDA_VISIBLE_DEVICES, device 0 should be our assigned GPU
            torch.cuda.set_device(0)
            print(f"[rank {rank}] Set torch device to 0 (which maps to physical GPU {gpu_id})")
        
        pdb_dir   = f'/data1/GNN_AS/GNN_AS_graph_dataset/benchmarks/ready_data/{benchmark}/protein_pdb_files'
        cache_dir = f'/data1/GNN_AS/GNN_AS_graph_dataset/benchmarks/ready_data/{benchmark}/protein_graphs_esmc_600m'

        # ds = ProteinDataset(
        #     train=train,
        #     pdb_path=pdb_dir,
        #     cache_path=cache_dir,
        #     **DATASET_KW,
        # )

        ds = ProteinDataset(
            train=train,                 # split logic still runs; fine since we iterate both splits below
            pdb_path=pdb_dir,
            cache_path=cache_dir,
            enable_pdb_processing=True, # allow on-the-fly PDB → graph
            force_pdb_processing=True,  # ignore any pre-existing .pt set; process PDBs
            validate_on_init=True,      # quick sanity check of PDBs upfront
            skip_invalid_files=True,    # drop bad PDBs instead of crashing
            esm_use_cpu=False,          # set True to force CPU for ESM if GPUs are busy
            cache_size=0,               # disable in-RAM cache for this bulk job
        )

        ok = fail = 0
        n = len(ds)
        
        print(f"[rank {rank}] Processing {n} files for {benchmark} {'train' if train else 'test'}")

        # Shard by index so workers don't duplicate effort
        my_indices = [i for i in range(n) if i % world == rank]
        print(f"[rank {rank}] Processing {len(my_indices)} files (indices {my_indices[:3]}...)")
        
        for i in my_indices:
            try:
                sample = ds[i]                  # triggers PDB → graph, then CacheManager.save(...)
                if sample.get('load_error'):
                    fail += 1
                else:
                    ok += 1
                    
                # Print progress every 10 files
                if (ok + fail) % 10 == 0:
                    print(f"[rank {rank}] Progress: {ok + fail}/{len(my_indices)} (OK={ok}, Fail={fail})")
                    
            except Exception as e:
                print(f"[rank {rank}] Error processing index {i}: {e}")
                fail += 1

        print(f"[rank {rank}] {benchmark} {'train' if train else 'test'} DONE. OK={ok}, Fail={fail}")
        
    except Exception as e:
        print(f"[rank {rank}] Fatal error in process_split: {e}")
        import traceback
        traceback.print_exc()

def worker(rank: int, world: int):
    """
    Each worker processes *both* splits for all benchmarks on its assigned GPU.
    """
    try:
        print(f"[Worker {rank}] Starting worker process")
        
        for benchmark in BENCHMARKS:
            print(f"[Worker {rank}] Processing benchmark: {benchmark}")
            try:
                # Train split (≈90%)
                process_split(benchmark, train=True,  rank=rank, world=world)
                # Test split (≈10%)
                process_split(benchmark, train=False, rank=rank, world=world)
                print(f"[Worker {rank}] Completed benchmark: {benchmark}")
            except Exception as e:
                print(f"[Worker {rank}] Error processing benchmark {benchmark}: {e}")
                import traceback
                traceback.print_exc()
                
        print(f"[Worker {rank}] Worker completed all benchmarks")
        
    except Exception as e:
        print(f"[Worker {rank}] Fatal worker error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import time
    
    print(f"Starting parallel processing with {len(GPU_IDS)} workers on GPUs: {GPU_IDS}")
    print(f"Processing benchmarks: {BENCHMARKS}")
    
    world = len(GPU_IDS)
    procs = []
    start_time = time.time()
    
    # Start all workers
    for rank in range(world):
        print(f"Starting worker {rank} on GPU {GPU_IDS[rank % len(GPU_IDS)]}")
        p = Process(target=worker, args=(rank, world), daemon=False)
        p.start()
        procs.append(p)
        
    print(f"All {len(procs)} workers started. Waiting for completion...")
    
    # Wait for all workers with progress monitoring
    completed = 0
    while completed < len(procs):
        time.sleep(30)  # Check every 30 seconds
        completed = sum(1 for p in procs if not p.is_alive())
        elapsed = time.time() - start_time
        print(f"Progress: {completed}/{len(procs)} workers completed. Elapsed: {elapsed:.1f}s")
        
        # Check for failed processes
        for i, p in enumerate(procs):
            if not p.is_alive() and p.exitcode != 0:
                print(f"WARNING: Worker {i} exited with code {p.exitcode}")
    
    # Final join to clean up
    for i, p in enumerate(procs):
        p.join()
        print(f"Worker {i} joined with exit code {p.exitcode}")
        
    total_time = time.time() - start_time
    print(f"All workers completed in {total_time:.1f} seconds")
