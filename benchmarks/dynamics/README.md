<!-- markdownlint-disable MD029 -->
# Dynamics Benchmarks

This directory contains benchmark scripts for molecular dynamics integrators
and geometry optimizers.

## Benchmark Scripts

The benchmarks are organized into 4 modular scripts:

### Single-System Benchmarks

1. **`benchmark_md_single.py`** - Single-system MD benchmarks

   - Methods: VelocityVerlet, Langevin, NoseHoover, NPT, NPH
   - Output: CSV files with 11 columns (single-system schema)

   ```bash
   python benchmark_md_single.py --backend both --config benchmark_config.yaml
   ```

2. **`benchmark_opt_single.py`** - Single-system optimization benchmarks

   - Methods: FIRE
   - Output: CSV files with convergence metrics

   ```bash
   python benchmark_opt_single.py --backend both --config benchmark_config.yaml
   ```

### Batched Benchmarks

3. **`benchmark_md_batch.py`** - Batched MD benchmarks

   - Methods: VelocityVerlet, Langevin
   - Output: CSV files with 14 columns (batched schema)

   ```bash
   python benchmark_md_batch.py --backend nvalchemiops --config benchmark_config.yaml
   ```

4. **`benchmark_opt_batch.py`** - Batched optimization benchmarks

   - Methods: FIRE optimizer
   - Output: CSV files with batch throughput metrics

   ```bash
   python benchmark_opt_batch.py --backend nvalchemiops --config benchmark_config.yaml
   ```

## Shared Utilities

- **`shared_utils.py`** - Common utilities used by all benchmark scripts
  - `BenchmarkResult` dataclass with CSV export
  - `create_lj_system()` - FCC argon system generator
  - `write_results_csv()` - CSV writer with schema detection
  - `get_gpu_sku()` - GPU identification
  - Unit conversion constants

## Configuration

All benchmarks are configured via **`benchmark_config.yaml`**:

```yaml
# Single-system MD
md_single:
  enabled: true
  system_sizes: [256, 512, 1024, 2048, 4096]
  integrators:
    velocity_verlet:
      steps: 10000
      dt: 0.001  # fs
      warmup_steps: 100
    langevin:
      steps: 10000
      temperature: 300.0  # K
      friction: 0.01  # 1/fs

# Single-system optimization
opt_single:
  enabled: true
  system_sizes: [256, 512, 1024, 2048]
  optimizers:
    fire:
      max_steps: 1000
      force_tolerance: 0.01  # eV/Å

# Batched MD
md_batch:
  enabled: true
  system_sizes: [256, 512, 1024]
  batch_sizes: [1, 2, 4, 8, 16, 32]

# Batched optimization
opt_batch:
  enabled: true
  system_sizes: [256, 512]
  batch_sizes: [1, 2, 4, 8, 16]
```

## Output Format

### CSV Schemas

**Single-system schema (11 columns):**

```bash
backend,method,num_atoms,ensemble,steps,dt,warmup_steps,avg_step_time_ms,
total_time_s,throughput_steps_per_s,throughput_atom_steps_per_s
```

**Batched schema (14 columns):**

```bash
backend,method,num_atoms,ensemble,batch_size,total_atoms,steps,dt,warmup_steps,
avg_step_time_ms,total_time_s,throughput_steps_per_s,throughput_atom_steps_per_s,
batch_throughput_system_steps_per_s
```

### Output Location

CSV files are saved to `../../docs/benchmarks/benchmark_results/` with naming convention:

```bash
dynamics_{md|opt}_{single|batch}_{backend}_{gpu_sku}.csv
```

Examples:

- `dynamics_md_single_nvalchemiops_rtx4090.csv`
- `dynamics_md_batch_nvalchemiops_rtx4090.csv`

## Documentation

Benchmark results are documented in `../../docs/benchmarks/`:

- `dynamics_md_single.md` - Single-system MD documentation
- `dynamics_opt_single.md` - Single-system optimization documentation
- `dynamics_md_batch.md` - Batched MD documentation
- `dynamics_opt_batch.md` - Batched optimization documentation

### Generating Plots

Plots are generated from CSV files using:

```bash
cd ../../docs/benchmarks
python generate_plots.py
```

This creates plots in `docs/benchmarks/_static/` which are embedded in the documentation.

## Running All Benchmarks

To run the complete benchmark suite:

```bash
# Single-system benchmarks
python benchmark_md_single.py --backend both
python benchmark_opt_single.py --backend both

# Batched benchmarks (when enabled in config)
python benchmark_md_batch.py --backend nvalchemiops
python benchmark_opt_batch.py --backend nvalchemiops

# Generate plots
cd ../../docs/benchmarks
python generate_plots.py

# Build documentation
cd ../..
make html  # or your documentation build command
```

## System Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 4090, A100, H100)
- **Python:** 3.8+
- **Dependencies:**
  - nvalchemiops (with warp backend)
  - PyTorch
  - NumPy, pandas, matplotlib (for plot generation)

## Performance Tips

1. **GPU Selection:** Use `CUDA_VISIBLE_DEVICES` to select specific GPU
2. **System Size:** Larger systems better utilize GPU parallelism
3. **Batch Size:** Increase until GPU saturates (monitor memory usage)
4. **Neighbor List:** Adjust rebuild interval in config for optimal performance
5. **Warmup:** Always include warmup steps to exclude kernel compilation overhead

## Contributing

To add new benchmarks:

1. Follow existing script patterns (shared_utils, CSV output, CLI args)
2. Update `benchmark_config.yaml` with new sections
3. Add plot generation to `docs/benchmarks/generate_plots.py`
4. Create documentation page in `docs/benchmarks/`
5. Update this README

## Questions?

- See documentation: `../../docs/benchmarks/`
- Check configuration: `benchmark_config.yaml`
- Review shared utilities: `shared_utils.py`
