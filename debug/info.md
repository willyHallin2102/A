# Debug Scripts 

Initialize the debugging or test run scripts initially the environment is required to be activated `cd path-to:/A` and run `source venv/bin/activate` to activate the environment within the terminal. Thereafter, the run `cd debug` which direct to where each script is defined.

## Loader

## Maths



```bash
# Basic correctness test
python coords_cl

# # 1️⃣ Basic correctness test
# python coords_cli.py --mode roundtrip
```


# # 2️⃣ Random vector test with float64
# python coords_cli.py --mode random --samples 10000 --dtype float64

# # 3️⃣ Rotation test (angle combination)
# python coords_cli.py --mode rotate --phi0 45 --theta0 60 --phi1 120 --theta1 90

# # 4️⃣ Benchmark performance
# python coords_cli.py --mode bench --samples 500000 --dtype float32
