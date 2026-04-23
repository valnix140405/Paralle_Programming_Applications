import subprocess
import re
import pandas as pd
import sys
import os

def run_command(cmd, cwd=None):
    """Executes a terminal command and returns its standard output."""
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
        if result.returncode != 0:
            print(f"[!] Error running command: {' '.join(cmd)}")
            print(result.stderr)
            return None
        return result.stdout
    except Exception as e:
        print(f"[!] Exception running command: {e}")
        return None

def extract_time(output, keyword):
    """Parses standard output to find the execution time based on a keyword."""
    if not output: return None
    for line in output.split('\n'):
        if keyword in line:
            # Busca un número decimal seguido de la palabra 'seconds'
            matches = re.findall(r"([0-9]+\.[0-9]+)\s*seconds", line)
            if matches:
                return float(matches[0])
    return None

def benchmark_ex3():
    print("\n=======================================================")
    print("--- Benchmarking Exercise 3: Cellular Automaton ---")
    print("=======================================================")
    results = []
    
    print("Running Serial CA (Base)...")
    out = run_command([sys.executable, 'serial_ca.py'], cwd='exercise_3')
    t_serial = extract_time(out, "[Serial]")
    
    if t_serial:
        results.append({'Procesos': 1, 'Tiempo (s)': t_serial, 'Speedup': 1.0, 'Eficiencia': 1.0})
    else:
        print("Failed to run Serial CA. Ensure fetch_firms_data.py was run.")
        return pd.DataFrame()
    
    for procs in [2, 4, 8]:
        print(f"Running Parallel CA with {procs} processes...")
        out = run_command(['mpiexec', '-n', str(procs), sys.executable, 'parallel_ca.py'], cwd='exercise_3')
        t_par = extract_time(out, "[Parallel]")
        if t_par:
            speedup = t_serial / t_par
            efficiency = speedup / procs
            results.append({'Procesos': procs, 'Tiempo (s)': t_par, 'Speedup': speedup, 'Eficiencia': efficiency})
                
    df = pd.DataFrame(results)
    print("\n-> Resultados Exercise 3 (CA):")
    print(df.to_string(index=False, float_format="%.4f"))
    return df

def benchmark_ex4():
    print("\n=======================================================")
    print("--- Benchmarking Exercise 4: Parallel K-Means ---")
    print("=======================================================")
    results = []
    
    print("Running Serial K-Means (Base)...")
    out = run_command([sys.executable, 'serial_kmeans.py'], cwd='exercise_4')
    t_serial = extract_time(out, "[Serial]")
    
    if t_serial:
        results.append({'Procesos': 1, 'Tiempo (s)': t_serial, 'Speedup': 1.0, 'Eficiencia': 1.0})
    else:
        print("Failed to run Serial K-Means. Ensure fetch_covertype.py was run.")
        return pd.DataFrame()
    
    for procs in [2, 4, 8]:
        print(f"Running Parallel K-Means with {procs} processes...")
        out = run_command(['mpiexec', '-n', str(procs), sys.executable, 'parallel_kmeans.py'], cwd='exercise_4')
        t_par = extract_time(out, "[Parallel]")
        if t_par:
            speedup = t_serial / t_par
            efficiency = speedup / procs
            results.append({'Procesos': procs, 'Tiempo (s)': t_par, 'Speedup': speedup, 'Eficiencia': efficiency})
                
    df = pd.DataFrame(results)
    print("\n-> Resultados Exercise 4 (K-Means):")
    print(df.to_string(index=False, float_format="%.4f"))
    return df

if __name__ == '__main__':
    # Make sure we are in the right base directory
    if not os.path.exists('exercise_3') or not os.path.exists('exercise_4'):
        print("Please run this script from the root directory of the project.")
        sys.exit(1)
        
    df3 = benchmark_ex3()
    df4 = benchmark_ex4()
    
    # Save the dataframes for report generation later
    os.makedirs('docs/assets', exist_ok=True)
    if not df3.empty:
        df3.to_csv('docs/assets/benchmark_ex3.csv', index=False)
    if not df4.empty:
        df4.to_csv('docs/assets/benchmark_ex4.csv', index=False)
    
    print("\nBenchmarks completos. Resultados exportados a docs/assets/ como CSV.")
