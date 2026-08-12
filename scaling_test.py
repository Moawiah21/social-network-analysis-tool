"""
scaling_test.py
===============
Stress test to find the maximum graph size the manual implementations
can handle before becoming too slow or crashing.

Pushes graph sizes from 100 all the way to 10,000 nodes.
Betweenness is the bottleneck — the script reports how long each size
takes so you can see exactly where the program starts struggling.

To run:
    python scaling_test.py

You can stop the script at any time with Ctrl+C if betweenness
becomes too slow. Results collected so far will still be saved.

To submit with code: change BETWEENNESS_THRESHOLD to e.g. 500
so markers do not have to wait. Remove that line to run the full test.
"""

import time
import csv
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from degree_centrality import calculate_degree_centrality
from pagerank import calculate_pagerank
from betweenness_centrality import calculate_betweenness_centrality
from closeness_centrality import calculate_closeness_centrality


# ── Configuration ─────────────────────────────────────────────────────────────

# Graph sizes to test — pushing well beyond 1000 to find the limit
GRAPH_SIZES = [100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000]

# Edge probability — kept low so graphs are sparse and connected
EDGE_PROBABILITY = 0.006

SEED = 42
OUTPUT_FILE = "scaling_results.csv"

# ── Set this to e.g. 500 when submitting code so markers don't wait ──────────
# Remove this line (set to None) to run the full experiment
BETWEENNESS_THRESHOLD = None  # e.g. change to 500 to skip above 500 nodes


# ── Helpers ───────────────────────────────────────────────────────────────────
def measure(func, *args):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    return result, elapsed


def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds * 1000:.4f} ms"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.4f} s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"


def build_graph(n):
    """Build a connected graph of n nodes."""
    if n == 34:
        return nx.karate_club_graph()

    seed = SEED
    attempts = 0
    while True:
        G = nx.erdos_renyi_graph(n, EDGE_PROBABILITY, seed=seed)
        if nx.is_connected(G):
            return G
        seed += 1
        attempts += 1
        # After many failed attempts, use Barabasi-Albert which is always connected
        if attempts > 100:
            print(f"    (switching to Barabasi-Albert for connectivity at n={n})")
            return nx.barabasi_albert_graph(n, 3, seed=SEED)


# ── Main experiment ───────────────────────────────────────────────────────────
def run_experiment():
    print("=" * 90)
    print("STRESS TEST — How large a graph can the manual implementations handle?")
    print("=" * 90)
    print(f"Graph sizes tested: {GRAPH_SIZES}")
    print(f"Betweenness threshold: {BETWEENNESS_THRESHOLD if BETWEENNESS_THRESHOLD else 'None — running at all sizes'}")
    print(f"Note: Press Ctrl+C at any time to stop. Results so far will be saved.")
    print()

    header = f"{'Nodes':>7} {'Edges':>8} {'Degree':>12} {'PageRank':>14} {'Closeness':>14} {'Betweenness':>16}  Status"
    print(header)
    print("-" * 90)

    results = []

    for n in GRAPH_SIZES:
        try:
            print(f"  Building graph: {n} nodes...", end=" ", flush=True)
            G = build_graph(n)
            edges = G.number_of_edges()
            print(f"{edges} edges", flush=True)

            # Degree
            print(f"    Degree...", end=" ", flush=True)
            _, t_degree = measure(calculate_degree_centrality, G)
            print(f"{format_time(t_degree)}", flush=True)

            # PageRank
            print(f"    PageRank...", end=" ", flush=True)
            _, t_pagerank = measure(calculate_pagerank, G)
            print(f"{format_time(t_pagerank)}", flush=True)

            # Closeness
            print(f"    Closeness...", end=" ", flush=True)
            _, t_closeness = measure(calculate_closeness_centrality, G)
            print(f"{format_time(t_closeness)}", flush=True)

            # Betweenness
            if BETWEENNESS_THRESHOLD and n > BETWEENNESS_THRESHOLD:
                t_betweenness = None
                betweenness_display = "skipped"
                status = "betweenness skipped"
                print(f"    Betweenness... skipped (above threshold)")
            else:
                print(f"    Betweenness... (this may take a while at large sizes)", flush=True)
                t_start = time.perf_counter()
                _, t_betweenness = measure(calculate_betweenness_centrality, G)
                betweenness_display = format_time(t_betweenness)
                status = "OK"
                print(f"    Betweenness: {betweenness_display}", flush=True)

            # Summary row
            row = (
                f"{n:>7} "
                f"{edges:>8} "
                f"{format_time(t_degree):>12} "
                f"{format_time(t_pagerank):>14} "
                f"{format_time(t_closeness):>14} "
                f"{betweenness_display:>16}  {status}"
            )
            print(f"\n  RESULT: {row}")
            print("-" * 90)

            results.append({
                "nodes": n,
                "edges": edges,
                "degree_s": round(t_degree, 6),
                "pagerank_s": round(t_pagerank, 6),
                "closeness_s": round(t_closeness, 6),
                "betweenness_s": round(t_betweenness, 6) if t_betweenness is not None else "skipped",
                "status": status
            })

        except KeyboardInterrupt:
            print(f"\n\n  Stopped by user at n={n}.")
            print(f"  Saving results collected so far...")
            break
        except MemoryError:
            print(f"\n  MEMORY ERROR at n={n} — graph too large for available RAM.")
            results.append({
                "nodes": n, "edges": "N/A",
                "degree_s": "N/A", "pagerank_s": "N/A",
                "closeness_s": "N/A", "betweenness_s": "N/A",
                "status": "MEMORY ERROR"
            })
            break
        except Exception as e:
            print(f"\n  ERROR at n={n}: {e}")
            results.append({
                "nodes": n, "edges": "N/A",
                "degree_s": "N/A", "pagerank_s": "N/A",
                "closeness_s": "N/A", "betweenness_s": "N/A",
                "status": f"ERROR: {e}"
            })
            continue

    # ── Print final summary table ─────────────────────────────────────────────
    print()
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    summary_header = f"{'Nodes':>7} {'Edges':>8} {'Degree':>12} {'PageRank':>14} {'Closeness':>14} {'Betweenness':>16}"
    print(summary_header)
    print("-" * 90)
    for r in results:
        def fmt(v):
            if v in ("N/A", "skipped"):
                return v
            try:
                return format_time(float(v))
            except:
                return str(v)

        print(
            f"{r['nodes']:>7} "
            f"{str(r['edges']):>8} "
            f"{fmt(r['degree_s']):>12} "
            f"{fmt(r['pagerank_s']):>14} "
            f"{fmt(r['closeness_s']):>14} "
            f"{fmt(r['betweenness_s']):>16}"
        )
    print("-" * 90)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if results:
        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["nodes", "edges", "degree_s", "pagerank_s", "closeness_s", "betweenness_s", "status"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {OUTPUT_FILE}")

    print()
    print("Dissertation notes:")
    print("  - The point where betweenness exceeds ~60 seconds is the practical limit")
    print("  - Degree and PageRank can likely handle 10,000+ nodes without issue")
    print("  - Any MemoryError tells you the maximum graph size for your machine")
    print("  - Screenshot this output and the summary table for your dissertation")


if __name__ == "__main__":
    run_experiment()