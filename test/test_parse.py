import random
import typing
from collections import defaultdict

import lps_synthesis.database.ship as lps_ship


# ==== Casos de teste ====

TEST_CASES = {
    "single_int": "10",
    "single_float": "3.75",
    "int_range": "5-10",
    "float_range": "1.5-3.2",
    "list_ints": "1/2/3/4",
    "list_ranges": "10-12/20-22",
    "list_mixed": "5/7-9/12",
    "list_floats": "1.1/2.2/3.3",
    "complex_list": "1/ 3-5/ 7.2/ 10-12",
    "invalid_text": "abc",
    "partially_invalid": "1/ a/ 3-5",
    "empty": "",
    "spaces": "   ",
    "numeric_int": 42,
    "numeric_float": 6.28,
}


def run_tests(n_runs: int = 20):
    print("=== Testing parse_value ===\n")

    for name, expr in TEST_CASES.items():
        results = defaultdict(int)

        for seed in range(n_runs):
            val = lps_ship.ShipCatalog._parse_value(expr, seed)
            results[val] += 1

        print(f"Test: {name}")
        print(f"Input: {expr}")
        for k, v in sorted(results.items(), key=lambda x: str(x[0])):
            percent = 100.0 * v / n_runs
            print(f"  {k}: {percent:6.1f}% ({v}/{n_runs})")
        print("-" * 40)


if __name__ == "__main__":
    run_tests()
