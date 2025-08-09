import os

def count_lines(path):
    with open(path, "rb") as f:
        lines = 0
        for _ in f:
            lines += 1
    return lines

if __name__ == "__main__":
    csv_path = "data/ratings.csv"
    # if file has header, subtract 1
    n = count_lines(csv_path)
    print("Total lines (incl header):", n)
    total_rows = n - 1
    print("Data rows:", total_rows)