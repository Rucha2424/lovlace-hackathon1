import numpy as np

SLOT_TIME_SEC = 0.0005
BUFFER_TIME_SEC = 143e-6

def load_throughput(cell_id):
    values = []
    with open(f'raw_dat/throughput-{cell_id}.dat') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('<'):
                continue
            values.append(float(line.split()[1]))
    return values

def build_cell_signal():
    cell_signal = {}

    for i in range(1, 25):
        cell_id = f'cell-{i}'
        pkt_file = f'raw_dat/pkt-stats-{cell_id}.dat'
        thr_file = f'raw_dat/throughput-{cell_id}.dat'

        pkt_late = []
        throughput = []

        with open(pkt_file) as f:
            for line in f:
                if not line.strip() or line.startswith('<'):
                    continue
                pkt_late.append(int(line.split()[3]))

        with open(thr_file) as f:
            for line in f:
                if not line.strip() or line.startswith('<'):
                    continue
                throughput.append(float(line.split()[1]))

        L = min(len(pkt_late), len(throughput))
        pkt_late = pkt_late[:L]
        throughput = throughput[:L]

        thr_threshold = np.percentile(throughput, 95)

        signal = [
            1 if (pkt_late[j] > 0 or throughput[j] > thr_threshold) else 0
            for j in range(L)
        ]

        cell_signal[cell_id] = signal

    return cell_signal

def infer_topology(cell_signal, threshold=0.4):
    cells = list(cell_signal.keys())
    corr_map = {}

    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            c1 = cell_signal[cells[i]]
            c2 = cell_signal[cells[j]]

            if np.std(c1) == 0 or np.std(c2) == 0:
                corr = 0
            else:
                corr = np.corrcoef(c1, c2)[0, 1]

            corr_map[(cells[i], cells[j])] = corr

    links = []
    used = set()

    for cell in cells:
        if cell in used:
            continue

        group = [cell]
        used.add(cell)

        for other in cells:
            if other in used or other == cell:
                continue

            key = (cell, other) if (cell, other) in corr_map else (other, cell)
            if corr_map.get(key, 0) > threshold:
                group.append(other)
                used.add(other)

        links.append(group)

    return links

def compute_capacity(links):
    results = {}

    for idx, cells in enumerate(links, start=1):
        throughputs = [load_throughput(c) for c in cells]
        min_len = min(len(t) for t in throughputs)
        throughputs = [t[:min_len] for t in throughputs]

        aggregated = [
            sum(t[i] for t in throughputs)
            for i in range(min_len)
        ]

        peak_bits = np.percentile(aggregated, 99)
        cap_no_buffer = (peak_bits / SLOT_TIME_SEC) / 1e9

        buffer_bits = BUFFER_TIME_SEC * cap_no_buffer * 1e9
        effective_peak = max(peak_bits - buffer_bits, 0)
        cap_with_buffer = (effective_peak / SLOT_TIME_SEC) / 1e9

        results[f'Link_{idx}'] = {
            'cells': cells,
            'capacity_no_buffer_Gbps': round(cap_no_buffer, 3),
            'capacity_with_buffer_Gbps': round(cap_with_buffer, 3)
        }

    return results
