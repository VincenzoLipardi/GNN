import os
import sys
import pickle
import shutil
from typing import List, Tuple, Optional


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def list_pkls(directory: str) -> List[str]:
    try:
        return sorted([n for n in os.listdir(directory) if n.endswith('.pkl')])
    except Exception:
        return []


def backup_file(path: str) -> str:
    backup_path = path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
    return backup_path


def fix_pair(src_dir: str, oslo_dir: str, filename: str) -> Tuple[int, int]:
    src_path = os.path.join(src_dir, filename)
    oslo_path = os.path.join(oslo_dir, filename)

    if not (os.path.exists(src_path) and os.path.exists(oslo_path)):
        return 0, 0

    src_data = load_pickle(src_path)
    oslo_data = load_pickle(oslo_path)

    if not isinstance(src_data, list) or not isinstance(oslo_data, list):
        return 0, 0
    if len(src_data) != len(oslo_data):
        # Mismatch; skip to be safe
        return 0, 0

    backup_file(oslo_path)

    replaced = 0
    total = len(oslo_data)
    new_data = []
    for i in range(total):
        src_item = src_data[i]
        oslo_item = oslo_data[i]

        if not (isinstance(src_item, tuple) and len(src_item) >= 1):
            new_data.append(oslo_item)
            continue
        if not (isinstance(oslo_item, tuple) and len(oslo_item) >= 2):
            new_data.append(oslo_item)
            continue

        src_info = src_item[0]
        oslo_info, label = oslo_item[0], oslo_item[1]

        if not (isinstance(src_info, dict) and isinstance(oslo_info, dict)):
            new_data.append(oslo_item)
            continue

        # Replace only the 'qasm' field; keep all other metadata and the label
        if 'qasm' in src_info and 'qasm' in oslo_info:
            if oslo_info.get('qasm') != src_info.get('qasm'):
                updated_info = dict(oslo_info)
                updated_info['qasm'] = src_info['qasm']
                new_data.append((updated_info, label))
                replaced += 1
            else:
                new_data.append(oslo_item)
        elif 'qasm' in src_info and 'qasm' not in oslo_info:
            updated_info = dict(oslo_info)
            updated_info['qasm'] = src_info['qasm']
            new_data.append((updated_info, label))
            replaced += 1
        else:
            new_data.append(oslo_item)

    if replaced > 0:
        save_pickle(oslo_path, new_data)
    return replaced, total


def fix_directory_pair(src_dir: str, oslo_dir: str) -> Tuple[int, int, int]:
    names = sorted(set(list_pkls(src_dir)).intersection(list_pkls(oslo_dir)))
    files_done = 0
    entries_replaced = 0
    entries_total = 0
    for name in names:
        rep, tot = fix_pair(src_dir, oslo_dir, name)
        if tot > 0:
            files_done += 1
            entries_replaced += rep
            entries_total += tot
            print(f"[fixed] {name}: replaced {rep}/{tot}")
        else:
            print(f"[skip] {name}: incompatible or missing")
    return files_done, entries_replaced, entries_total


def _qasm_from_circuit(circ) -> Optional[str]:
    # Prefer QuantumCircuit.qasm() when available; fallback to qasm2.dumps
    try:
        qasm_method = getattr(circ, 'qasm', None)
        if callable(qasm_method):
            return qasm_method()
    except Exception:
        pass
    try:
        from qiskit.qasm2 import dumps as qasm2_dumps
        return qasm2_dumps(circ)
    except Exception:
        return None


def _transpile_to_oslo_qasm(qasm_str: str, optimization_level: int = 3) -> Optional[str]:
    try:
        from qiskit import QuantumCircuit as _QuantumCircuit
        from qiskit import transpile as _transpile
        try:
            from qiskit_ibm_runtime.fake_provider import FakeOslo as _FakeOslo
        except Exception:
            from qiskit.providers.fake_provider import FakeOslo as _FakeOslo
    except Exception:
        return None

    try:
        circ = _QuantumCircuit.from_qasm_str(qasm_str)
    except Exception:
        return None

    # Transpile exactly as in noise_simulation.py: use backend=FakeOslo with optimization_level
    try:
        backend = _FakeOslo()
        native = _transpile(circ, backend=backend, optimization_level=optimization_level)
    except Exception:
        native = circ
    return _qasm_from_circuit(native)


def transpile_directory(oslo_dir: str, limit_files: Optional[int] = None) -> Tuple[int, int, int]:
    names_all = list_pkls(oslo_dir)
    names = names_all if limit_files is None else names_all[: int(max(0, limit_files))]
    files_done = 0
    entries_replaced = 0
    entries_total = 0
    for name in names:
        path = os.path.join(oslo_dir, name)
        try:
            data = load_pickle(path)
        except Exception:
            print(f"[skip] {name}: failed to load")
            continue
        if not isinstance(data, list) or not data:
            print(f"[skip] {name}: unexpected data format")
            continue

        backup_file(path)

        replaced = 0
        total = len(data)
        new_data = []
        for item in data:
            if not (isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], dict)):
                new_data.append(item)
                continue
            info, label = item[0], item[1]
            qasm_old = info.get('qasm') if isinstance(info, dict) else None
            if not isinstance(qasm_old, str):
                new_data.append(item)
                continue
            qasm_new = _transpile_to_oslo_qasm(qasm_old)
            if isinstance(qasm_new, str) and qasm_new and qasm_new != qasm_old:
                updated_info = dict(info)
                updated_info['qasm'] = qasm_new
                new_data.append((updated_info, label))
                replaced += 1
            else:
                new_data.append(item)

        if replaced > 0:
            save_pickle(path, new_data)
        files_done += 1
        entries_replaced += replaced
        entries_total += total
        print(f"[transpiled] {name}: replaced {replaced}/{total}")

    return files_done, entries_replaced, entries_total


def main():
    # Modes:
    # 1) Copy qasm from noiseless to oslo: python fix_oslo_qasm.py <noiseless_src_dir> <oslo_target_dir>
    # 2) Transpile and overwrite qasm in oslo: python fix_oslo_qasm.py --transpile <oslo_target_dir> [--limit N]
    args = sys.argv[1:]
    if not args:
        print("Usage:\n  python fix_oslo_qasm.py <noiseless_src_dir> <oslo_target_dir>\n  python fix_oslo_qasm.py --transpile <oslo_target_dir> [--limit N]")
        sys.exit(1)

    if args[0] == '--transpile':
        if len(args) < 2:
            print("Usage: python fix_oslo_qasm.py --transpile <oslo_target_dir> [--limit N]")
            sys.exit(1)
        oslo_dir = args[1]
        limit: Optional[int] = None
        if len(args) >= 4 and args[2] == '--limit':
            try:
                limit = int(args[3])
            except Exception:
                limit = None
        files_done, entries_replaced, entries_total = transpile_directory(oslo_dir, limit_files=limit)
        print(f"Done (transpile). Files processed: {files_done}. Entries replaced: {entries_replaced}/{entries_total}.")
        return

    if len(args) == 2:
        src_dir = args[0]
        oslo_dir = args[1]
        files_done, entries_replaced, entries_total = fix_directory_pair(src_dir, oslo_dir)
        print(f"Done. Files updated: {files_done}. Entries replaced: {entries_replaced}/{entries_total}.")
        return

    print("Usage:\n  python fix_oslo_qasm.py <noiseless_src_dir> <oslo_target_dir>\n  python fix_oslo_qasm.py --transpile <oslo_target_dir> [--limit N]")
    sys.exit(1)


if __name__ == '__main__':
    main()


