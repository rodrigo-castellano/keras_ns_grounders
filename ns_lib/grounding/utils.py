from typing import List, Set, Tuple


class AtomIndex():
    def __init__(self, facts: List[Tuple[str, str, str]]):
        _index = {}
        for f in facts:
            r_idx, s_idx, o_idx = f
            key = (r_idx, None, None)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = (r_idx, s_idx, None)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = (r_idx,  None, o_idx)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = f
            _index[key] = set([f])

        # Store tuple instead of sets.
        self._index = {k: tuple(v) for k, v in _index.items()}

    def get_matching_atoms(self,
                           atom: Tuple[str, str, str]) -> Tuple[Tuple[str, str, str]]:
        return self._index.get(atom, [])


class AtomIndexDeterministic():
    def __init__(self, facts: List[Tuple[str, str, str]]):
        _index = {}
        self.is_deterministic = True
        for f in facts:
            r_idx, s_idx, o_idx = f
            key = (r_idx, None, None)
            if key not in _index:
                _index[key] = []  # Use a list
            _index[key].append(f)  # Use append

            key = (r_idx, s_idx, None)
            if key not in _index:
                _index[key] = []  # Use a list
            _index[key].append(f)  # Use append

            key = (r_idx, None, o_idx)
            if key not in _index:
                _index[key] = []  # Use a list
            _index[key].append(f)  # Use append

            key = f
            _index[key] = [f]  # Use a list

        self._index = {k: tuple(v) for k, v in _index.items()}  # Convert to tuple *after* ordering with lists

    def get_matching_atoms(self,
                           atom: Tuple[str, str, str]) -> Tuple[Tuple[str, str, str]]:
        return self._index.get(atom, [])


def get_atoms_on_groundings(groundings: Set[Tuple[Tuple, Tuple]]) -> Set[Tuple]:
    atoms = set()
    for rule_atoms in groundings:
        for atom in rule_atoms[0]:  # head
            atoms.add(atom)
        for atom in rule_atoms[1]:  # tail
            atoms.add(atom)
    return atoms
