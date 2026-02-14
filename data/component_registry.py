from typing import Dict, List, Final, Iterator, Sequence
import numpy as np

class ComponentRegistry(Sequence[str]):
    def __init__(self, components: List[str]):
        self.components: Final = tuple(components)
        self.index: Final = {c: i for i, c in enumerate(self.components)}
        self.nc: Final = len(self.components)

    def idx(self, component: str) -> int:
        try:
            return self.index[component]
        except KeyError:
            raise ValueError(f"Component '{component}' not registered")

    def validate(self, component: str) -> None:
        if component not in self.index:
            raise ValueError(f"Unknown component: {component}")

    def vector(self, values: Dict[str, float]) -> np.ndarray:
        """Convert dict {component: value} → ordered numpy array"""
        vec = np.zeros(self.nc)
        for c, v in values.items():
            vec[self.idx(c)] = v
        return vec

    # --- Sequence / iterable sugar ---
    def __len__(self) -> int:
        return self.nc

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def __getitem__(self, i: int) -> str:
        return self.components[i]