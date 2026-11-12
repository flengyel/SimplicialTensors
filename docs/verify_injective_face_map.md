# Distinct surviving coordinate images

[Source](../experiments/verify_injective_face_map.py) restricts source indices to tuples avoiding a fixed face label. It constructs the corresponding coordinate tensors, applies that face, and checks flattened images for duplicates.

```sh
mkdir -p logs
python experiments/verify_injective_face_map.py
```

The program logs to the console and overwrites `logs/injectivity_verification.log`. Its configured shapes include arrays through six axes, with every face index checked. Larger cases repeatedly allocate full arrays and can require substantial time and memory.

A smaller check is

```python
from experiments.verify_injective_face_map import verify_face_map_injectivity

assert verify_face_map_injectivity((3, 4, 5), 1)
```

It has 24 surviving indices, each mapping to a distinct coordinate unit in shape `(2, 3, 4)`. This establishes injectivity on the surviving coordinate basis. The full face map also kills coordinate tensors containing the deleted label.
