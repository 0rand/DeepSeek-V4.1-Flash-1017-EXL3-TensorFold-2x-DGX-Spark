# Attribution

The file `tests/fixtures/http.py.txt` is an unchanged copy of `src/tensorfold/cuda/http.py` from
TensorFold (https://github.com/ashhart/TensorFold) at revision `07aa02585f64cfb04c60a96de6554c0d4708ed46`.
TensorFold is licensed under Apache-2.0, as is this recipe (see the repository's `LICENSE`). The installer tests
need the exact bytes of that file, because `apply.py` checks its SHA-256 value before it changes anything.
The file keeps its original headers. The other file in `tests/fixtures/` (`parser.py`) is written for these tests.

The installer `apply.py` and the overlay `overlay/tool_output.py` are written for this recipe.
