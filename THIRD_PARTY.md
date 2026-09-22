# Third-party code

`src/semif_phase1/` (`__init__.py`, `core.py`, `direct.py`) is vendored
verbatim from **SemIf** by Theo C. J. Lee, <https://github.com/TheoLeeCJ/SemIf>,
under the MIT licence — reproduced in [LICENSE.upstream](LICENSE.upstream).

It is vendored rather than reimplemented on purpose: keeping the scoring path
byte-identical is what makes the vision results in this repo comparable to the
published text ones. SIV's additions (`src/semif_vl.py`, `src/cascade.py`,
`src/bench_cpu4b.py`, `scripts/`) import that engine and do not modify it.

The CPU loader shim that this work started from is **JEV-CPU** by Meanblock,
<https://huggingface.co/Meanblock/JEV-CPU>, also MIT.

Images in `data/img/` are from **COCO val2017** (<https://cocodataset.org>),
annotations under CC BY 4.0; the photographs are from Flickr and retain their
original licences.
