# VERITAS -- VERified Interpretable Tsetlin ASsurance
# (Windows users without `make`: run the python commands shown in each rule.)

PY ?= python

.PHONY: help install test reproduce quick figures paper demo clean

help:
	@echo "make install    - editable install + experiment/dev extras"
	@echo "make test       - run the unit test suite (pytest)"
	@echo "make reproduce  - run every experiment, regenerate results/*.json"
	@echo "make quick      - fast 3-seed smoke of all experiments"
	@echo "make figures    - rebuild paper/figures/*.pdf from results/*.json"
	@echo "make paper      - compile paper/veritas.pdf"
	@echo "make demo       - run the quickstart demo"

install:
	$(PY) -m pip install -e ".[experiments,data,dev]"

test:
	$(PY) -m pytest -q

reproduce:
	$(PY) -u experiments/run_all.py

quick:
	$(PY) -u experiments/run_all.py --quick

figures:
	$(PY) scripts/make_figures.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode veritas.tex && bibtex veritas && pdflatex -interaction=nonstopmode veritas.tex && pdflatex -interaction=nonstopmode veritas.tex

demo:
	$(PY) demo/quickstart.py

clean:
	$(PY) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)]"
	cd paper && $(PY) -c "import glob,os; [os.remove(f) for e in ('aux','log','out','bbl','blg','toc') for f in glob.glob('*.'+e)]"
