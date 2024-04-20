
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv tests/*.py

format:
	black development/*.py

lint: 
	pylint --disable=C=103,C0114,C0115,C0116 development/*.py

