
install:
	pip install --upgrade pip &&\
		pip instal l-r requirements.txt

test:
	python -m pytest -vv *.py

format:
	black *.py

lint: 
	pylint *.py

