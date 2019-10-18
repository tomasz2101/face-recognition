
init:
	virtualenv .venv
	source .venv/bin/activate
	pip3 install -r requirements.txt
start:
	source .venv/bin/activate
