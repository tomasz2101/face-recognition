SHELL := /bin/bash

init:
	virtualenv .venv
	source .venv/bin/activate && pip3 install -r requirements.txt