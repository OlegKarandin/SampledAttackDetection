
all: raylib_run

register_environments:
	poetry run python ./register_environments.py

raylib_run:
	poetry run python ./raylib_implementation.py

clean:
	rm ./logs/*
