# define the name of the virtual environment directory
VENV := /home/jschiffeler/.virtualenv/6_mmseg

# default target, when make executed without arguments
all: venv

#$(VENV)/bin/activate: requirements.txt
#	make -C ./libcluon
#	python3 -m venv $(VENV)
#	./$(VENV)/bin/pip install --upgrade pip
#	./$(VENV)/bin/pip install --editable .
#	./$(VENV)/bin/pip install -r requirements.txt

# venv is a shortcut target
venv: $(VENV)/bin/activate
export PYTHONPATH=$PYTHONPATH:./

train_fresh:
	$(VENV)/bin/python3 ./pipeline/train_boulder_segmentation.py -f

prepare_set:
	$(VENV)/bin/python3 ./boulderSet/transform_labels.py
	$(VENV)/bin/python3 ./boulderSet/create_split.py

evaluate:
	$(VENV)/bin/python3 ./pipeline/test_boulder_segmentation_development.py

clean:
	rm -rf ./$(VENV)
	rm -rf ./hubert.egg-info
	find -name "*.pyc" -delete