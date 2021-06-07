# define the name of the virtual environment directory
VENV := $(HOME)/.virtualenv/6_mmseg

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

train_continue:
	$(VENV)/bin/python3 ./pipeline/train_boulder_segmentation.py

evaluate:
	$(VENV)/bin/python3 ./pipeline/test_boulder_segmentation_development.py

test:
	$(VENV)/bin/python3 ./pipeline/test_boulder_segmentation.py --eval mIoU

real_test:
	$(VENV)/bin/python3 ./pipeline/live_segmentation.py


#clean:
#	rm -rf ./$(VENV)
#	rm -rf ./hubert.egg-info
#	find -name "*.pyc" -delete