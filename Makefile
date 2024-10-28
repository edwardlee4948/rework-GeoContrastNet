main ?= main

run:
	docker run --gpus=all -it --rm pytorch-cu118-pip python $(main).py


