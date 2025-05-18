# Makefile
all:
	odin build . -out:register_image.exe --debug

opti:
	odin build . -out:register_image.exe -o:speed

opti_max:
	odin build . -out:register_image.exe -o:aggressive -microarch:native \
	-no-bounds-check -disable-assert -no-type-assert

clean:
	rm register_image.exe

run:
	./register_image.exe
