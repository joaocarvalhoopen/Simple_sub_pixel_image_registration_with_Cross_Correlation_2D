# Simple sub pixel image registration with Cross_Correlation_2D
A simple, yet useful lib for experimentation. 

## Description
This is a lib and an example test for doing registration of one image B in relation to another image A, that have to have the same size. The registration that is done is only with regard to translational movements in XX and in YY axis, but it registers with sub pixel accuracy. And it registers in a fast way the two images. It also provides a function to shift the image with sub pixel precision. There is also in this code a small lib that is a wrapper over the Odin stb image bindings to load and save images in several image formats. <br>
This code was developed on Linux, bu tit probably will work with minimal modifications in other Odin supported operating systems. <br>
Note: The image is the famous iamge of Lena used historically in Computer Graphics. 

## Demo of image registration with 2 modes Normal and Rolling

![registration_demo_normal_vs_rolling.png](registration_demo_normal_vs_rolling.png)


## To compile this code you will need to compile stb image in the Odin diretory

```
# To compile stb :

$ cd Odin/vendor/stb/src
$ make

# To compile and run the project :

$ make clean
$ make opti
$ make run
```

## License
MIT Open Source License

## Have fun
Best regards, <br>
Joao Carvalho
