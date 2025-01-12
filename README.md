# cpp_facial_privacy_protection

## Project Introduction

* The project is divided into four modules
* The first module YuNet.cpp load the YuNet class, and we will load the trained YuNet model
* The second module config.cpp are some constants in the project
* The third module util.cpp some of the functions in the project
* In the main method module main.cpp, we will call the above three modules and write the call logic

## Dependency libraries

* This project mainly calls opencv4.10.0

## Function introduction

* This project contains a total of four functions, using '-mode' in the parameters to achieve function switching, and the mode can be switched in real time at runtime.
	* Normal Face Detection: Normal Mode (Button n)
	* Blur Processing: Blur Mode (Button b)
	* Pixelation: Pixel mode (button p)
	* Mask mode: Mask mode (button m)
* Parameter adjustments
	* Blurring:
		* Reduce Blur: Press button c
		* Increase Blur: press button v
	* Pixelation:
		* To reduce the pixel block size: press button c
		* To induce the pixel block size: press buttonv
	* mask：
		* After pressing the U button, the input is displayed in the command line, and the covered photo can be changed by entering the legal photo path

## Running instructions

### Command-line arguments

* -mode: set the initial mode(normal, blur, pixel, mask), default is mask
* -blur_kernel_size: takes effect only in blur mode. Set the initial blur kernel size, the default value is 5*5
* -pixel_size: Takes effect only in pixel mode. Sets the initial pixel block size, the default value is 10
* -mask: takes effect only in mask mode. Specify the path of the mask image, and the default image name is mask1.jpg.
* -device: specifies the device ID of the camera, which is 0 by default

### Run the sample

1. Default parameters

```shell
./main
```

2. Specify the parameter to run

```shell
./main -mode mask -mask /path/to/image.png
./main -mode normal -mask /path/to/image.png
./main -mode blur -mask /path/to/image.png -blur_kernel_size 5 
./main -mode pixel -mask /path/to/image.png -pixel_size 10
```



