# Cycle-CNN for Colorization towards Real Monochrome-Color Camera Systems
 Xuan Dong, Weixin Li, Xiaojie Wang, Yunhong Wang. In AAAI, 2020<br><br>
This is the implementation code of AAAI2020's paper "Cycle-CNN for Colorization towards Real Monochrome-Color Camera Systems".The example coloring result of gray image is shown in the figure below.<br>
![图片](https://user-images.githubusercontent.com/84729271/123550440-085d3f00-d7a0-11eb-8f4b-72dd6569c801.png)<br><br>
Download the original paper.<br>
[Cycle-CNN for Colorization towards Real Monochrome-Color Camera Systems.pdf](https://github.com/bupt-wx/Colorization-Towards-Real-Monochrome-Color-Camera-Systems/files/6731688/Cycle-CNN.for.Colorization.towards.Real.Monochrome-Color.Camera.Systems.pdf)<br><br>
Clone the repository.<br>
`git clone https://github.com/bupt-wx/Colorization-Towards-Real-Monochrome-Color-Camera-Systems.git`<br>
Required environment version information.<br>
`Tensorflow 1.8.0; Keras 2.1.6; Python 3.6`<br><br>
You can test this project by using the following commands and using the images in the Sample_input folder.It should be noted that the algorithm uses the Ycbcr color space, and the pre-processing and post-processing of the algorithm requires converting the color space of the image.<br>
`python cycle_colorization_test.py -fpath test_input_file_path -outpath test_output_file_path`<br>
Please replace "test_input_file_path" with the input image path to be tested and "test_output_file_path" with the output image path after testing.The results should match the images in the Sample_out folder.
