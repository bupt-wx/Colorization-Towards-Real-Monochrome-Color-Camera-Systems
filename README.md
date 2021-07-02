# Colorization Towards Real Monochrome-Color Camera Systems
Colorization in monochrome-color camera systems aims to colorize the gray image from the monochrome camera using the color image from the color camera as reference. Since 
monochrome cameras have better imaging quality than color cameras, the colorization can help obtain higher quality color images. Related learning based methods usually simulate 
the monochrome-color camera systems to generate the synthesized data for training, due to the lack of ground-truth color information of the gray image in the real data. However,
the methods that are trained relying on the synthesized data may get poor results when colorizing real data, because the synthesized data may deviate from the real data. We
present a self-supervised CNN model, named Cycle CNN, which can directly use the real data from monochrome-color camera systems for training.  <br><br>
The example results of gray image are shown in the figure below.<br>
![图片](https://user-images.githubusercontent.com/84729271/123550440-085d3f00-d7a0-11eb-8f4b-72dd6569c801.png)<br><br>
Papers:<br>
[Cycle-CNN for Colorization towards Real Monochrome-Color Camera Systems.pdf](https://github.com/bupt-wx/Colorization-Towards-Real-Monochrome-Color-Camera-Systems/files/6731688/Cycle-CNN.for.Colorization.towards.Real.Monochrome-Color.Camera.Systems.pdf)<br>
[Self-Supervised Colorization towards Monochrome-Color Camera Systems Using Cycle CNN.pdf](https://github.com/bupt-wx/Colorization-Towards-Real-Monochrome-Color-Camera-Systems/files/6751913/Self-Supervised.Colorization.towards.Monochrome-Color.Camera.Systems.Using.Cycle.CNN.pdf)
<br><br>

Clone the repository.<br>
`git clone https://github.com/bupt-wx/Colorization-Towards-Real-Monochrome-Color-Camera-Systems.git`<br>
Required environment version information.<br>
`Tensorflow 1.8.0; Keras 2.1.6; Python 3.6`<br><br>
You can test this project by using the following commands and using the images in the Sample_input folder.It should be noted that the algorithm uses the Ycbcr color space, and the pre-processing and post-processing of the algorithm requires converting the color space of the image.<br>
`python cycle_colorization_test.py -fpath test_input_file_path -outpath test_output_file_path`<br>
Please replace "test_input_file_path" with the input image path to be tested and "test_output_file_path" with the output image path after testing.The results should match the images in the Sample_out folder.
