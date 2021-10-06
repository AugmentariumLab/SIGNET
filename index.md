## SIGNET: Efficient Neural Representations For Light Fields | Brandon Yushan Feng&nbsp;&nbsp;&nbsp;&nbsp;Amitabh Varshney International Conference on Computer Vision (ICCV 2021) - Oral

## Abstract

[![Teaser image of SIGNET](resources/teaser.png)](https://brandonyfeng.github.io/papers/SIGNET.pdf)

We present a novel neural representation for light field content that enables compact storage and easy local reconstruction with high fidelity. 
We use a fully-connected neural network to learn the mapping function between each light field pixel's coordinates and its corresponding color values.
Since neural networks that simply take in raw coordinates are unable to accurately learn data containing fine details,
we present an input transformation strategy based on the Gegenbauer polynomials, which previously showed theoretical advantages over the Fourier basis. 
We conduct experiments that show our Gegenbauer-based design combined with sinusoidal activation functions leads to a better light field reconstruction quality than a variety of network designs, including those with Fourier-inspired techniques introduced by prior works. Moreover, our SInusoidal Gegenbauer NETwork, or SIGNET, can represent light field scenes more compactly than the state-of-the-art compression methods while maintaining a comparable reconstruction quality. SIGNET also innately allows random access to encoded light field pixels due to its functional design. We further demonstrate that SIGNET's super-resolution capability without any additional training.

## Downloads

<div style="display: flex; text-align:center; flex-direction: row; flex-wrap: wrap;">
<div style="margin:1rem; flex-grow: 1;"><a href="https://brandonyfeng.github.io/papers/SIGNET.pdf"><img style="max-width:10rem;" src="resources/paper.jpg"><br><label>Paper</label></a><br></div>
<div style="margin:1rem; flex-grow: 1;"><a href="resources/SIGNET_supplementary.pdf"><img style="max-width:10rem;" src="resources/supplementary.jpg"><br>Supplementary</a></div>
<div style="margin:1rem; flex-grow: 1;"><a href="https://github.com/AugmentariumLab/SIGNET"><img style="max-width:10rem;" src="resources/github.jpg"><br>Code</a></div>
<div style="margin:1rem; flex-grow: 1;"><a href="https://docs.google.com/presentation/d/15iIS2_9XapnSUtHnTNXNibJ7aeYD9ZYEnJqey0AlB88"><img style="max-width:10rem;" src="resources/slides.jpg"><br>Slides</a></div>
</div>

