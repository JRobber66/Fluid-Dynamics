"""pixel_flow: rearrange the pixels of image B into image A via a fluid simulation.

Pipeline:
    preprocess  -> load A and B, downsize the larger to match the smaller
    pdf_export  -> write the images (and the optimal rearrangement) to a PDF
    assignment  -> Hungarian algorithm: optimal destination for every pixel of B
    fluid_sim   -> incompressible fluid sim moves each pixel to its destination in <= 10 s
"""

__version__ = "1.0.0"
