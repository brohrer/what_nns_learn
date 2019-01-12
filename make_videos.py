import animation_tools as at

image_dirs = [
    "linear_1-layer_1-input_1-output_plots",
    "linear_1-layer_2-input_1-output_plots",
    "linear_2-layer_1-input_1-output_plots",
    "linear_3-layer_1-input_1-output_plots",
    "logistic_1-layer_1-input_1-output_plots",
    "logistic_1-layer_2-input_1-output_plots",
    "hyperbolic_tangent_1-layer_1-input_1-output_plots",
    "hyperbolic_tangent_2-layer_1-input_1-output_plots",
    "hyperbolic_tangent_2-layer_2-input_1-output_plots",
    "hyperbolic_tangent_3-layer_1-input_1-output_plots",
    "hyperbolic_tangent_3-layer_1-input_2-output_plots",
    "hyperbolic_tangent_3-layer_1-input_3-output_plots",
    "hyperbolic_tangent_3-layer_2-input_1-output_plots",
    "hyperbolic_tangent_3-layer_2-input_2-output_plots",
    "rectified_linear_units_3-layer_2-input_1-output_plots",
    "relu_3-layer_2-input_2-output_plots",
]

video_dirname = "videos"

for image_dir in image_dirs:
    filename = image_dir + ".mp4"
    at.render_movie(
        filename=filename,
        fps=1,
        frame_dirname=image_dir,
        output_dirname=video_dirname,
    )
    at.convert_to_gif(filename=filename, dirname=video_dirname)
