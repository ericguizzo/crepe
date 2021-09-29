import json
import tempfile
from pathlib import Path
import shutil

import cog
from crepe.core import *
import crepe
from scipy.io import wavfile

class Predictor(cog.Predictor):
    def setup(self):
        """Load the model"""
        s_weights = "/usr/local/lib/python3.7/site-packages/crepe/model-full.h5"
        d_weights = "/src/crepe/model-full.h5"
        if not os.path.exists(d_weights):
            shutil.move(s_weights, d_weights)

    @cog.input("input", type=Path, help="Audio file")
    @cog.input(
        "viterbi",
        type=bool,
        default=False,
        help="Apply viterbi smoothing to the estimated pitch curve",
    )
    @cog.input(
        "plot_voicing",
        type=bool,
        default=False,
        help="Include a visual representation of the voicing activity detection",
    )
    @cog.input(
        "step_size",
        type=int,
        default=10,
        help="The step size in milliseconds for running pitch estimation",
    )
    @cog.input(
        "output_type",
        type=str,
        default="plot",
        options=["plot", "json"],
        help="Type of output representation: could be plot or json (list of [time, frequency, confidence] values)",
    )
    def predict(self, input, viterbi, plot_voicing, step_size, output_type):
        """Compute f0 plot"""
        output_path = Path(tempfile.mkdtemp()) / "output.png"

        sr, audio = wavfile.read(input)
        time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=viterbi, step_size=step_size)

        f0_data = np.vstack([time, frequency, confidence]).transpose()

        if output_type == "plot":
            # save the salience visualization in a PNG file
            import matplotlib.cm
            from imageio import imwrite
            #plot_file = output_path(file, ".activation.png", output)
            # to draw the low pitches in the bottom
            salience = np.flip(activation, axis=1)
            inferno = matplotlib.cm.get_cmap('inferno')
            image = inferno(salience.transpose())
            if plot_voicing:
                # attach a soft and hard voicing detection result under the
                # salience plot
                image = np.pad(image, [(0, 20), (0, 0), (0, 0)], mode='constant')
                image[-20:-10, :, :] = inferno(confidence)[np.newaxis, :, :]
                image[-10:, :, :] = (
                    inferno((confidence > 0.5).astype(np.float))[np.newaxis, :, :])

            imwrite(output_path, (255 * image).astype(np.uint8))
            return output_path

        elif output_type == "json":
            out = f0_data.tolist()
            return json.dumps(out)
