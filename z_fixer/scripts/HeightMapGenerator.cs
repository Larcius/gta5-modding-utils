using GTA;
using GTA.Native;
using GTA.Math;
using System;
using System.Windows.Forms;
using System.IO;

namespace HeightMap
{
	public class HeightMapGenerator : Script
	{
		private const float maxCoordinate = 10000f;
		private const int maxHeight = 2048;

		// resolution (samples per meter)
		private const float resolution = 0.1f;

		public StreamWriter writer;
		private StreamReader reader;

		private string filenameInput = "Full path (e.g. C:\\<SOME PATH>\\z_fixer\\generated\\coords.txt)";

		public bool startRequested, abortRequested;

		private int stepSize;

		private float prevZ;

		private Vector3 coords;
		private Quaternion rotation;
		private float radius;

		public HeightMapGenerator()
		{
			Tick += OnTick;
			KeyDown += OnKeyDown;
		}

		private void OnKeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode != Keys.F10) return;

			if (writer == null) {
				string newFilenameInput = Game.GetUserInput(filenameInput, 255);
				if (newFilenameInput == null || newFilenameInput == "") {
					return;
				}
				filenameInput = newFilenameInput;
				string filenameOutput = filenameInput + ".hmap";
				try {
					reader = new StreamReader(filenameInput);
				} catch (IOException ex) {
					UI.Notify("invalid filename");
					return;
				}
				writer = File.CreateText(filenameOutput);
				abortRequested = false;
				startRequested = true;
			} else {
				abortRequested ^= true;
			}
		}

		private bool readCoords() {
			string line;
			while ((line = reader.ReadLine()) != null) {
				string[] splitted = line.Split(new char[] { '|', ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
				if (splitted.Length == 0) {
					continue;
				}

				if (splitted.Length != 8) {
					writer.Write("ERROR: malformed line: ");
					writer.WriteLine(line);
					continue;
				}

				float x, y, z, qx, qy, qz, qw, r;
				if (!Single.TryParse(splitted[0], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out x) ||
						!Single.TryParse(splitted[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out y) ||
						!Single.TryParse(splitted[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out z) ||
						!Single.TryParse(splitted[3], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out qx) ||
						!Single.TryParse(splitted[4], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out qy) ||
						!Single.TryParse(splitted[5], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out qz) ||
						!Single.TryParse(splitted[6], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out qw) ||
						!Single.TryParse(splitted[7], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out r)) {

					writer.Write("ERROR: malformed coordinates: ");
					writer.WriteLine(line);
					continue;
				}

				if (x <= -maxCoordinate || x >= maxCoordinate || y <= -maxCoordinate || y >= maxCoordinate) {
					writer.Write("ERROR: coordinates out of range: ");
					writer.WriteLine(line);
					continue;
				}

				if (Single.IsNaN(z) || z < 0) {
					z = 0;
				} else if (z > maxHeight) {
					z = maxHeight;
				}

				coords = new Vector3(x, y, z);
				rotation = new Quaternion(qx, qy, qz, qw);
				radius = Math.Abs(r);
				return true;
			}
			return false;
		}

		private void finish()
		{
			reader.Close();
			reader = null;

			writer.Flush();
			writer.Close();
			writer = null;
		}

		private float getHeight(float x, float y) {
			float result = World.GetGroundHeight(new Vector2(x, y));
			if (result == 0) {
				return Single.NaN;
			}
			return result;
		}

		private bool isApproxEqual(float f1, float f2) {
			return Math.Abs(f1 - f2) < 0.001f;
		}

		private void setPlayerPosition(float x, float y, float z) {
			prevZ = z;
			Game.Player.Character.Position = new Vector3(x, y, z);
		}

		private void OnTick(object sender, EventArgs e)
		{
			if (writer == null) {
				return;
			}

			if (abortRequested) {
				finish();
				return;
			}

			if (startRequested) {
				startRequested = false;
			} else {
				bool addWarning;
				float z;
				if (stepSize == 0) {
					addWarning = true;
					if (Single.IsNaN(coords.Z)) {
						z = 0f;
					} else {
						z = coords.Z;
					}
				} else {
					addWarning = false;
					z = getHeight(coords.X, coords.Y);

					if (Single.IsNaN(z)) {
						z = prevZ - stepSize;

						if (z <= 0) {
							stepSize /= 2;
							z = maxHeight - stepSize;
						} else if (isApproxEqual(z % (2f * stepSize), 0)) {
							z -= stepSize;
						}
					}
					if (stepSize == 0 || !isApproxEqual(prevZ, z)) {
						setPlayerPosition(coords.X, coords.Y, z);
						return;
					}
				}

				writer.Write(coords.X.ToString(System.Globalization.CultureInfo.InvariantCulture));
				writer.Write(",");
				writer.Write(coords.Y.ToString(System.Globalization.CultureInfo.InvariantCulture));
				writer.Write(",");
				writer.Write(z.ToString(System.Globalization.CultureInfo.InvariantCulture));
				writer.Write(",");

				float minZ = z;
				for (float curRadius = resolution; radius > 0 && curRadius < radius + resolution; curRadius += resolution) {
					// if curRadius is approx. or even greater than the max radius then set it to max radius
					if (curRadius >= radius - 0.001f) {
						curRadius = radius;
					}

					// calculate number of angle steps depending on the radius
					int numAngleSteps;
					if (resolution / curRadius > Math.Sqrt(3)) {
						// use at least 3 angle steps
						numAngleSteps = 3;
					} else {
						// using the formula for side-length of an regular n-polygon for a given outer radius:
						// side-length = 2 * outer-radius * sin(PI / n)
						// note that the previous check also ensured that the argument for arcsin is always less than 1
						// furthermore it is ensured that the result of arcsin is always strictly positive
						// since resolution and curRadius are always strictly positive
						numAngleSteps = (int) Math.Ceiling(Math.PI / (Math.Asin(resolution / curRadius / 2f)));
					}

					double angleStep = 2d * Math.PI / numAngleSteps;
					for (double angle = 0; angle < 2d * Math.PI; angle += angleStep) {
						Vector3 point = new Vector3((float) Math.Sin(angle) * curRadius, (float) Math.Cos(angle) * curRadius, 0f);
						point = rotation * point;

						float curZ = getHeight(coords.X + point.X, coords.Y + point.Y);

						if (!Single.IsNaN(curZ)) {
							minZ = Math.Min(minZ, curZ - point.Z);
						}
					}
				}

				writer.Write(minZ.ToString(System.Globalization.CultureInfo.InvariantCulture));

				if (addWarning) {
					writer.Write(",WARNING: could not get z coordinate so just used the one from the input file or 0 if NaN was given");
				}

				writer.WriteLine("");
			}

			if (readCoords()) {
				// prepare for next step
				stepSize = 2 * maxHeight;
				setPlayerPosition(coords.X, coords.Y, coords.Z);
			} else {
				finish();
			}
		}
	}
}
