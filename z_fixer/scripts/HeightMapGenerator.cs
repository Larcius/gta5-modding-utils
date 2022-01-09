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
		private const int numAngleSteps = 8;
		private const double angleStep = 2d * Math.PI / numAngleSteps;
		private static Vector2[] unitCircle = new Vector2[numAngleSteps];
		static HeightMapGenerator() {
			for (int i = 0; i < numAngleSteps; i++) {
				unitCircle[i] = new Vector2((float) Math.Sin(angleStep * i), (float) Math.Cos(angleStep * i));
			}
		}

		private const float maxCoordinate = 10000f;
		private const int maxHeight = 1024;
		private const int maxRadiusSteps = 20;
		private const float radiusStepSize = 0.5f;

		public StreamWriter writer;
		private StreamReader reader;

		private string filenameInput = "Full path (e.g. C:\\<SOME PATH>\\z_fixer\\generated\\coords.txt)";
		
		public bool startRequested, abortRequested;

		private int stepSize;

		private float prevZ;
		
		private Vector2 coords;

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

				if (splitted.Length != 2) {
					writer.Write("ERROR: malformed line: ");
					writer.WriteLine(line);
					continue;
				}

				float x = 0, y = 0;
				if (!Single.TryParse(splitted[0], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out x) ||
						!Single.TryParse(splitted[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out y)) {

					writer.Write("ERROR: malformed coordinates: ");
					writer.WriteLine(line);
					continue;
				}

				if (x <= -maxCoordinate || x >= maxCoordinate || y <= -maxCoordinate || y >= maxCoordinate) {
					writer.Write("ERROR: coordinates out of range: ");
					writer.WriteLine(line);
					continue;
				}

				coords = new Vector2(x, y);
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
			/*
			unsafe
			{
				float result = 0;
				if (!Function.Call<bool>(Hash.GET_GROUND_Z_FOR_3D_COORD, x, y, (float) maxHeight, &result)) {
					result = Single.NaN;//World.GetGroundHeight(new Vector3(x, y, (float) maxHeight));
				}
				return result;
			}
			*/
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
				float z = Single.NaN;
				if (stepSize > 0) {
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

				writer.Write(z.ToString(System.Globalization.CultureInfo.InvariantCulture));
				writer.Write(",");
				writer.Write(z.ToString(System.Globalization.CultureInfo.InvariantCulture));
			
				float min = z, max = z;
				for (int radiusStep = 1; radiusStep <= maxRadiusSteps; radiusStep++)
				{
					float radius = radiusStepSize * radiusStep;
					for (int i = 0; i < numAngleSteps; i++)
					{
						float curZ = getHeight(coords.X + unitCircle[i].X * radius, coords.Y + unitCircle[i].Y * radius);

						if (!Single.IsNaN(curZ)) {
							min = Math.Min(min, curZ);
							max = Math.Max(max, curZ);
						}
					}

					writer.Write(";");
					writer.Write(min.ToString(System.Globalization.CultureInfo.InvariantCulture));
					writer.Write(",");
					writer.Write(max.ToString(System.Globalization.CultureInfo.InvariantCulture));
				}

				writer.WriteLine("");
			}

			if (readCoords()) {
				// prepare for next step
				stepSize = maxHeight;
				setPlayerPosition(coords.X, coords.Y, maxHeight);
			} else {
				finish();
			}
		}
	}
}
